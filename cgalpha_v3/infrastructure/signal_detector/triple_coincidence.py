"""
CGAlpha v3 — Triple Coincidence Signal Detector (CORRECTED LOGIC)
=================================================================
Lógica correcta de la estrategia:
1. Detectar Triple Coincidence (vela clave + zona)
2. Rastrear zona activa
3. Detectar retest del precio a la zona
4. Capturar features en el momento del retest (VWAP, OBI, cumulative delta)
5. Observar outcome (rebote vs ruptura)
6. Generar dataset para entrenamiento del Oracle
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("detector")


# ─── Shadow Harvesting v1: Zone Lifecycle ────────────────────────────────────

from enum import Enum


class ZoneLifecycleState(str, Enum):
    """Estado de vida de una zona para el harvesting multi-toque."""

    ACTIVE = "active"  # Zona viva, esperando primer retest
    HARVESTING = "harvesting"  # Breakout confirmado, esperando 2do/3er toque
    EXHAUSTED = "exhausted"  # Zona expirada (3 toques o TTL vencido)


from dataclasses import dataclass as _dc
from dataclasses import field as _field


@_dc
class TouchRecord:
    """
    Registro inmutable de un toque a la zona.
    Se acumula en ActiveZone.touch_history.
    """

    touch_sequence: int  # 1 = primer retest, 2 = post-flip, 3 = terciario
    touch_ts_unix_ms: float
    touch_price: float
    obi_10_at_touch: float = 0.0
    cumulative_delta_at_touch: float = 0.0
    outcome: str = "PENDING"  # Relleno diferidamente por DeferredOutcomeMonitor
    polarity_flipped: bool = False  # True si la zona había cambiado de polaridad


# ─────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
# DATACLASSES PARA LÓGICA CORRECTA
# ──────────────────────────────────────────────────────────────


@dataclass
class ActiveZone:
    """Zona activa detectada por Triple Coincidence."""

    candle_index: int
    zone_top: float
    zone_bottom: float
    vwap_at_detection: float
    detection_timestamp: int
    direction: str  # 'bullish' or 'bearish'
    key_candle: Dict
    accumulation_zone: Dict
    mini_trend: Dict
    atr_at_detection: float
    max_price_since_detection: float = -1.0
    min_price_since_detection: float = 1e10
    max_price_since_last_touch: float = -1.0  # Reset en cada register_touch
    min_price_since_last_touch: float = 1e10  # Reset en cada register_touch
    retest_detected: bool = False  # Legado — usar lifecycle_state en su lugar
    zone_id: str = ""

    # ── Shadow Harvesting: Zone Lifecycle ──────────────────────────────────
    lifecycle_state: ZoneLifecycleState = ZoneLifecycleState.ACTIVE
    touch_count: int = 0
    touch_history: list = None  # list[TouchRecord] — None → init en __post_init__
    polarity_flipped: bool = False
    flip_ts: float = None  # Unix timestamp del Breakout confirmado
    flip_price: float = None  # Precio donde se confirmó el Breakout
    harvest_expiry_ts: float = None  # TTL post-flip
    HARVEST_TTL_SECONDS: float = 172800  # 48 horas
    MAX_TOUCHES: int = 3

    def __post_init__(self):
        if self.touch_history is None:
            object.__setattr__(self, "touch_history", [])

    def register_breakout(self, breakout_price: float) -> None:
        """
        Transiciona la zona de ACTIVE → HARVESTING en lugar de eliminarla.
        La polaridad se invierte: soporte previo se convierte en resistencia
        y viceversa (Support/Resistance Flip).
        """
        import time as _time

        self.polarity_flipped = True
        self.flip_ts = _time.time()
        self.flip_price = breakout_price
        self.harvest_expiry_ts = self.flip_ts + self.HARVEST_TTL_SECONDS
        self.lifecycle_state = ZoneLifecycleState.HARVESTING

    def register_touch(
        self, price: float, obi: float = 0.0, cum_delta: float = 0.0
    ) -> int:
        """
        Registra un toque y retorna el touch_sequence asignado.
        Incrementa touch_count; si alcanza MAX_TOUCHES, marca EXHAUSTED.
        """
        import time as _time

        self.touch_count += 1
        seq = self.touch_count
        rec = TouchRecord(
            touch_sequence=seq,
            touch_ts_unix_ms=_time.time() * 1000,
            touch_price=price,
            obi_10_at_touch=obi,
            cumulative_delta_at_touch=cum_delta,
            polarity_flipped=self.polarity_flipped,
        )
        self.touch_history.append(rec)
        # Reset per-touch extremes for clearance tracking of NEXT touch
        self.max_price_since_last_touch = price
        self.min_price_since_last_touch = price
        if self.touch_count >= self.MAX_TOUCHES:
            self.lifecycle_state = ZoneLifecycleState.EXHAUSTED
        return seq

    def is_exhausted(self) -> bool:
        """
        Una zona está agotada cuando:
          - Alcanzó MAX_TOUCHES, o
          - El TTL post-flip expiró (48h por defecto)
        """
        import time as _time

        if self.lifecycle_state == ZoneLifecycleState.EXHAUSTED:
            return True
        if self.harvest_expiry_ts is not None and _time.time() > self.harvest_expiry_ts:
            self.lifecycle_state = ZoneLifecycleState.EXHAUSTED
            return True
        return False

    @property
    def effective_direction(self) -> str:
        """
        Dirección real post-flip para evaluar retests secundarios.
        Si la zona fue bullish y rompió → ahora actúa como resistencia bearish.
        """
        if not self.polarity_flipped:
            return self.direction
        return "bearish" if self.direction == "bullish" else "bullish"

    @property
    def prior_outcomes(self) -> list:
        """Lista de outcomes ya resueltos para contexto del 2do/3er toque."""
        return [
            r.outcome for r in self.touch_history if r.outcome not in ("PENDING", "")
        ]

    retest_index: Optional[int] = None
    outcome: Optional[str] = None  # 'BOUNCE' | 'BREAKOUT' | 'PENDING'
    last_retest_ts_ms: Optional[int] = (
        None  # Debounce: last tick-level retest timestamp
    )


@dataclass
class RetestEvent:
    """Evento de retest del precio a una zona activa."""

    zone: ActiveZone
    retest_index: int
    retest_price: float
    retest_timestamp: int
    # Features capturadas en el momento del retest
    vwap_at_retest: float
    obi_10_at_retest: float
    cumulative_delta_at_retest: float
    delta_divergence: str
    atr_14: float
    regime: str
    # Outcome observado después del retest
    outcome: Optional[str] = None  # 'BOUNCE' | 'BREAKOUT'
    outcome_confidence: Optional[float] = None


@dataclass
class TrainingSample:
    """Sample de entrenamiento para el Oracle."""

    features: Dict[str, Any]  # VWAP, OBI, delta, etc. en retest
    outcome: str  # 'BOUNCE' | 'BREAKOUT'
    zone_id: str
    retest_timestamp: int


# ──────────────────────────────────────────────────────────────
# COMPONENTE 1 — Detector de Velas Clave
# ──────────────────────────────────────────────────────────────


class KeyCandleDetector:
    """
    Detector de velas clave para la estrategia Triple Coincidence.
    Identifica velas con alto volumen relativo y cuerpo pequeño.
    """

    def __init__(self, config: Dict = None):
        defaults = {
            "volume_percentile_threshold": 70,
            "body_percentage_threshold": 40,
            "lookback_candles": 30,
        }
        self.config = {**defaults, **(config or {})}
        self.data = None

    def load_data(self, data: pd.DataFrame):
        """Carga datos OHLCV."""
        self.data = data.copy()

    def detect(self, index: int) -> Optional[Dict]:
        """
        Evalúa si la vela en `index` es una vela clave.
        Calcula Z-Score de volumen relativo y morfología observacional.
        """
        if self.data is None or index < self.config["lookback_candles"]:
            return None

        bpt = self.config["body_percentage_threshold"]
        lookback = self.config["lookback_candles"]

        # ── VOLUMEN ADAPTATIVO (Z-Score) ─────────────────────────
        vol_history = self.data["volume"].iloc[index - lookback : index].values
        mean_vol = np.mean(vol_history)
        std_vol = np.std(vol_history)
        current_vol = self.data["volume"].iloc[index]

        # Z-Score local (evita rigidez de percentiles globales)
        z_score = (current_vol - mean_vol) / std_vol if std_vol > 0 else 0

        # Umbral configurable (default 0.5 para capturar velas como las 06:10)
        # Nota: En v3 se usa volume_percentile_threshold de forma legacy si no hay z_threshold
        z_threshold = self.config.get("volume_z_threshold", 0.5)
        is_high_vol = z_score >= z_threshold

        current = self.data.iloc[index]
        body_size = abs(current["close"] - current["open"])
        candle_range = current["high"] - current["low"]

        if candle_range == 0:
            return None

        body_pct = 100 * body_size / candle_range
        is_small_body = body_pct <= bpt

        # ── MORFOLOGÍA OBSERVACIONAL ─────────────────────────────
        upper_wick = current["high"] - max(current["open"], current["close"])
        lower_wick = min(current["open"], current["close"]) - current["low"]

        upper_wick_pct = 100 * upper_wick / candle_range
        lower_wick_pct = 100 * lower_wick / candle_range
        rejection = "NONE"
        if upper_wick_pct > 40 and lower_wick_pct > 40:
            rejection = "BOTH"
        elif upper_wick_pct > 40:
            rejection = "UP"
        elif lower_wick_pct > 40:
            rejection = "DOWN"

        if is_high_vol and is_small_body:
            return {
                "index": index,
                "open": float(current["open"]),
                "high": float(current["high"]),
                "low": float(current["low"]),
                "close": float(current["close"]),
                "volume": float(current["volume"]),
                "vol_z_score": float(z_score),
                "body_percentage": float(body_pct),
                "upper_wick_pct": float(upper_wick_pct),
                "lower_wick_pct": float(lower_wick_pct),
                "rejection_direction": rejection,
                "timestamp": int(current.get("open_time", index)),
            }
        return None

    def detect_all(self) -> List[Dict]:
        """Detecta todas las velas clave en el dataset."""
        if self.data is None:
            return []

        key_candles = []
        for idx in range(self.config["lookback_candles"], len(self.data)):
            candle = self.detect(idx)
            if candle:
                key_candles.append(candle)
        return key_candles


# ──────────────────────────────────────────────────────────────
# COMPONENTE 2 — Detector de Zona de Acumulación
# ──────────────────────────────────────────────────────────────


class AccumulationZoneDetector:
    """
    Detecta zonas de acumulación (price consolidation + volume)
    anteriores a las velas clave.
    """

    def __init__(self, config: Dict = None):
        defaults = {
            "atr_period": 14,
            "atr_multiplier": 1.5,
            "volume_threshold": 1.2,
            "min_zone_bars": 5,
            "quality_threshold": 0.7,
        }
        self.config = {**defaults, **(config or {})}
        self.data = None

    def load_data(self, data: pd.DataFrame):
        """Carga datos OHLCV."""
        self.data = data.copy()
        # Asegurar columnas numéricas
        for col in ["open", "high", "low", "close", "volume"]:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce")

    def _calculate_atr(self, index: int) -> float:
        """Calcula ATR en el índice dado (optimizado con numpy)."""
        period = self.config["atr_period"]
        if index < period:
            return self.data["close"].iloc[index] * 0.01

        high = self.data["high"].values[index - period : index]
        low = self.data["low"].values[index - period : index]
        close = self.data["close"].values[index - period : index]
        close_prev = np.roll(close, 1)
        close_prev[0] = close[0]
        tr = np.maximum(
            high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev))
        )
        return float(np.mean(tr))

    def _quality_score(
        self,
        start: int,
        end: int,
        range_width: float,
        avg_vol: float,
        vwap: float,
        mfi: float,
    ) -> float:
        """
        Calcula el quality_score de la zona (0-1).
        """
        try:
            atr = self._calculate_atr(end)
            if atr == 0:
                atr = self.data["close"].iloc[end] * 0.01

            vol_pct = np.percentile(
                self.data["volume"].values[max(0, start - 150) : end], 65
            )

            # Criterio 1: rango estrecho
            range_score = 1 - min(
                range_width / (self.config["atr_multiplier"] * atr * 1.5), 1
            )

            # Criterio 2: volumen
            vol_score = (
                max(0.3, min(avg_vol / (vol_pct * 0.8), 1)) if vol_pct > 0 else 0.3
            )

            # Criterio 3: precio ≈ VWAP
            vwap_score = (
                1.0 if abs(self.data["close"].iloc[end] - vwap) / vwap <= 0.03 else 0.6
            )

            # Criterio 4: MFI en rango neutro
            mfi_score = 1.0 if 30 <= mfi <= 70 else 0.6

            quality = (
                0.35 * range_score
                + 0.35 * vol_score
                + 0.15 * vwap_score
                + 0.10 * mfi_score
            )

            # Bonus por duración
            n_bars = end - start
            quality += min(0.15, 0.05 * max(0, n_bars - 2))

            return min(quality, 1.0)
        except Exception:
            return 0.5

    def _vwap(self, start: int, end: int) -> float:
        """Calcula VWAP del segmento."""
        tp = (
            self.data["high"].values[start:end]
            + self.data["low"].values[start:end]
            + self.data["close"].values[start:end]
        ) / 3
        vol = self.data["volume"].values[start:end]
        total_vol = vol.sum()
        return (
            float((tp * vol).sum() / total_vol)
            if total_vol > 0
            else float(self.data["close"].iloc[start])
        )

    def detect(self, candle_index: int) -> Optional[Dict]:
        """
        Busca la mejor zona de acumulación anterior a la vela.
        """
        if self.data is None or candle_index < self.config["min_zone_bars"]:
            return None

        lookback = min(candle_index, 30)  # Reducido de 50 para performance
        start_idx = max(0, candle_index - lookback)

        best_zone = None
        best_quality = 0
        min_window = max(self.config["min_zone_bars"], 2)

        # Pre-compute candle data una sola vez
        c_high = self.data["high"].iloc[candle_index]
        c_low = self.data["low"].iloc[candle_index]
        c_close = self.data["close"].iloc[candle_index]
        price_2pct = c_close * 0.02
        global_avg = (
            self.data["volume"].values[max(0, start_idx - 50) : candle_index].mean()
        )

        for win in range(min_window, min(lookback, 12) + 1):
            for ws in range(start_idx, candle_index - win + 1):
                we = ws + win
                high_max = self.data["high"].values[ws:we].max()
                low_min = self.data["low"].values[ws:we].min()
                rng = high_max - low_min

                atr = self._calculate_atr(we)
                if atr == 0:
                    atr = c_close * 0.01

                # Filtro 1: rango estrecho
                if rng > self.config["atr_multiplier"] * atr * 1.5:
                    continue

                # Filtro 2: volumen suficiente
                avg_vol = self.data["volume"].values[ws:we].mean()
                if (
                    avg_vol
                    < max(0.5, self.config["volume_threshold"]) * global_avg * 0.7
                ):
                    continue

                # Filtro 3: zona toca la vela clave
                touches = (
                    (low_min <= c_high + price_2pct and high_max >= c_low - price_2pct)
                    or (abs(high_max - c_low) <= price_2pct)
                    or (abs(low_min - c_high) <= price_2pct)
                )
                if not touches:
                    continue

                # Calcular métricas
                vwap = self._vwap(ws, we)
                mfi = 50  # Simplificado
                quality = self._quality_score(ws, we, rng, avg_vol, vwap, mfi)

                # Bonus de recencia
                quality += (
                    0.2 * (1 - (candle_index - we) / lookback) if lookback > 0 else 0
                )

                if (
                    quality > best_quality
                    and quality >= self.config["quality_threshold"] * 0.8
                ):
                    best_quality = quality
                    best_zone = {
                        "start_idx": ws,
                        "end_idx": we,
                        "high": float(high_max),
                        "low": float(low_min),
                        "volume_avg": float(avg_vol),
                        "vwap": float(vwap),
                        "mfi": mfi,
                        "quality_score": quality,
                    }

        return best_zone

    def detect_all(self, key_candle_indices: List[int]) -> List[Dict]:
        """Detecta zonas para todas las velas clave."""
        zones = []
        for idx in key_candle_indices:
            zone = self.detect(idx)
            if zone:
                zones.append(zone)
        return zones


# ──────────────────────────────────────────────────────────────
# COMPONENTE 3 — Detector de Mini-Tendencia
# ──────────────────────────────────────────────────────────────


class MiniTrendDetector:
    """
    Detecta mini-tendencias usando segmentación ZigZag y regresión lineal.
    """

    def __init__(self, config: Dict = None):
        defaults = {
            "r2_min": 0.45,
            "min_trend_length": 5,
            "zigzag_threshold": 0.0018,  # 0.18% — P75 rango real vela 5m BTCUSDT
        }
        self.config = {**defaults, **(config or {})}
        self.data = None

    def load_data(self, data: pd.DataFrame):
        """Carga datos OHLCV."""
        self.data = data.copy()

    def _zigzag_segment(self, threshold_pct: Optional[float] = None) -> List[Dict]:
        """
        Segmentación ZigZag.
        """
        if threshold_pct is None:
            threshold_pct = self.config.get("zigzag_threshold", 0.0018)

        if self.data is None or len(self.data) < 10:
            return []

        pivots = []
        last_pivot = self.data["close"].iloc[0]
        last_idx = 0
        direction = None

        for i in range(1, len(self.data)):
            price = self.data["close"].iloc[i]
            if last_pivot == 0:
                change_pct = 0.0
            else:
                change_pct = abs(price - last_pivot) / last_pivot

            if change_pct >= threshold_pct:
                direction = "up" if price > last_pivot else "down"
                pivots.append({"index": i, "price": price, "direction": direction})
                last_pivot = price
                last_idx = i

        # Crear segmentos
        segments = []
        for i in range(len(pivots) - 1):
            start = pivots[i]
            end = pivots[i + 1]
            segments.append(
                {
                    "start_idx": start["index"],
                    "end_idx": end["index"],
                    "direction": start["direction"],
                    "start_price": start["price"],
                    "end_price": end["price"],
                }
            )

        return segments

    def _linear_regression(self, start_idx: int, end_idx: int) -> Dict:
        """
        Calcula regresión lineal y R² para un segmento.
        """
        segment = self.data.iloc[start_idx : end_idx + 1]
        x = np.arange(len(segment))
        y = segment["close"].values

        # Regresión lineal
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        # Calcular R²
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Normalizar slope (0-1)
        slope_norm = min(abs(slope) / (segment["close"].mean() * 0.01), 1.0)

        return {
            "slope": float(slope),
            "slope_normalized": slope_norm,
            "r2": float(r2),
            "direction": "bullish" if slope > 0 else "bearish",
        }

    def detect_all(self) -> List[Dict]:
        """Detecta todas las mini-tendencias."""
        if self.data is None:
            return []

        segments = self._zigzag_segment()
        trends = []

        for seg in segments:
            if seg["end_idx"] - seg["start_idx"] >= self.config["min_trend_length"]:
                regression = self._linear_regression(seg["start_idx"], seg["end_idx"])

                if regression["r2"] >= self.config["r2_min"]:
                    trends.append({**seg, **regression})

        return trends


# ──────────────────────────────────────────────────────────────
# COMPONENTE 4 — Scoring de Triple Coincidencia
# ──────────────────────────────────────────────────────────────


def score_triple_signal(
    quality_score: float,
    r2: float,
    slope: float,
    direction: str,
    candle_volume: float,
    body_pct: float,
) -> Dict[str, Any]:
    """
    Calcula el score final de una señal de triple coincidencia (0.0 – 1.0).
    """
    # Nivel 1: componentes básicos (70%)
    zona_score = max(0.0, min((quality_score - 0.45) / 0.4, 1.0))

    r2_factor = 1.3 if r2 >= 0.6 else (1.0 if r2 >= 0.45 else 0.9)
    dir_factor = 1.15 if direction == "bullish" else 0.90
    slope_f = max(0.3, min(slope, 1.0))
    trend_score = min(r2 * r2_factor * dir_factor * slope_f, 1.0)

    vol_score = min(candle_volume / 150, 1.0)
    if 15 <= body_pct <= 40:
        morph = 1.0
    elif 40 < body_pct <= 60:
        morph = 0.8
    elif 5 <= body_pct < 15:
        morph = 0.6
    else:
        morph = 0.3
    candle_score = 0.6 * vol_score + 0.4 * morph

    basic_score = 0.35 * zona_score + 0.35 * trend_score + 0.30 * candle_score

    # Nivel 2: factores avanzados (30%)
    scores = [zona_score, trend_score, candle_score]
    convergence = 1 - (max(scores) - min(scores))

    reliability = 1.0 if r2 >= 0.75 else (0.7 + r2 * 0.4)

    if direction == "bullish" and candle_volume > 80:
        potential = 0.85
    elif direction == "bullish" and candle_volume > 50:
        potential = 0.75
    elif direction == "bearish" and body_pct > 20:
        potential = 0.70
    else:
        potential = 0.60

    advanced = 0.20 * convergence + 0.15 * reliability + 0.15 * potential
    final = min(0.70 * basic_score + 0.30 * advanced, 1.0)

    if final >= 0.7:
        label = "🟢 Muy fuerte"
    elif final >= 0.6:
        label = "🟠 Fuerte"
    elif final >= 0.5:
        label = "🟡 Moderada"
    else:
        label = "⚪ Débil"

    return {
        "zona_score": round(zona_score, 3),
        "trend_score": round(trend_score, 3),
        "candle_score": round(candle_score, 3),
        "basic_score": round(basic_score, 3),
        "convergence": round(convergence, 3),
        "reliability": round(reliability, 3),
        "potential": round(potential, 3),
        "final_score": round(final, 3),
        "label": label,
    }


# ──────────────────────────────────────────────────────────────
# COMPONENTE 5 — Detector Principal (Triple Coincidence)
# ──────────────────────────────────────────────────────────────


@dataclass
class TripleSignal:
    """Resultado de una detección de triple coincidencia."""

    index: int
    price: float
    direction: str
    quality_score: float
    label: str
    key_candle: Dict
    accumulation_zone: Dict
    mini_trend: Dict
    scoring_details: Dict


class TripleCoincidenceDetector:
    """
    Detector principal con lógica correcta de la estrategia:
    1. Detecta Triple Coincidence (vela clave + zona)
    2. Rastrea zonas activas
    3. Detecta retests del precio
    4. Captura features en el momento del retest
    5. Determina outcome (rebote vs ruptura)
    """

    def __init__(self, config: Dict = None):
        defaults = {
            "volume_percentile_threshold": 70,
            "body_percentage_threshold": 40,
            "lookback_candles": 30,
            "atr_period": 14,
            "atr_multiplier": 1.5,
            "volume_threshold": 1.2,
            "min_zone_bars": 5,
            "quality_threshold": 0.45,
            "r2_min": 0.45,
            "min_trend_length": 5,
            "zigzag_threshold": 0.0018,  # 0.18% — P75 rango real vela 5m BTCUSDT
            "proximity_tolerance": 8,
            "retest_timeout_bars": 288,
            "outcome_lookahead_bars": 10,
            "breakout_confirm_atr_buffer": 0.03,
            "volume_z_threshold": 0.5,
            "min_clearance_atr": 1.0,
            "retest_proximity_pct": 0.001,  # 0.1% of price — proximity buffer for zone touch
            # EVO-TICKET-0005: zone_max_distance_atr is PROVISIONAL.
            # Calibrated via intuition; must be replaced with percentile analysis
            # of real expired-vs-active zone distances once enough detection
            # cycles are collected. See EVO_TICKET_LOG.md entry 0005.
            "zone_max_distance_atr": 5.0,
            "state_path": "aipha_memory/operational/detector_state.json",
            "market": "futures",
        }
        self.config = {**defaults, **(config or {})}
        self.market_prefix = self.config.get("market", "futures")
        self.key_candle_detector = KeyCandleDetector(self.config)
        self.zone_detector = AccumulationZoneDetector(self.config)
        self.trend_detector = MiniTrendDetector(self.config)

        # Estado: zonas activas pendientes de retest
        self.active_zones: List[ActiveZone] = []
        # Dataset de entrenamiento para Oracle
        self.training_samples: List[TrainingSample] = []

        # Buffers internos para Live Mode
        self._kline_buffer = []
        self._pending_live_retests = []

        # Intentar cargar estado persistente (Evita Cold Start de zonas)
        self.load_state()

    def seed_history(self, df: pd.DataFrame):
        """
        Pre-puebla el buffer de historial con datos previos (Warm Start).
        Permite que el primer tick de WebSocket tenga contexto estadístico.
        """
        if df is None or df.empty:
            return
        # Convertir a lista de dicts (formato de ticks)
        klines = df.to_dict("records")
        self._kline_buffer = klines[-200:]  # Mantener ultimas 200 para ATR/Z-Score
        logger.info(
            f"📊 Detector hidratado con {len(self._kline_buffer)} velas de historial (Bootstrap)."
        )

    def save_state(self):
        """Persiste las zonas activas en disco (estado interno del detector)."""
        import json
        from pathlib import Path

        import numpy as np

        # Ruta absoluta basada en __file__ para evitar dependencia del CWD
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        path = project_root / self.config["state_path"]
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serializar zonas a formato JSON puro
        data = []
        for z in self.active_zones:
            # Simplificamos dataclass a dict para persistencia
            z_dict = {
                "zone_id": z.zone_id,
                "candle_index": z.candle_index,
                "zone_top": z.zone_top,
                "zone_bottom": z.zone_bottom,
                "vwap_at_detection": z.vwap_at_detection,
                "detection_timestamp": z.detection_timestamp,
                "direction": z.direction,
                "key_candle": z.key_candle,
                "accumulation_zone": z.accumulation_zone,
                "mini_trend": z.mini_trend,
                "atr_at_detection": z.atr_at_detection,
                "max_price_since_detection": z.max_price_since_detection,
                "min_price_since_detection": z.min_price_since_detection,
                "max_price_since_last_touch": z.max_price_since_last_touch,
                "min_price_since_last_touch": z.min_price_since_last_touch,
                "retest_detected": z.retest_detected,
                "last_retest_ts_ms": z.last_retest_ts_ms,  # Debounce persistence
                "lifecycle_state": z.lifecycle_state.value,  # Enum a str
                "touch_count": z.touch_count,
                "polarity_flipped": z.polarity_flipped,
                "flip_ts": z.flip_ts,
                "flip_price": z.flip_price,
                "harvest_expiry_ts": z.harvest_expiry_ts,
            }
            data.append(z_dict)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_state(self):
        """
        Carga zonas activas desde disco al iniciar.
        Aplica un TTL de 24 horas para evitar cargar zonas obsoletas.
        """
        import json
        import time
        from pathlib import Path

        # Ruta absoluta basada en __file__
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        path = project_root / self.config["state_path"]
        if not path.exists():
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)

            loaded_zones = []
            for d in data:
                # Convertir str de vuelta a Enum
                state_str = d.pop("lifecycle_state", "active")
                state = ZoneLifecycleState(state_str)
                d.setdefault("max_price_since_detection", -1.0)
                d.setdefault("min_price_since_detection", 1e10)
                d.setdefault("max_price_since_last_touch", -1.0)
                d.setdefault("min_price_since_last_touch", 1e10)
                d.setdefault("retest_detected", False)
                d.setdefault("last_retest_ts_ms", None)  # Debounce restore
                d.setdefault("touch_count", 0)
                d.setdefault("polarity_flipped", False)
                d.setdefault("zone_id", "")
                d.setdefault("flip_ts", None)
                d.setdefault("flip_price", None)
                d.setdefault("harvest_expiry_ts", None)

                z = ActiveZone(**d)
                z.lifecycle_state = state
                loaded_zones.append(z)

            self.active_zones = loaded_zones
            if loaded_zones:
                logger.info(
                    f"💾 Recuperadas {len(loaded_zones)} zonas del estado persistente."
                )
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar el estado del detector: {e}")

    def detect(self, df: pd.DataFrame) -> List[TripleSignal]:
        """
        Detecta señales de Triple Coincidence (método legacy para compatibilidad).
        """
        # Cargar datos en sub-detectores
        self.key_candle_detector.load_data(df)
        self.zone_detector.load_data(df)
        self.trend_detector.load_data(df)

        key_candles = self.key_candle_detector.detect_all()
        zones = self.zone_detector.detect_all([kc["index"] for kc in key_candles])
        trends = self.trend_detector.detect_all()

        coincidences = self._find_coincidences(key_candles, zones, trends, df)
        signals = self._score_signals(coincidences)

        return signals

    def _find_coincidences(
        self,
        key_candles: List[Dict],
        zones: List[Dict],
        trends: List[Dict],
        data: pd.DataFrame,
    ) -> List[Dict]:
        """
        Busca coincidencias espaciales entre los tres componentes.
        """
        coincidences = []
        tolerance = self.config["proximity_tolerance"]

        # Para cada vela clave, buscar zona y tendencia cercanas
        for candle in key_candles:
            candle_idx = candle["index"]

            # Buscar zona cercana
            matching_zone = None
            for zone in zones:
                zone_center = (zone["start_idx"] + zone["end_idx"]) // 2
                if abs(candle_idx - zone_center) <= tolerance:
                    matching_zone = zone
                    break

            if not matching_zone:
                continue

            # Buscar tendencia cercana
            matching_trend = None
            for trend in trends:
                trend_center = (trend["start_idx"] + trend["end_idx"]) // 2
                if abs(candle_idx - trend_center) <= tolerance:
                    matching_trend = trend
                    break

            if not matching_trend:
                continue

            # Verificar umbrales mínimos
            if matching_zone.get("quality_score", 0) < self.config["quality_threshold"]:
                continue
            if matching_trend.get("r2", 0) < self.config["r2_min"]:
                continue

            coincidences.append(
                {
                    "key_candle": candle,
                    "accumulation_zone": matching_zone,
                    "mini_trend": matching_trend,
                }
            )

        return coincidences

    def _score_signals(self, coincidences: List[Dict]) -> List[TripleSignal]:
        """
        Calcula score final para cada coincidencia.
        """
        signals = []

        for coinc in coincidences:
            candle = coinc["key_candle"]
            zone = coinc["accumulation_zone"]
            trend = coinc["mini_trend"]

            scoring = score_triple_signal(
                quality_score=zone.get("quality_score", 0.5),
                r2=trend.get("r2", 0),
                slope=trend.get("slope_normalized", 0),
                direction=trend.get("direction", "bullish"),
                candle_volume=candle.get("volume_percentile", 0),
                body_pct=candle.get("body_percentage", 30),
            )

            signals.append(
                TripleSignal(
                    index=candle["index"],
                    price=candle["close"],
                    direction=trend["direction"],
                    quality_score=scoring["final_score"],
                    label=scoring["label"],
                    key_candle=candle,
                    accumulation_zone=zone,
                    mini_trend=trend,
                    scoring_details=scoring,
                )
            )

        return signals

    def process_live_tick(self, new_kline: Dict, micro_data: Dict) -> List[RetestEvent]:
        """
        Procesa una vela cerrada en tiempo real.
        Mantiene el historial interno (buffer) para los cálculos de indicadores.

        Incluye etiquetado diferido: los retests detectados se almacenan con sus
        features L2 y se etiquetan con outcome después de outcome_lookahead_bars
        velas adicionales, momento en el que se genera el TrainingSample completo.
        """
        # 1. Mantener buffer interno de velas (necesario para indicadores como ATR/regresión)
        if not hasattr(self, "_kline_buffer"):
            self._kline_buffer = []
        if not hasattr(self, "_pending_live_retests"):
            self._pending_live_retests = []

        self._kline_buffer.append(new_kline)
        if len(self._kline_buffer) > 200:
            self._kline_buffer.pop(0)

        if len(self._kline_buffer) < self.config["lookback_candles"]:
            return []

        # Convertir buffer a DataFrame para compatibilidad con lógica interna
        df = pd.DataFrame(self._kline_buffer)
        current_idx = len(df) - 1

        # 2. Cargar datos en sub-detectores
        self.key_candle_detector.load_data(df)
        self.zone_detector.load_data(df)
        self.trend_detector.load_data(df)

        retest_events = []

        # 3. Detectar nuevas zonas en esta vela
        new_zones = self._detect_new_zones(df, current_idx)
        self.active_zones.extend(new_zones)

        # 4. Monitorear retests de zonas activas
        curr_high = float(new_kline["high"])
        curr_low = float(new_kline["low"])

        for zone in self.active_zones:
            # Instrumentación observacional de clearance (Live)
            zone.max_price_since_detection = max(
                zone.max_price_since_detection, curr_high
            )
            zone.min_price_since_detection = min(
                zone.min_price_since_detection, curr_low
            )

            if not zone.retest_detected:
                # En live, micro_data viene del WS con obi_10 y cumulative_delta reales
                mf = pd.DataFrame([micro_data])
                retest = self._check_retest(
                    df, current_idx, zone, mf.rename(index={0: current_idx})
                )
                if retest:
                    retest_events.append(retest)
                    zone.retest_detected = (
                        True  # Legado — touch registrado en Zone.register_touch()
                    )
                    zone.last_retest_ts_ms = int(
                        new_kline.get("close_time", 0)
                    )  # Debounce sync
                    zone.retest_index = current_idx

                    # Calcular clearance en el momento del retest
                    if zone.direction == "bullish":
                        clearance = (
                            zone.max_price_since_detection - zone.zone_top
                        ) / zone.atr_at_detection
                    else:
                        clearance = (
                            zone.zone_bottom - zone.min_price_since_detection
                        ) / zone.atr_at_detection

                    # Guardar en buffer de pendientes para etiquetado diferido
                    is_interior = (
                        zone.zone_bottom <= retest.retest_price <= zone.zone_top
                    )
                    self._pending_live_retests.append(
                        {
                            "retest": retest,
                            "zone": zone,
                            "detected_at_buffer_idx": current_idx,
                            "features": {
                                "vwap_at_retest": retest.vwap_at_retest,
                                "obi_10_at_retest": retest.obi_10_at_retest,
                                "cumulative_delta_at_retest": retest.cumulative_delta_at_retest,
                                "delta_divergence": retest.delta_divergence,
                                "atr_14": retest.atr_14,
                                "regime": retest.regime,
                                "direction": zone.direction,
                                "max_clearance_atr": round(float(clearance), 4),
                                "retest_type": "ZONE_INTERIOR"
                                if is_interior
                                else "PROXIMITY_BUFFER",
                            },
                            "zone_id": zone.zone_id,
                        }
                    )

        # 5. Etiquetado diferido: resolver outcomes de retests pendientes
        lookahead = self.config["outcome_lookahead_bars"]
        resolved = []
        for i, pending in enumerate(self._pending_live_retests):
            bars_elapsed = current_idx - pending["detected_at_buffer_idx"]
            if bars_elapsed >= lookahead:
                # Tenemos suficientes velas futuras para etiquetar
                outcome = self._determine_outcome(
                    df, pending["detected_at_buffer_idx"], pending["zone"]
                )
                if outcome != "PENDING":
                    training_sample = TrainingSample(
                        features=pending["features"],
                        outcome=outcome,
                        zone_id=pending["zone_id"],
                        retest_timestamp=pending["retest"].retest_timestamp,
                    )
                    self.training_samples.append(training_sample)
                    resolved.append(i)

        # Limpiar resueltos (en orden inverso para no corromper índices)
        for i in sorted(resolved, reverse=True):
            self._pending_live_retests.pop(i)

        # 6. Limpieza de zonas expiradas
        self._cleanup_expired_zones(current_idx)

        return retest_events

    def process_stream(
        self, df: pd.DataFrame, micro_features: Optional[pd.DataFrame] = None
    ) -> List[RetestEvent]:
        """
        Procesa stream de datos con lógica correcta:
        1. Detecta nuevas zonas (Triple Coincidence)
        2. Monitorea retests de zonas activas
        3. Captura features en retest
        4. Determina outcomes

        Args:
            df: DataFrame OHLCV
            micro_features: DataFrame opcional con VWAP, OBI, cumulative delta

        Returns:
            Lista de RetestEvent detectados
        """
        # Pre-cargar datos en sub-detectores
        self.key_candle_detector.load_data(df)
        self.zone_detector.load_data(df)
        self.trend_detector.load_data(df)

        # Pre-calcular tendencias (no cambian por iteración)
        all_trends = self.trend_detector.detect_all()

        retest_events = []
        self.is_bootstrapping = True

        try:
            for idx in range(len(df)):
                # 1. Detectar nuevas zonas
                new_zones = self._detect_new_zones(df, idx, all_trends)
                self._add_zones_without_duplicates(new_zones)

                # 2. Monitorear retests de zonas activas
                for zone in self.active_zones:
                    # 2. Actualizar extremos de precio (Instrumentación Cat.1)
                    curr_high = df.iloc[idx]["high"]
                    curr_low = df.iloc[idx]["low"]
                    zone.max_price_since_detection = max(
                        zone.max_price_since_detection, curr_high
                    )
                    zone.min_price_since_detection = min(
                        zone.min_price_since_detection, curr_low
                    )

                    if not zone.retest_detected:
                        retest = self._check_retest(df, idx, zone, micro_features)
                        if retest:
                            retest_events.append(retest)
                            zone.retest_detected = True  # Legado — touch registrado en Zone.register_touch()
                            zone.last_retest_ts_ms = (
                                retest.retest_timestamp
                            )  # Debounce sync
                            zone.retest_index = retest.retest_index

                            # 3. Determinar outcome (si hay suficientes datos futuros)
                            if idx + self.config["outcome_lookahead_bars"] < len(df):
                                outcome = self._determine_outcome(
                                    df, retest.retest_index, retest.zone
                                )
                                retest.outcome = outcome
                                zone.outcome = outcome

                                # Calcular max_clearance_atr normalizado por ATR de detección
                                if zone.direction == "bullish":
                                    clearance = (
                                        zone.max_price_since_detection - zone.zone_top
                                    ) / zone.atr_at_detection
                                else:
                                    clearance = (
                                        zone.zone_bottom
                                        - zone.min_price_since_detection
                                    ) / zone.atr_at_detection

                                # 4. Guardar sample de entrenamiento con la nueva feature
                                training_sample = TrainingSample(
                                    features={
                                        "vwap_at_retest": retest.vwap_at_retest,
                                        "obi_10_at_retest": retest.obi_10_at_retest,
                                        "cumulative_delta_at_retest": retest.cumulative_delta_at_retest,
                                        "delta_divergence": retest.delta_divergence,
                                        "atr_14": retest.atr_14,
                                        "regime": retest.regime,
                                        "direction": zone.direction,
                                        "max_clearance_atr": round(float(clearance), 4),
                                    },
                                    outcome=outcome,
                                    zone_id=zone.zone_id,
                                    retest_timestamp=retest.retest_timestamp,
                                )
                                self.training_samples.append(training_sample)

                # 3. Limpiar zonas expiradas (timeout)
                current_ts = int(df.iloc[idx].get("close_time", time.time() * 1000))
                current_price = float(df.iloc[idx]["close"])
                self._cleanup_expired_zones(idx, current_ts, current_price)
        finally:
            self.is_bootstrapping = False

        return retest_events

    def _detect_new_zones(
        self, df: pd.DataFrame, current_idx: int, precomputed_trends: List[Dict] = None
    ) -> List[ActiveZone]:
        """Detecta nuevas zonas de Triple Coincidence en el índice actual."""
        if current_idx < self.config["lookback_candles"]:
            return []

        # Detectar vela clave en índice actual
        key_candle = self.key_candle_detector.detect(current_idx)
        if key_candle is None:
            return []

        # Buscar zona de acumulación asociada
        zone = self.zone_detector.detect(current_idx)
        if zone is None:
            return []

        # Usar tendencias pre-calculadas o calcular
        trends = (
            precomputed_trends
            if precomputed_trends is not None
            else self.trend_detector.detect_all()
        )

        # Buscar coincidencias
        coincidences = self._find_coincidences([key_candle], [zone], trends, df)

        if not coincidences:
            logger.debug(
                f"_detect_new_zones idx={current_idx}: key_candle={key_candle['index']} "
                f"zone={zone.get('start_idx')}-{zone.get('end_idx')} qs={zone.get('quality_score', 0):.3f} "
                f"trends={len(trends)} -> no coincidence"
            )

        new_zones = []
        for coinc in coincidences:
            candle = coinc["key_candle"]
            zone_data = coinc["accumulation_zone"]
            trend = coinc["mini_trend"]

            # Capturar ATR en el momento de detección para normalización futura
            atr_now = self._calculate_atr(df, current_idx)
            current_price = df.iloc[current_idx]["close"]

            active_zone = ActiveZone(
                candle_index=candle["index"],
                zone_top=zone_data.get(
                    "high", zone_data.get("zone_top", candle["high"])
                ),
                zone_bottom=zone_data.get(
                    "low", zone_data.get("zone_bottom", candle["low"])
                ),
                vwap_at_detection=zone_data.get("vwap", candle["close"]),
                detection_timestamp=int(
                    df.iloc[current_idx].get("close_time", candle["index"] * 300000)
                ),
                direction=trend["direction"],
                key_candle=candle,
                accumulation_zone=zone_data,
                mini_trend=trend,
                atr_at_detection=atr_now,
                max_price_since_detection=current_price,
                min_price_since_detection=current_price,
            )
            new_zones.append(active_zone)

        return new_zones

    def _check_retest(
        self,
        df: pd.DataFrame,
        idx: int,
        zone: ActiveZone,
        micro_features: Optional[pd.DataFrame] = None,
    ) -> Optional[RetestEvent]:
        """Verifica si el precio retestea la zona activa."""
        current_price = df.iloc[idx]["close"]

        # Verificar si el precio está en la zona (con proximity buffer)
        proximity = current_price * self.config.get("retest_proximity_pct", 0.001)
        if (
            (zone.zone_bottom - proximity)
            <= current_price
            <= (zone.zone_top + proximity)
        ):
            # Capturar features del retest
            vwap = (
                micro_features.iloc[idx]["vwap"]
                if micro_features is not None
                else zone.vwap_at_detection
            )
            obi_10 = (
                micro_features.iloc[idx]["obi_10"]
                if micro_features is not None
                else 0.0
            )
            cumulative_delta = (
                micro_features.iloc[idx]["cumulative_delta"]
                if micro_features is not None
                else 0.0
            )

            # Calcular delta divergence
            if zone.direction == "bullish":
                delta_divergence = "BULLISH_ABSORPTION"
            else:
                delta_divergence = "BEARISH_EXHAUSTION"

            # Calcular ATR
            atr = self._calculate_atr(df, idx)

            # Determinar régimen
            regime = self._determine_regime(df, idx)

            return RetestEvent(
                zone=zone,
                retest_index=idx,
                retest_price=current_price,
                retest_timestamp=int(df.iloc[idx].get("close_time", idx * 300000)),
                vwap_at_retest=vwap,
                obi_10_at_retest=obi_10,
                cumulative_delta_at_retest=cumulative_delta,
                delta_divergence=delta_divergence,
                atr_14=atr,
                regime=regime,
            )

        return None

    def _determine_outcome(self, df: pd.DataFrame, retest_idx: int, zone=None) -> str:
        """Determina outcome (BOUNCE vs BREAKOUT) usando los límites de la zona."""
        if zone is None:
            return "BOUNCE"  # Fallback legacy

        lookahead = self.config["outcome_lookahead_bars"]
        if retest_idx + lookahead >= len(df):
            return "PENDING"

        zone_top = zone.zone_top
        zone_bottom = zone.zone_bottom

        for i in range(1, lookahead + 1):
            future_idx = retest_idx + i
            future_close = df.iloc[future_idx]["close"]

            if zone.direction == "bullish":
                if future_close < zone_bottom:
                    return "BREAKOUT"  # rompió hacia abajo
                if future_close > zone_top:
                    return "BOUNCE"  # rebotó hacia arriba
            else:
                if future_close > zone_top:
                    return "BREAKOUT"  # rompió hacia arriba
                if future_close < zone_bottom:
                    return "BOUNCE"  # rebotó hacia abajo

        return "BOUNCE"  # se mantuvo en zona

    def _calculate_atr(self, df: pd.DataFrame, idx: int) -> float:
        """Calcula ATR."""
        period = self.config["atr_period"]
        if idx < period:
            return 0.0

        high_range = df.iloc[idx - period : idx + 1]["high"].max()
        low_range = df.iloc[idx - period : idx + 1]["low"].min()
        return high_range - low_range

    def _determine_regime(self, df: pd.DataFrame, idx: int) -> str:
        """Determina régimen de mercado."""
        if idx < 20:
            return "LATERAL"

        # Calcular R² de regresión lineal
        prices = df.iloc[idx - 20 : idx + 1]["close"].values
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        if r2 >= 0.6:
            return "TREND"
        else:
            return "LATERAL"

    def _cleanup_expired_zones(
        self,
        current_idx: Optional[int] = None,
        current_timestamp_ms: Optional[int] = None,
        current_price: Optional[float] = None,
    ):
        """Elimina zonas expiradas, preservando zonas en HARVESTING hasta su TTL.

        Uses real time (detection_timestamp) because candle_index is not stable
        across process restarts — persisted zones would otherwise never expire.
        Also expires zones that have moved too far from the current price so the
        GUI does not display stale zones that are no longer relevant.
        """
        if getattr(self, "is_bootstrapping", False):
            return

        if current_timestamp_ms is None:
            current_timestamp_ms = int(time.time() * 1000)

        timeout_bars = self.config["retest_timeout_bars"]
        timeout_ms = timeout_bars * 60 * 1000  # 1m candles
        max_distance_atr = self.config.get("zone_max_distance_atr", 5.0)

        kept = []
        for z in self.active_zones:
            if z.is_exhausted():
                continue

            # Time-based expiry (robust across restarts)
            age_ms = current_timestamp_ms - z.detection_timestamp
            timed_out = age_ms > timeout_ms

            # Index-based expiry (only meaningful within the same run)
            # Bootstrap zones have candle_index from historical data (e.g. 225, 469)
            # which is incompatible with live _kline_buffer indices (starting at 0).
            # Skip index check for negative index_age (cross-run zones).
            if current_idx is not None:
                index_age = current_idx - z.candle_index
                index_in_range = index_age < 0 or 0 <= index_age < timeout_bars
            else:
                index_in_range = True

            # Price-distance expiry: remove zones that have been left behind.
            # A bullish zone (support) is stale if price is far below it.
            # A bearish zone (resistance) is stale if price is far above it.
            price_too_far = False
            if current_price is not None and z.atr_at_detection > 0:
                if z.direction == "bullish" and current_price < z.zone_bottom:
                    distance = z.zone_bottom - current_price
                    price_too_far = (distance / z.atr_at_detection) > max_distance_atr
                elif z.direction == "bearish" and current_price > z.zone_top:
                    distance = current_price - z.zone_top
                    price_too_far = (distance / z.atr_at_detection) > max_distance_atr

            # Harvesting zones survive until their 48h TTL or explicit exhaustion,
            # but they can still be removed if they are absurdly far from price.
            is_harvesting = z.lifecycle_state == ZoneLifecycleState.HARVESTING

            if is_harvesting:
                # Use harvest_expiry_ts (48h from breakout) for harvesting zones
                harvest_expired = current_timestamp_ms > getattr(z, 'harvest_expiry_ts', 0)
                if harvest_expired or price_too_far:
                    continue
                kept.append(z)
            elif not timed_out and index_in_range and not price_too_far:
                kept.append(z)

        self.active_zones = kept

    # ── Tick-level Retest Detection (Semana 2) ──────────────────

    def check_intra_candle_retest(
        self, price: float, timestamp_ms: int, debounce_ms: int = 5000
    ) -> list:
        """
        Check if `price` touches any active zone. Called at tick speed
        (~10-50x/second) from LiveDataFeedAdapter._process_trade().

        O(n_zones) pure: no I/O, no DataFrame ops, no logging.
        Heavy L2 synthesis happens ONLY when this returns a hit.

        Args:
            price: current trade price
            timestamp_ms: Binance event timestamp in ms
            debounce_ms: min ms between retests on the same zone (default 5s)

        Returns:
            List of dicts, one per zone touched:
            [{
                "zone": ActiveZone,
                "price": float,
                "timestamp_ms": int,
                "clearance_atr": float,
            }]
        """
        hits = []

        for zone in self.active_zones:
            if zone.is_exhausted():
                continue

            # Debounce: skip if same zone was hit < debounce_ms ago
            if zone.last_retest_ts_ms is not None:
                if (timestamp_ms - zone.last_retest_ts_ms) < debounce_ms:
                    continue
            elif zone.retest_detected:
                # Historical zone loaded without debounce ts — skip silently
                logger.debug(
                    f"Zone {zone.zone_id or zone.candle_index}: "
                    f"retest_detected=True but last_retest_ts_ms=None — legacy zone, skipping debounce"
                )

            # Update price extremes (clearance tracking)
            zone.max_price_since_detection = max(zone.max_price_since_detection, price)
            zone.min_price_since_detection = min(zone.min_price_since_detection, price)

            # Shadow Harvesting: al perforar el lado adverso, la zona pasa a HARVESTING.
            if not zone.polarity_flipped:
                atr_for_breakout = (
                    zone.atr_at_detection if zone.atr_at_detection > 0 else 0.0
                )
                breakout_buffer = atr_for_breakout * self.config.get(
                    "breakout_confirm_atr_buffer", 0.0
                )
                lower_breakout_level = zone.zone_bottom - breakout_buffer
                upper_breakout_level = zone.zone_top + breakout_buffer

                if zone.direction == "bullish" and price < lower_breakout_level:
                    zone.register_breakout(breakout_price=price)
                elif zone.direction == "bearish" and price > upper_breakout_level:
                    zone.register_breakout(breakout_price=price)

            # Update per-touch price extremes (for clearance tracking)
            zone.max_price_since_last_touch = max(
                zone.max_price_since_last_touch, price
            )
            zone.min_price_since_last_touch = min(
                zone.min_price_since_last_touch, price
            )

            # Check if price is inside the zone (with proximity buffer)
            proximity = price * self.config.get("retest_proximity_pct", 0.001)
            if (zone.zone_bottom - proximity) <= price <= (zone.zone_top + proximity):
                # Calculate clearance at this moment
                atr = zone.atr_at_detection
                if atr <= 0:
                    atr = price * 0.001

                if zone.direction == "bullish":
                    clearance = (zone.max_price_since_detection - zone.zone_top) / atr
                else:
                    clearance = (
                        zone.zone_bottom - zone.min_price_since_detection
                    ) / atr

                # ── CLEARANCE LOGIC (Multi-Touch) ────────────────────
                # Para el 2do+ toque, exigir que el precio se haya
                # alejado > min_clearance_atr * ATR de la zona antes
                # de aceptar un nuevo retest independiente.
                min_clearance = self.config.get("min_clearance_atr", 1.0)
                if zone.touch_count > 0:
                    if zone.direction == "bullish":
                        touch_clearance = (
                            zone.max_price_since_last_touch - zone.zone_top
                        ) / atr
                    else:
                        touch_clearance = (
                            zone.zone_bottom - zone.min_price_since_last_touch
                        ) / atr

                    if touch_clearance < min_clearance:
                        continue  # No acepto el toque — el precio no se alejó lo suficiente

                # Mark zone as retested (prevents future triggers)
                zone.retest_detected = (
                    True  # Legado — touch registrado en Zone.register_touch()
                )
                zone.last_retest_ts_ms = timestamp_ms

                # Classify touch type for future calibration
                is_interior = zone.zone_bottom <= price <= zone.zone_top
                retest_type = "ZONE_INTERIOR" if is_interior else "PROXIMITY_BUFFER"

                hits.append(
                    {
                        "zone": zone,
                        "price": price,
                        "timestamp_ms": timestamp_ms,
                        "clearance_atr": round(float(clearance), 4),
                        "retest_type": retest_type,
                    }
                )

        return hits

    def feed_kline_for_zone_detection(self, kline: dict, micro_data: dict):
        """
        Called once per closed candle (1min) to detect NEW zones.
        Separated from retest detection which now runs at tick speed.

        This replaces the zone-detection portion of process_live_tick().
        """
        if not hasattr(self, "_kline_buffer"):
            self._kline_buffer = []

        self._kline_buffer.append(kline)
        if len(self._kline_buffer) > 200:
            self._kline_buffer.pop(0)

        if len(self._kline_buffer) < self.config["lookback_candles"]:
            return

        df = pd.DataFrame(self._kline_buffer)
        current_idx = len(df) - 1

        # Load data into sub-detectors
        self.key_candle_detector.load_data(df)
        self.zone_detector.load_data(df)
        self.trend_detector.load_data(df)

        # Precalculate trends once per candle over the whole buffer.
        # Without this, _detect_new_zones recalculates ZigZag trends on every
        # single candle and rarely finds a valid segment in a small buffer.
        precomputed_trends = self.trend_detector.detect_all()

        # Detect new zones at this candle
        new_zones = self._detect_new_zones(df, current_idx, precomputed_trends)

        if new_zones:
            logger.info(
                f"🔍 NUEVAS ZONAS detectadas: {len(new_zones)} en idx={current_idx} "
                f"buffer={len(self._kline_buffer)} active={len(self.active_zones)}"
            )
        self._add_zones_without_duplicates(new_zones)

        # ── Z-Score Calibration Instrumentation (Cat.1, observacional) ──
        self._log_zscore_calibration(df, current_idx, len(new_zones) > 0)

        # Cleanup expired zones using the closed candle's timestamp and price
        current_ts = int(df.iloc[current_idx].get("close_time", time.time() * 1000))
        current_price = float(df.iloc[current_idx]["close"])
        self._cleanup_expired_zones(current_idx, current_ts, current_price)

    def _add_zones_without_duplicates(self, new_zones: List[ActiveZone]):
        """Añade zonas a la lista filtrando duplicados espaciales o temporales."""
        for nz in new_zones:
            is_dupe = False
            for az in self.active_zones:
                if az.candle_index == nz.candle_index and az.direction == nz.direction:
                    is_dupe = True
                    break

                # Solapamiento de precios (si comparten > 90% del rango)
                inter_top = min(az.zone_top, nz.zone_top)
                inter_bot = max(az.zone_bottom, nz.zone_bottom)
                if inter_top > inter_bot:
                    union_top = max(az.zone_top, nz.zone_top)
                    union_bot = min(az.zone_bottom, nz.zone_bottom)
                    overlap = (inter_top - inter_bot) / (union_top - union_bot)
                    if overlap > 0.9:
                        is_dupe = True
                        break

            if not is_dupe:
                self.active_zones.append(nz)

    def _log_zscore_calibration(self, df: pd.DataFrame, idx: int, zone_detected: bool):
        """
        Registra Z-Score de volumen para cada vela procesada.
        Archivo: aipha_memory/operational/zscore_calibration_log.jsonl
        Cat.1: observacional, no cambia comportamiento.
        """
        import json
        from datetime import datetime, timezone
        from pathlib import Path

        try:
            current = df.iloc[idx]
            vol = float(current["volume"])
            candle_range = float(current["high"]) - float(current["low"])
            body_size = abs(float(current["close"]) - float(current["open"]))
            body_pct = (100 * body_size / candle_range) if candle_range > 0 else 0

            # Calcular Z-Score local (misma ventana que KeyCandleDetector)
            window = min(30, idx)
            if window < 5:
                return
            vol_series = df.iloc[max(0, idx - window) : idx + 1]["volume"].astype(float)
            mean_vol = vol_series.mean()
            std_vol = vol_series.std()
            z_score = (vol - mean_vol) / std_vol if std_vol > 0 else 0.0

            threshold = self.config.get("volume_z_threshold", 0.5)
            passed = z_score >= threshold

            entry = {
                "ts_utc": datetime.fromtimestamp(
                    int(current.get("open_time", 0)) / 1000, tz=timezone.utc
                ).isoformat()
                if current.get("open_time", 0) > 0
                else datetime.now(timezone.utc).isoformat(),
                "vol_z_score": round(float(z_score), 4),
                "body_pct": round(float(body_pct), 1),
                "passed_filter": passed,
                "zone_detected": zone_detected,
            }

            project_root = Path(__file__).resolve().parent.parent.parent.parent
            log_path = (
                project_root
                / "aipha_memory"
                / "operational"
                / "zscore_calibration_log.jsonl"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # Instrumentación Cat.1: no debe interrumpir el flujo principal

    def get_training_dataset(self) -> List[TrainingSample]:
        """Retorna dataset de entrenamiento para Oracle."""
        return self.training_samples


# ──────────────────────────────────────────────────────────────
# ADAPTADOR PARA ORACLE INTEGRATION (LEGACY - DEPRECATED)
# ──────────────────────────────────────────────────────────────


def triple_signal_to_microstructure(
    signal: TripleSignal, data: pd.DataFrame, symbol: str = "BTCUSDT"
) -> Dict:
    """
    Convierte TripleSignal a formato compatible con Oracle.

    Nota: Esta es una implementación simplificada que usa placeholders
    para features de microestructura que requieren datos de order book.
    En producción, estos valores deben calcularse desde datos reales.
    """
    candle = signal.key_candle
    zone = signal.accumulation_zone
    trend = signal.mini_trend

    # Calcular VWAP aproximado desde la zona
    vwap = zone.get("vwap", signal.price)

    # Calcular ATR aproximado desde datos
    atr = (
        data["high"].iloc[max(0, signal.index - 14) : signal.index + 1].max()
        - data["low"].iloc[max(0, signal.index - 14) : signal.index + 1].min()
    )

    # Placeholder para OBI (requiere order book)
    obi_10 = 0.0

    # Placeholder para cumulative delta (requiere trade-by-trade)
    cumulative_delta = 0.0

    # Determinar delta divergence basado en dirección
    if signal.direction == "bullish":
        delta_divergence = "BULLISH_ABSORPTION"
    else:
        delta_divergence = "BEARISH_EXHAUSTION"

    # Determinar régimen basado en tendencia
    if trend["r2"] >= 0.6:
        regime = "TREND"
    elif atr > signal.price * 0.01:
        regime = "HIGH_VOL"
    else:
        regime = "LATERAL"

    # Crear dict compatible con MicrostructureRecord
    microstructure = {
        "timestamp": int(signal.index),
        "symbol": symbol,
        "open": candle["open"],
        "high": candle["high"],
        "low": candle["low"],
        "close": candle["close"],
        "volume": candle["volume"],
        "vwap": vwap,
        "vwap_std_1": atr * 0.5,  # Placeholder
        "vwap_std_2": atr * 0.3,  # Placeholder
        "obi_10": obi_10,
        "cumulative_delta": cumulative_delta,
        "delta_divergence": delta_divergence,
        "atr_14": atr,
        "regime": regime,
    }

    return microstructure
