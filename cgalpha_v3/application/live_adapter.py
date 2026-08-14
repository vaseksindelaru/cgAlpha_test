"""
CGAlpha v3 — Live Data Feed Adapter (v2: Two-Speed Architecture)
================================================================
Puente asíncrono que conecta el WebSocket Manager con el Signal Detector.

Arquitectura de dos velocidades:
  - ZONAS:   detectadas al cerrar la vela (cada 1 minuto)
  - RETESTS: detectados a velocidad de tick (cada aggTrade, ~10-50/s)
  - L2 PROFILE: sintetizado SOLO en el ms del toque de zona (~3 veces/día)

Esto garantiza que el perfil temporal L2 de 30s se capture en el instante
exacto del retest, no 45 segundos después al cierre de la vela.
"""

import asyncio
import json
import logging
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from cgalpha_v3.data_quality.nexus_gate import NexusGate
from cgalpha_v3.domain.base_component import BaseComponentV3, ComponentManifest
from cgalpha_v3.domain.deferred_outcome_monitor import DeferredOutcomeMonitor
from cgalpha_v3.infrastructure.binance_websocket_manager import BinanceWebSocketManager
from cgalpha_v3.infrastructure.signal_detector.triple_coincidence import (
    RetestEvent,
    TripleCoincidenceDetector,
)
from cgalpha_v3.risk.order_manager import DryRunOrderManager

logger = logging.getLogger("live_adapter")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TRAINING_DATASET_V2 = (
    _PROJECT_ROOT / "aipha_memory" / "operational" / "training_dataset_v2.jsonl"
)


class LiveDataFeedAdapter(BaseComponentV3):
    """
    ╔═══════════════════════════════════════════════════════════════╗
    ║  APPLICATION — Live Data Feed Adapter (v2)                    ║
    ║  Bridge: WebSocket -> Detector -> ShadowTrader                ║
    ║                                                               ║
    ║  Two-Speed Architecture:                                      ║
    ║    Speed 1 (1min):  Zone detection at candle close            ║
    ║    Speed 2 (tick):  Retest detection at every aggTrade        ║
    ║                     L2 synthesis ONLY on zone touch           ║
    ╚═══════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        manifest: ComponentManifest,
        ws_manager: BinanceWebSocketManager,
        detector: TripleCoincidenceDetector,
        order_manager: Optional[DryRunOrderManager] = None,
    ):
        super().__init__(manifest)
        self.ws = ws_manager
        self.detector = detector
        self.order_mgr = order_manager or DryRunOrderManager()
        self._oracle = None
        self._oracle_regressor = None

        # Candle aggregation
        self.current_kline: Dict[str, Any] = {}
        # EVO-TICKET-0006: 5m operation aligns with detector calibration.
        # The detector's zigzag_threshold (0.0018) was calibrated against
        # real BTCUSDT 5m candle ranges (P75). Operating at 1m was an
        # undocumented MVP default (commit aa0190df). See ADR-EVO-TICKET-0006-1.
        self.interval_s = 300
        self.symbol = "BTCUSDT"
        self._last_kline_close = 0

        # Heartbeat watchdog (EVO-TICKET-0003 fix)
        # Tracks last aggTrade arrival to detect silent stream failure.
        # When aggTrade is absent > _heartbeat_timeout_ms, depthUpdate
        # timestamps drive candle close as fallback clock.
        self._last_aggtrade_ts_ms: int = int(time.time() * 1000)
        self._heartbeat_timeout_ms: int = 30_000  # 30 seconds
        self._last_heartbeat_price: float = 0.0
        self._heartbeat_interval_s: int = 10  # heartbeat file refresh interval
        self._last_heartbeat_write_ts: float = 0.0
        self.live_signals: List[Dict] = []

        # NexusGate & Causal Drift
        self.nexus = NexusGate()
        self.micro_buffer: List[Dict] = []
        self.delta_causal = 0.0
        self._is_causally_safe = True
        self._disable_nexus_gate = os.environ.get(
            "CGALPHA_DISABLE_NEXUS_GATE", "0"
        ).lower() in {"1", "true", "yes"}
        self._last_bypass_warn_ts = 0.0
        self._bypass_disable_mode = (
            os.environ.get("CGALPHA_BYPASS_DISABLE_MODE", "set_a_ready").strip().lower()
        )
        self._bypass_disable_full_target = int(
            os.environ.get("CGALPHA_BYPASS_DISABLE_AT_FULL", "50")
        )

        # GUI-mode adapter: when True, this adapter does NOT write active_zones.json.
        # Only the shadow trader process should write; GUI server reads passively.
        self._gui_readonly = os.environ.get(
            "CGALPHA_GUI_READONLY", "0"
        ).lower() in {"1", "true", "yes"}
        self._set_a_min_bounce = int(os.environ.get("CGALPHA_SET_A_MIN_BOUNCE", "8"))
        self._set_a_min_breakout = int(
            os.environ.get("CGALPHA_SET_A_MIN_BREAKOUT", "16")
        )
        self._set_a_min_full = int(os.environ.get("CGALPHA_SET_A_MIN_FULL", "24"))
        self._proximity_penalty = float(
            os.environ.get("CGALPHA_PROXIMITY_PENALTY", "0.85")
        )

        # Incremental Counter Patch (Feature Flagged)
        self._use_incremental_counter = (
            os.environ.get("CGALPHA_USE_INCREMENTAL_COUNTER", "0") == "1"
        )
        self._cached_full_count = -1  # -1 means not yet initialized
        self._set_a_cache_mtime = None
        self._set_a_cache_stats = None

        # Deferred Labeling (Semana 3)
        self.deferred_monitor = DeferredOutcomeMonitor()

        # Register callback
        self.ws.add_callback(self.on_ws_message)

    def inject_oracle(self, oracle):
        """Inyecta el modelo entrenado y actualiza NexusGate con la firma causal real."""
        self._oracle = oracle
        logger.info("🧠 Oracle validado e inyectado en LiveAdapter.")

        # Actualizar NexusGate con la firma causal del dataset de entrenamiento
        # (evita baseline hardcodeado obsoleto que infla ΔCausal)
        if hasattr(oracle, "get_causal_signature"):
            causal_sig = oracle.get_causal_signature()
            if causal_sig and causal_sig.get("obi_std", 0) > 0:
                # Verificar si el baseline del modelo difiere significativamente del dataset actual
                # Si es así, usar los stats del dataset actual como baseline (más representativo)
                baseline_to_use = causal_sig
                try:
                    from pathlib import Path
                    import json
                    import statistics
                    dataset_path = Path(__file__).resolve().parent.parent.parent / "aipha_memory" / "operational" / "training_dataset_v2.jsonl"
                    if dataset_path.exists():
                        obi_vals = []
                        with open(dataset_path) as f:
                            for line in f:
                                d = json.loads(line)
                                l2 = d.get('l2_snapshot_at_touch', {})
                                obi = l2.get('obi_10')
                                if obi is not None:
                                    obi_vals.append(obi)
                        if obi_vals:
                            current_mean = statistics.mean(obi_vals)
                            current_std = statistics.stdev(obi_vals) if len(obi_vals) > 1 else 0
                            model_mean = causal_sig.get("obi_mean", 0)
                            model_std = causal_sig.get("obi_std", 0)

                            # Decidir qué baseline usar
                            if model_std > 0 and current_std > 0:
                                std_ratio = current_std / model_std
                                mean_diff = abs(current_mean - model_mean) / (model_std + 1e-6)

                                if std_ratio > 1.5 or std_ratio < 0.67 or mean_diff > 2.0:
                                    logger.warning(
                                        f"⚠️ DERIVA DE BASELINE DETECTADA: "
                                        f"modelo std={model_std:.4f} vs dataset std={current_std:.4f} "
                                        f"(ratio={std_ratio:.2f}x, mean_diff={mean_diff:.1f}σ). "
                                        f"Usando stats del dataset actual como baseline."
                                    )
                                    # Crear baseline con stats actuales
                                    baseline_to_use = {
                                        "obi_mean": current_mean,
                                        "obi_std": current_std,
                                        "delta_mean": causal_sig.get("delta_mean", 0),
                                        "delta_std": causal_sig.get("delta_std", 100.0),
                                    }
                except Exception as e:
                    logger.debug(f"No se pudo comparar baseline: {e}")

                self.nexus = NexusGate(baseline_signature=baseline_to_use)
                logger.info(
                    f"🔄 NexusGate recalibrado con firma causal real: "
                    f"obi_mean={baseline_to_use.get('obi_mean', 0):.4f}, "
                    f"obi_std={baseline_to_use.get('obi_std', 0):.4f}"
                )
            else:
                logger.warning("⚠️ Oracle sin firma causal válida (obi_std=0), NexusGate mantiene baseline por defecto")
        else:
            logger.warning("⚠️ Oracle no expone get_causal_signature, NexusGate mantiene baseline por defecto")

    def inject_regressor(self, regressor):
        """Inyecta el regresor MAE (Capa 2)."""
        self._oracle_regressor = regressor
        logger.info("🎯 Regresor MAE (Capa 2) inyectado en LiveAdapter.")

    def warm_start(self, lookback_bars: int = 60) -> bool:
        """
        Hidrata el buffer de klines del detector desde Binance REST.
        Evita el cold start de 30+ minutos después de reinicios.
        """
        if not hasattr(self.detector, "seed_history"):
            logger.warning("⚠️ Detector no soporta seed_history — warm start omitido.")
            return False

        # EVO-TICKET-0006: warm_start must match the live candle interval.
        # Detector calibration assumes 5m candles; requesting 1m here would
        # feed 5x denser history and break the lookback/timeout semantics.
        url = (
            "https://api.binance.com/api/v3/klines"
            f"?symbol={self.symbol.upper()}&interval=5m&limit={lookback_bars}"
        )
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                raw = json.loads(resp.read().decode())
        except Exception as e:
            logger.warning(f"⚠️ Warm start REST falló para {self.symbol}: {e}")
            return False

        if not raw:
            logger.warning(f"⚠️ Warm start REST vacío para {self.symbol}.")
            return False

        # Binance klines columns:
        # 0 open_time, 1 open, 2 high, 3 low, 4 close, 5 volume,
        # 6 close_time, 7 quote_volume, 8 trades, 9 taker_buy_base, 10 taker_buy_quote, 11 ignore
        records = []
        for row in raw:
            records.append(
                {
                    "open_time": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "close_time": int(row[6]),
                    "quote_asset_volume": float(row[7]),
                    "trades": int(row[8]),
                }
            )

        df = pd.DataFrame(records)
        self.detector.seed_history(df)

        # Bootstrap: replay historical candles through zone detection so the
        # detector does not stay blind until a brand-new key candle forms.
        # Deduplication inside the detector prevents duplicate zones.
        # Skip index-based cleanup during bootstrap so zones detected early in
        # the historical window are not expired by the time we reach the end.
        self.detector.is_bootstrapping = True
        try:
            df = pd.DataFrame(records)
            self.detector.key_candle_detector.load_data(df)
            self.detector.zone_detector.load_data(df)
            self.detector.trend_detector.load_data(df)
            precomputed_trends = self.detector.trend_detector.detect_all()
            lookback = self.detector.config["lookback_candles"]
            timeout_bars = self.detector.config["retest_timeout_bars"]
            # Only replay the most recent candles that would still be alive.
            # This avoids detecting zones that are already beyond the timeout
            # and would be discarded by the final cleanup.
            start_idx = max(lookback, len(records) - timeout_bars)
            for idx in range(start_idx, len(records)):
                new_zones = self.detector._detect_new_zones(df, idx, precomputed_trends)
                self.detector._add_zones_without_duplicates(new_zones)
        finally:
            self.detector.is_bootstrapping = False

        # Sincronizar current_kline con la última vela histórica para evitar gaps
        last = records[-1]

        # Final cleanup after bootstrap: remove stale persisted zones using real
        # time and price distance. Use current_idx=None so recently detected
        # historical zones are not discarded by index-based timeout.
        self.detector._cleanup_expired_zones(
            current_idx=None,
            current_timestamp_ms=last["close_time"],
            current_price=last["close"],
        )

        self.current_kline = {
            "open_time": last["open_time"],
            "open": last["open"],
            "high": last["high"],
            "low": last["low"],
            "close": last["close"],
            "volume": last["volume"],
            "close_time": last["close_time"],
        }
        self._last_kline_close = last["open_time"]

        logger.info(
            f"🔥 Warm start completado para {self.symbol}: "
            f"{len(records)} velas de 1m hidratadas. "
            f"Zonas activas tras bootstrap: {len(self.detector.active_zones)}. "
            f"Última vela: {last['close']:.2f}"
        )

        # Persistir vista para la GUI inmediatamente
        self._persist_active_zones()

        # Emitir primer heartbeat tras warm_start exitoso
        self._export_heartbeat()

        return True

    # ══════════════════════════════════════════════════════════════
    # SPEED 2 (TICK): Entry point for every aggTrade (~10-50/s)
    # ══════════════════════════════════════════════════════════════

    async def on_ws_message(self, data: Dict[str, Any]):
        """Callback procesador de mensajes del WebSocket.

        EVO-TICKET-0003 fix: no longer depends exclusively on aggTrade.
        When aggTrade is absent for >30s (silent Binance Futures stream
        failure), depthUpdate timestamps drive candle close as fallback.
        """
        event_type = data.get("e")
        if event_type in ("aggTrade", "trade"):
            self._last_aggtrade_ts_ms = int(
                data.get("T", data.get("E", time.time() * 1000))
            )
            self._last_heartbeat_price = float(data["p"])
            await self._process_trade(data)
            self._export_heartbeat()
        elif event_type == "depthUpdate" or ("b" in data and "a" in data):
            # Heartbeat watchdog: use depthUpdate clock when aggTrade is dead
            depth_ts_ms = int(data.get("E", data.get("T", time.time() * 1000)))
            gap_ms = depth_ts_ms - self._last_aggtrade_ts_ms
            if gap_ms > self._heartbeat_timeout_ms:
                self._heartbeat_candle_close(depth_ts_ms)

    async def _process_trade(self, trade: Dict[str, Any]):
        """
        Procesa cada trade individual (aggTrade o trade). Dos responsabilidades:
        1. Agregar al candle actual (OHLCV aggregation)
        2. Check tick-level retest contra zonas activas
        """
        price = float(trade["p"])
        qty = float(trade["q"])
        ts = int(trade.get("T", trade.get("E", time.time() * 1000)))
        kline_start = (ts // (self.interval_s * 1000)) * (self.interval_s * 1000)

        # ── 1. OHLCV Candle Aggregation ──────────────────────────
        if kline_start > self._last_kline_close and self._last_kline_close > 0:
            # New candle → dispatch the previous one
            if self.current_kline:
                self._on_candle_close(self.current_kline)

            self.current_kline = {
                "open_time": kline_start,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": qty,
                "close_time": kline_start + (self.interval_s * 1000) - 1,
            }
            self._last_kline_close = kline_start
        elif self._last_kline_close == 0:
            self._last_kline_close = kline_start
            self.current_kline = {
                "open_time": kline_start,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": qty,
                "close_time": kline_start + (self.interval_s * 1000) - 1,
            }
        else:
            self.current_kline["high"] = max(self.current_kline["high"], price)
            self.current_kline["low"] = min(self.current_kline["low"], price)
            self.current_kline["close"] = price
            self.current_kline["volume"] += qty

        # ── 2. TICK-LEVEL RETEST CHECK (the critical change) ─────
        # O(n_zones) pure — no I/O, no DataFrame. ~5-10 zones × 30 ticks/s = trivial
        if self.detector.active_zones:
            hits = self.detector.check_intra_candle_retest(price, ts)

            for hit in hits:
                # THIS IS THE MOMENT — synthesize L2 profile NOW, not 45s later
                self._on_retest_detected(hit, price, ts)

        # ── 3. DEFERRED LABELING: update MFE/MAE on every tick ───
        # Lightweight: just float comparisons per pending label
        self.deferred_monitor.tick(price, bar_closed=False)

        # ── 4. Update open DryRun positions with current price ───
        self.order_mgr.update_positions(price)

    # ══════════════════════════════════════════════════════════════
    # HEARTBEAT WATCHDOG (EVO-TICKET-0003)
    # ══════════════════════════════════════════════════════════════

    def _heartbeat_candle_close(self, depth_ts_ms: int):
        """Fallback clock: close candle using depthUpdate timestamp
        when aggTrade stream has been silent > heartbeat_timeout.

        This prevents full pipeline freeze when Binance Futures stops
        distributing aggTrade while depth20@100ms continues flowing.
        Uses last known price from aggTrade or best bid from order book.
        """
        kline_start = (depth_ts_ms // (self.interval_s * 1000)) * (
            self.interval_s * 1000
        )

        # Only fire if we've actually crossed into a new candle period
        if kline_start <= self._last_kline_close:
            return

        # Derive price from order book best bid if no aggTrade price available
        price = self._last_heartbeat_price
        if price <= 0:
            state = self.ws.order_book_state.get(self.symbol.upper(), {})
            bids = state.get("bids", [])
            if bids:
                price = float(bids[0][0])
        if price <= 0:
            return  # No price source available at all

        logger.warning(
            f"💓 HEARTBEAT: aggTrade silent for "
            f"{(depth_ts_ms - self._last_aggtrade_ts_ms) / 1000:.0f}s. "
            f"Forcing candle close via depthUpdate clock. price={price:.2f}"
        )

        # Synthesize minimal candle from last known state
        if self.current_kline:
            self.current_kline["close"] = price
            self._on_candle_close(self.current_kline)

        # ── HEARTBEAT RETEST DETECTION ───────────────────────────
        # Critical: if we don't check retests here, the system is
        # blind to price touches during aggTrade outages.
        if self.detector.active_zones:
            hits = self.detector.check_intra_candle_retest(price, depth_ts_ms)
            for hit in hits:
                self._on_retest_detected(hit, price, depth_ts_ms)

        # Reset candle for next period
        self.current_kline = {
            "open_time": kline_start,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 0.0,
            "close_time": kline_start + (self.interval_s * 1000) - 1,
        }
        self._last_kline_close = kline_start

        # Tick deferred monitor with heartbeat price
        self.deferred_monitor.tick(price, bar_closed=False)

        # Update positions with heartbeat price
        self.order_mgr.update_positions(price)

        # Export heartbeat after recovery action
        self._export_heartbeat()

    # ══════════════════════════════════════════════════════════════
    # HEARTBEAT EXPORT (Watchdog Architecture)
    # ══════════════════════════════════════════════════════════════

    def _export_heartbeat(self):
        """
        Escribe heartbeat.json en aipha_memory/operational/ para watchdog externo.
        Llamada sin bloquear desde on_ws_message y _heartbeat_candle_close.
        """
        now = time.time()
        # Throttle: solo escribir cada _heartbeat_interval_s segundos
        if now - self._last_heartbeat_write_ts < self._heartbeat_interval_s:
            return

        project_root = Path(__file__).resolve().parent.parent.parent
        # Archivo específico por símbolo para evitar colisiones multi-activo
        heartbeat_path = project_root / "aipha_memory" / "operational" / f"heartbeat_{self.symbol}.json"
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            dataset_mtime = 0.0
            if _TRAINING_DATASET_V2.exists():
                dataset_mtime = _TRAINING_DATASET_V2.stat().st_mtime

            now_ms = int(now * 1000)
            aggtrade_gap = now_ms - self._last_aggtrade_ts_ms

            payload = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "ts_unix_ms": now_ms,
                "samples_full": self._count_full_samples(),
                "last_aggtrade_gap_ms": aggtrade_gap,
                "dataset_mtime": dataset_mtime,
                "active_zones_count": len(self.detector.active_zones) if self.detector else 0,
                "last_price": self._last_heartbeat_price,
                "symbol": self.symbol,
            }

            tmp_path = heartbeat_path.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=2)
            tmp_path.rename(heartbeat_path)

            self._last_heartbeat_write_ts = now
        except Exception as e:
            logger.warning(f"⚠️ heartbeat write failed: {e}")

    # ══════════════════════════════════════════════════════════════
    # SPEED 1 (1 MIN): Zone detection at candle close
    # ══════════════════════════════════════════════════════════════

    def _on_candle_close(self, kline: Dict[str, Any]):
        """
        Procesa la vela cerrada. Only two jobs:
        1. Feed detector for NEW zone detection (slow, requires DataFrame)
        2. Update NexusGate causal drift
        3. Resolve deferred labels (bar_closed=True)
        """
        logger.info(f"📊 Candle close procesada: precio={kline['close']:.0f}")
        obi = self.ws.get_current_obi(self.symbol, levels=10)
        cum_delta = self.ws.get_rolling_delta(self.symbol, window_seconds=300)
        micro_data = {
            "vwap": kline["close"],
            "obi_10": obi,
            "cumulative_delta": cum_delta,
            "timestamp": kline.get("close_time", int(time.time() * 1000)),
        }

        # 1. NexusGate & Causal Drift
        self.micro_buffer.append(micro_data)
        if len(self.micro_buffer) > 100:
            self.micro_buffer.pop(0)

        self.delta_causal = self.nexus.calculate_delta_causal(self.micro_buffer)
        self._is_causally_safe = self.nexus.is_safe(self.delta_causal)
        if self._disable_nexus_gate:
            self._is_causally_safe = True
            should_disable = False
            reason = ""
            if self._bypass_disable_mode == "full_count":
                full_samples = self._count_full_samples()
                should_disable = full_samples >= self._bypass_disable_full_target
                reason = (
                    f"FULL={full_samples} >= target {self._bypass_disable_full_target}"
                )
            else:
                # Default hardened mode: disable bypass only when Set A is truly ready.
                stats = self._compute_set_a_readiness()
                should_disable = bool(stats.get("ready"))
                reason = (
                    f"SetAReady={stats.get('ready')} "
                    f"(BOUNCE_STRONG={stats.get('bounce_strong')}/{self._set_a_min_bounce}, "
                    f"BREAKOUT={stats.get('breakout')}/{self._set_a_min_breakout}, "
                    f"FULL={stats.get('full_total')}/{self._set_a_min_full})"
                )

            if should_disable:
                self._disable_nexus_gate = False
                logger.warning(
                    f"🛑 NEXUSGATE BYPASS AUTO-DISABLED: {reason}. Gate reactivated."
                )

        logger.info(
            f"🕯️ Vela Live Cerrada: {self.symbol} "
            f"Close={kline['close']} OBI={obi:.4f} "
            f"ΔCausal={self.delta_causal:.2%} "
            f"Zonas={len(self.detector.active_zones)} "
            f"Pending={self.deferred_monitor.get_pending_count()}"
        )

        if not self._is_causally_safe:
            logger.warning(
                f"🚨 NEXUSGATE CLOSED: ΔCausal ({self.delta_causal:.4f}) > "
                f"Threshold ({self.nexus.threshold}). Señales suspendidas."
            )
        elif self._disable_nexus_gate:
            now = time.time()
            if now - self._last_bypass_warn_ts >= 300:
                self._last_bypass_warn_ts = now
                if self._bypass_disable_mode == "full_count":
                    detail = f"Auto-disable mode=full_count target FULL={self._bypass_disable_full_target}."
                else:
                    stats = self._compute_set_a_readiness()
                    detail = (
                        "Auto-disable mode=set_a_ready "
                        f"(BOUNCE_STRONG={stats.get('bounce_strong')}/{self._set_a_min_bounce}, "
                        f"BREAKOUT={stats.get('breakout')}/{self._set_a_min_breakout}, "
                        f"FULL={stats.get('full_total')}/{self._set_a_min_full})."
                    )
                logger.warning(f"⚠️ NEXUSGATE BYPASSED (harvest mode). {detail}")

        # 2. Feed closed candle to detector for ZONE DETECTION (Speed 1)
        self.detector.feed_kline_for_zone_detection(kline, micro_data)

        # Persist active zones AFTER zone detection so newly detected zones
        # appear in the GUI immediately.
        self._persist_active_zones()

        # 3. Deferred labeling: mark bar as closed (increment bars_elapsed)
        resolved = self.deferred_monitor.tick(kline["close"], bar_closed=True)
        if resolved:
            # Dataset may have changed; invalidate cached stats/counters.
            self._cached_full_count = -1
            self._set_a_cache_mtime = None
            self._set_a_cache_stats = None
        for r in resolved:
            logger.info(
                f"🏷️ Outcome resolved via candle close: "
                f"{r['sample_id']} → {r['outcome']}"
            )

    # ══════════════════════════════════════════════════════════════
    # RETEST HANDLER: The critical millisecond
    # ══════════════════════════════════════════════════════════════

    def _persist_active_zones(self):
        """
        Persiste el estado del detector y las zonas activas en disco.
        Usa ruta ABSOLUTA para garantizar que el servidor GUI lea el mismo archivo.
        GUI-readonly adapters skip writing — only the shadow trader owns the file.
        """
        if getattr(self, "_gui_readonly", False):
            return
        try:
            import traceback as _tb
            # 1. El detector guarda su estado completo (para auto-recuperación térmica)
            if hasattr(self.detector, "save_state"):
                self.detector.save_state()

            # 2. El adaptador genera la vista simplificada para la GUI (active_zones.json)
            import json
            from pathlib import Path

            # CRITICAL: Ruta absoluta calculada desde __file__, NO relativa al CWD
            project_root = Path(__file__).resolve().parent.parent.parent
            path = project_root / "aipha_memory" / "operational" / "active_zones.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            zones_data = []
            n_detector_zones = len(self.detector.active_zones)
            for z in self.detector.active_zones:
                effective_dir = (
                    ("bearish" if z.direction == "bullish" else "bullish")
                    if getattr(z, "polarity_flipped", False)
                    else z.direction
                )
                display_zone_id = z.zone_id or f"{z.candle_index}_{effective_dir}"
                zones_data.append(
                    {
                        "zone_id": display_zone_id,
                        "direction": z.direction,
                        "effective_direction": (
                            ("bearish" if z.direction == "bullish" else "bullish")
                            if getattr(z, "polarity_flipped", False)
                            else z.direction
                        ),
                        "original_direction": z.direction,
                        "polarity_flipped": getattr(z, "polarity_flipped", False),
                        "state": z.lifecycle_state.value
                        if hasattr(z.lifecycle_state, "value")
                        else z.lifecycle_state,
                        "zone_top": float(z.zone_top),
                        "zone_bottom": float(z.zone_bottom),
                        "touches": int(getattr(z, "touch_count", 0)),
                        "created_at": int(z.detection_timestamp),
                    }
                )
            try:
                # GUARD: never overwrite a non-empty file with [] unless detector truly has 0 zones
                if len(zones_data) == 0 and n_detector_zones > 0:
                    caller = _tb.format_stack(limit=6)[-3].strip()
                    logger.warning(f"⚠️ [PERSIST] SKIPPING write of [] — detector still has {n_detector_zones} zones! Caller: {caller}")
                    return
                with open(path, "w") as f:
                    json.dump(zones_data, f, indent=2)
                if len(zones_data) == 0 and n_detector_zones > 0:
                    caller = _tb.format_stack(limit=6)[-3].strip()
                    logger.warning(f"⚠️ [PERSIST] zones_data=[] but detector had {n_detector_zones} zones! Caller: {caller}")
                logger.info(
                    f"💾 Zonas GUI persistidas: {len(zones_data)} zonas → {path}"
                )
            except TypeError as e:
                logger.critical(
                    f"🔴 [PERSIST_ZONES] Serialización fallida — zona no guardada: {e}"
                )
                import sys

                print(f"🔴 CRITICAL PERSIST_ZONES: {e}", file=sys.stderr, flush=True)
            except IOError as e:
                logger.critical(
                    f"🔴 [PERSIST_ZONES] Error de disco — zona no guardada: {e}"
                )
                import sys

                print(f"🔴 CRITICAL IO_ZONES: {e}", file=sys.stderr, flush=True)

            # 4. Persist live signals for GUI dashboard (shadow trader's signals)
            safe_symbol = self.symbol.replace("/", "_").upper()
            signals_path = (
                project_root
                / "aipha_memory"
                / "operational"
                / f"live_signals_{safe_symbol}.json"
            )
            try:
                with open(signals_path, "w") as f:
                    json.dump(self.live_signals[-50:], f, indent=2)
            except (TypeError, IOError) as e:
                logger.warning(f"⚠️ [PERSIST_SIGNALS] Failed: {e}")
            #    Archivo separado por símbolo para evitar condición de carrera
            #    entre adaptadores multi-asset (BTC vs ETH).
            if self.current_kline and self.current_kline.get("close"):
                safe_symbol = self.symbol.replace("/", "_").upper()
                price_path = (
                    project_root
                    / "aipha_memory"
                    / "operational"
                    / f"market_price_{safe_symbol}.json"
                )
                price_data = {
                    "symbol": self.symbol,
                    "price": float(self.current_kline["close"]),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                try:
                    with open(price_path, "w") as f:
                        json.dump(price_data, f)
                except TypeError as e:
                    logger.critical(
                        f"🔴 [PERSIST_ZONES] Serialización fallida — zona no guardada (price): {e}"
                    )
                    import sys

                    print(
                        f"🔴 CRITICAL PERSIST_ZONES: {e}", file=sys.stderr, flush=True
                    )
                except IOError as e:
                    logger.critical(
                        f"🔴 [PERSIST_ZONES] Error de disco — zona no guardada (price): {e}"
                    )
                    import sys

                    print(f"🔴 CRITICAL IO_ZONES: {e}", file=sys.stderr, flush=True)
        except Exception as e:
            if not isinstance(e, (TypeError, IOError)):
                logger.error(f"Error persisting active zones: {e}", exc_info=True)
            else:
                pass

    def _on_retest_detected(self, hit: dict, price: float, timestamp_ms: int):
        """
        Called the INSTANT a tick touches a zone. This is where we:
        1. Synthesize the L2 temporal profile (30s ring buffer)
        2. Build the full ReentrySnapshot
        3. Run Oracle prediction
        4. Open ShadowTrade if confidence > threshold
        5. Register with DeferredOutcomeMonitor for labeling
        """
        zone = hit["zone"]
        clearance_atr = hit["clearance_atr"]
        retest_type = hit.get("retest_type", "ZONE_INTERIOR")
        retest_penalty = (
            self._proximity_penalty if retest_type == "PROXIMITY_BUFFER" else 1.0
        )

        # ── 1. SYNTHESIZE L2 PROFILE (the whole point of this refactor) ──
        sym_lower = self.symbol.lower()
        l2_buffer = self.ws.l2_buffers.get(sym_lower)

        if l2_buffer:
            l2_profile = l2_buffer.synthesize_at_retest(
                t_candle_close_ms=self.current_kline.get("close_time"),
                epsilon_ms=200
            )
            raw_buffer = l2_buffer.get_raw_buffer()
        else:
            l2_profile = {}
            raw_buffer = []

        # ── 2. CAPTURE L2 SNAPSHOT AT TOUCH ──────────────────────
        obi_10 = self.ws.get_current_obi(self.symbol, levels=10)
        obi_1 = self.ws.get_current_obi(self.symbol, levels=1)
        obi_5 = self.ws.get_current_obi(self.symbol, levels=5)
        obi_20 = self.ws.get_current_obi(self.symbol, levels=20)
        cum_delta = self.ws.get_rolling_delta(self.symbol, window_seconds=300)

        # Book depth
        state = self.ws.order_book_state.get(self.symbol, {})
        bids = state.get("bids", [])
        asks = state.get("asks", [])
        best_bid_size = float(bids[0][1]) if bids else 0.0
        best_ask_size = float(asks[0][1]) if asks else 0.0
        bid_depth_10 = sum(float(b[1]) for b in bids[:10])
        ask_depth_10 = sum(float(a[1]) for a in asks[:10])
        best_bid = float(bids[0][0]) if bids else 0.0
        best_ask = float(asks[0][0]) if asks else 0.0
        spread_bps = ((best_ask - best_bid) / best_bid * 10000) if best_bid > 0 else 0.0

        # ── 3. BUILD REENTRY SNAPSHOT (Schema v2.0) ──────────────
        zone_width = zone.zone_top - zone.zone_bottom
        atr = zone.atr_at_detection if zone.atr_at_detection > 0 else price * 0.001

        sample_id = (
            f"re_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            f"_{self.symbol}_{zone.direction}_{zone.candle_index}"
        )

        snapshot = {
            "_meta": {
                "schema_version": "2.0.0",
                "capture_ts_utc": datetime.now(timezone.utc).isoformat(),
                "capture_ts_unix_ms": timestamp_ms,
                "symbol": self.symbol,
                "sample_id": sample_id,
                "pipeline_version": "v4.2-tick-level",
            },
            "zone_geometry": {
                "zone_id": f"z_{zone.candle_index}_{zone.direction}",
                "direction": zone.direction,
                "zone_top": zone.zone_top,
                "zone_bottom": zone.zone_bottom,
                "zone_width_abs": zone_width,
                "zone_width_atr": round(zone_width / atr, 4),
                "detection_candle_index": zone.candle_index,
                "detection_ts_utc": datetime.fromtimestamp(
                    zone.detection_timestamp / 1000, tz=timezone.utc
                ).isoformat()
                if zone.detection_timestamp > 1e9
                else None,
                "key_candle_volume_ratio": zone.key_candle.get("volume_percentile", 0),
                "key_candle_body_pct": zone.key_candle.get("body_percentage", 0),
                "key_candle_upper_wick_pct": zone.key_candle.get("upper_wick_pct", 0),
                "key_candle_lower_wick_pct": zone.key_candle.get("lower_wick_pct", 0),
                "key_candle_rejection": zone.key_candle.get(
                    "rejection_direction", "NONE"
                ),
                "accumulation_bar_count": (
                    zone.accumulation_zone.get("end_idx", 0)
                    - zone.accumulation_zone.get("start_idx", 0)
                ),
                "mini_trend_r2": zone.mini_trend.get("r2", 0),
                "coincidence_score": zone.accumulation_zone.get("quality_score", 0),
            },
            "clearance": {
                "atr_at_detection": atr,
                "max_clearance_price": (
                    zone.max_price_since_detection
                    if zone.direction == "bullish"
                    else zone.min_price_since_detection
                ),
                "max_clearance_atr": clearance_atr,
                "seconds_since_detection": (
                    (timestamp_ms - zone.detection_timestamp) / 1000
                    if zone.detection_timestamp > 0
                    else 0
                ),
            },
            "l2_snapshot_at_touch": {
                "retest_price": price,
                "obi_1": obi_1,
                "obi_5": obi_5,
                "obi_10": obi_10,
                "obi_20": obi_20,
                "cumulative_delta": cum_delta,
                "best_bid_size_btc": best_bid_size,
                "best_ask_size_btc": best_ask_size,
                "bid_wall_depth_10_btc": bid_depth_10,
                "ask_wall_depth_10_btc": ask_depth_10,
                "spread_bps": round(spread_bps, 2),
                "vwap_session": self.current_kline.get("close", price),
                "retest_type": retest_type,
                "retest_type_penalty_applied": retest_penalty,
            },
            "l2_temporal_profile": l2_profile,
            "market_context": {
                "atr_14_current": atr,
                "regime": "LATERAL",  # Will be enriched by detector if kline buffer has data
                "session": self._get_session(timestamp_ms),
                "hour_utc": datetime.fromtimestamp(
                    timestamp_ms / 1000, tz=timezone.utc
                ).hour,
            },
            "outcome": None,  # Filled by DeferredOutcomeMonitor
        }

        # ── 4. ORACLE PREDICTION ─────────────────────────────────
        confidence = 0.5
        prediction = "PENDING"

        if (
            self._oracle
            and self._oracle.model
            and self._oracle.model != "placeholder_model_trained"
        ):
            try:
                signal_data = {
                    "vwap_at_retest": price,
                    "obi_10_at_retest": obi_10,
                    "cumulative_delta_at_retest": cum_delta,
                    "delta_divergence": (
                        "CONFIRMED"
                        if (zone.direction == "bullish" and cum_delta > 0)
                        or (zone.direction == "bearish" and cum_delta < 0)
                        else "DIVERGENT"
                    ),
                    "atr_14": atr,
                    "regime": "LATERAL",
                    "direction": zone.direction,
                    "index": zone.candle_index,
                    "zone_width_atr": round(zone_width / atr, 4),
                    "zone_top": zone.zone_top,
                    "zone_bottom": zone.zone_bottom,
                }
                oracle_result = self._oracle.predict(
                    micro=None, signal_data=signal_data
                )
                confidence = oracle_result.confidence
                confidence *= retest_penalty
                prediction = oracle_result.suggested_action
            except Exception as e:
                logger.warning(f"⚠️ Oracle prediction failed: {e}")

        # ── 4.5. REGRESIÓN MAE (CAPA 2) ──────────────────────────
        predicted_mae_atr = 0.0
        limit_price = price
        is_safe_by_mae = True
        mae_reason = "No MAE Regressor"

        if confidence > 0.70 and prediction == "EXECUTE" and self._oracle_regressor:
            try:
                mae_pred = self._oracle_regressor.predict_mae(
                    micro=None, signal_data=signal_data
                )
                predicted_mae_atr = mae_pred.predicted_mae_atr
                limit_price = mae_pred.limit_price
                is_safe_by_mae = mae_pred.is_safe
                mae_reason = mae_pred.reason
            except Exception as e:
                logger.warning(f"⚠️ MAE regression failed: {e}")

        # ── 5. BUILD SIGNAL & EXECUTE ────────────────────────────
        signal = {
            "id": sample_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": self.symbol,
            "price": price,
            "direction": zone.direction,
            "oracle_confidence": confidence,
            "prediction": prediction,
            "obi": obi_10,
            "l2_quality": l2_profile.get("l2_data_quality", "UNKNOWN"),
            "l2_snapshots": l2_profile.get("n_snapshots", 0),
            "clearance_atr": clearance_atr,
            "status": "active",
            "predicted_mae_atr": predicted_mae_atr,
            "limit_price": limit_price,
            "mae_reason": mae_reason,
        }
        self.live_signals.append(signal)

        # Trim signal history
        if len(self.live_signals) > 50:
            self.live_signals.pop(0)

        # Execute DryRun if Oracle approves
        is_secondary_retest = False
        if hasattr(zone, "lifecycle_state"):
            # Using value explicitly in case it's an Enum string
            is_secondary_retest = (
                getattr(zone, "lifecycle_state") == "harvesting"
                or getattr(zone, "lifecycle_state").value == "harvesting"
            )

        if not is_secondary_retest:
            if confidence > 0.70 and is_safe_by_mae and self._is_causally_safe:
                self.order_mgr.execute_signal(signal)
            elif confidence > 0.70 and not is_safe_by_mae:
                logger.warning(
                    f"🚫 Señal abortada por Capa 2 (MAE Seguridad): {mae_reason}"
                )
            elif confidence > 0.70 and not self._is_causally_safe:
                logger.warning(
                    f"🚫 Señal abortada por NEXUSGATE (ΔCausal={self.delta_causal:.4f}). Cosechando muestra..."
                )
        else:
            logger.info(
                f"🌾 [ShadowHarvest] Toque secuencial en {zone.zone_id} (polarity_flipped={getattr(zone, 'polarity_flipped', False)}). Sin ejecución viva."
            )

        # Registrar toque en la zona (actualiza touch_count y touch_history)
        if hasattr(zone, "register_touch"):
            assigned_seq = zone.register_touch(
                price=price, obi=obi_10, cum_delta=cum_delta
            )
        else:
            assigned_seq = 1

        # ── 6. REGISTER WITH DEFERRED MONITOR ────────────────────
        snapshot["_meta"]["touch_sequence"] = assigned_seq
        snapshot["_meta"]["polarity_flipped"] = getattr(zone, "polarity_flipped", False)
        self.deferred_monitor.register_retest(
            snapshot,
            raw_buffer=raw_buffer,
            touch_sequence=assigned_seq,
            polarity_flipped=getattr(zone, "polarity_flipped", False),
            prior_touch_outcomes=getattr(zone, "prior_outcomes", []),
            zone_original_direction=getattr(zone, "direction", ""),
            flip_ts=getattr(zone, "flip_ts", None),
        )

        # ── 7. LOG ───────────────────────────────────────────────
        logger.info(
            f"🚨 RETEST DETECTADO (tick-level): {zone.direction} "
            f"@ {price:.2f} | Zone [{zone.zone_bottom:.2f}-{zone.zone_top:.2f}] "
            f"| Clearance {clearance_atr:.2f} ATR "
            f"| OBI={obi_10:.4f} | CumΔ={cum_delta:.1f} "
            f"| RetestType={retest_type} Penalty={retest_penalty:.2f} "
            f"| L2 snaps={l2_profile.get('n_snapshots', 0)} "
            f"| Oracle conf={confidence:.2f} ({prediction}) "
            f"| Sample={sample_id}"
        )

        # ── 8. PERSIST STATE FOR GUI ───────────────────────────────
        self._persist_active_zones()

    # ══════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _get_session(timestamp_ms: int) -> str:
        """Determina la sesión de trading basada en hora UTC."""
        hour = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).hour
        if 0 <= hour < 8:
            return "ASIA"
        elif 8 <= hour < 14:
            return "EU"
        elif 14 <= hour < 21:
            return "US"
        else:
            return "LATE_US"

    def _count_full_samples(self) -> int:
        """Count FULL-quality entries in training_dataset_v2.jsonl."""
        # Incremental path
        if self._use_incremental_counter and self._cached_full_count != -1:
            return self._cached_full_count

        if not _TRAINING_DATASET_V2.exists():
            if self._use_incremental_counter:
                self._cached_full_count = 0
            return 0

        total = 0
        try:
            with open(_TRAINING_DATASET_V2, "r") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    quality = (row.get("l2_temporal_profile") or {}).get(
                        "l2_data_quality"
                    )
                    if quality == "FULL":
                        total += 1
        except OSError:
            return 0

        if self._use_incremental_counter:
            self._cached_full_count = total

        return total

    def _compute_set_a_readiness(self) -> Dict[str, Any]:
        """
        Compute Set A readiness from dataset v2:
        - full_total >= min_full
        - bounce_strong >= min_bounce
        - breakout >= min_breakout
        """
        if not _TRAINING_DATASET_V2.exists():
            return {"full_total": 0, "bounce_strong": 0, "breakout": 0, "ready": False}

        try:
            mtime = _TRAINING_DATASET_V2.stat().st_mtime
        except OSError:
            return {"full_total": 0, "bounce_strong": 0, "breakout": 0, "ready": False}

        if self._set_a_cache_mtime == mtime and self._set_a_cache_stats is not None:
            return self._set_a_cache_stats

        full_total = 0
        bounce_strong = 0
        breakout = 0
        try:
            with open(_TRAINING_DATASET_V2, "r") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    quality = (row.get("l2_temporal_profile") or {}).get(
                        "l2_data_quality"
                    )
                    if quality != "FULL":
                        continue
                    full_total += 1
                    label = ((row.get("outcome") or {}).get("label") or "").upper()
                    if label == "BOUNCE_STRONG":
                        bounce_strong += 1
                    elif label == "BREAKOUT":
                        breakout += 1
        except OSError:
            return {"full_total": 0, "bounce_strong": 0, "breakout": 0, "ready": False}

        ready = (
            full_total >= self._set_a_min_full
            and bounce_strong >= self._set_a_min_bounce
            and breakout >= self._set_a_min_breakout
        )
        stats = {
            "full_total": full_total,
            "bounce_strong": bounce_strong,
            "breakout": breakout,
            "ready": ready,
        }
        self._set_a_cache_mtime = mtime
        self._set_a_cache_stats = stats
        return stats

    @classmethod
    def create_default(cls, ws_manager, detector, order_mgr=None, symbol="BTCUSDT"):
        manifest = ComponentManifest(
            name="LiveDataFeedAdapter",
            category="application",
            function="Two-Speed Live Pipeline: Zones@1min, Retests@tick, L2@retest",
            inputs=["WS"],
            outputs=["Signals", "ReentrySnapshots"],
            causal_score=0.95,
        )
        adapter = cls(manifest, ws_manager, detector, order_mgr)
        adapter.symbol = symbol
        return adapter
