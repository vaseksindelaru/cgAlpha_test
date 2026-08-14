"""
CGAlpha v3 — ShadowTrader Live (Fase 3 Dispatcher)
=================================================
Lanza el sistema en modo monitoreo en vivo sobre BTCUSDT.
Conecta WebSocket -> LiveAdapter -> SignalDetector.
"""

import asyncio
import logging
import signal
import sys
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

# Ensure project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from cgalpha_v3.infrastructure.binance_websocket_manager import BinanceWebSocketManager
from cgalpha_v3.infrastructure.signal_detector.triple_coincidence import TripleCoincidenceDetector
from cgalpha_v3.application.live_adapter import LiveDataFeedAdapter
from cgalpha_v3.lila.llm.oracle import OracleTrainer_v3, OracleRegressor_MAE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("shadow_live")
ACTIVE_WS_MANAGER = None

# ─────────────────────────────────────────────────────────────────────
# MÓDULO DE ARRANQUE EN FRÍO (Historical Pre-load)
# ─────────────────────────────────────────────────────────────────────

BOOTSTRAP_LIMIT     = 1000
BOOTSTRAP_INTERVAL  = "5m"

def _ts_to_str(unix_ms: int | str) -> str:
    return datetime.fromtimestamp(int(unix_ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def _fetch_historical_klines(symbol: str, interval: str = BOOTSTRAP_INTERVAL, limit: int = BOOTSTRAP_LIMIT, retries: int = 3) -> list[list]:
    market = os.environ.get("CGALPHA_BINANCE_MARKET", "futures").strip().lower()
    if market == "spot":
        base = "https://api.binance.com"
        path = "/api/v3/klines"
    else:
        base = "https://fapi.binance.com"
        path = "/fapi/v1/klines"
    url = f"{base}{path}"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            raw: list[list] = resp.json()
            if not raw:
                raise ValueError("Binance devolvió un array vacío de klines.")
            now_ms = int(time.time() * 1000)
            last_close_ms = int(raw[-1][6])
            if last_close_ms > now_ms:
                raw = raw[:-1]
                logger.info("Bootstrap: última vela aún abierta — excluida del pre-load.")
            logger.info(f"Bootstrap: {len(raw)} velas cerradas descargadas ({symbol} {interval}) — desde {_ts_to_str(raw[0][0])} hasta {_ts_to_str(raw[-1][6])}.")
            return raw
        except requests.RequestException as exc:
            logger.warning(f"Bootstrap: intento {attempt}/{retries} fallido — {exc}")
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Bootstrap: no se pudo descargar klines tras {retries} intentos.") from exc
    return []

def _klines_to_ohlcv_df(raw_klines: list[list]) -> pd.DataFrame:
    columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(raw_klines, columns=columns)
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["open_time"]  = df["open_time"].astype(int)
    df["close_time"] = df["close_time"].astype(int)
    df["n_trades"]   = df["n_trades"].astype(int)
    df = df.reset_index(drop=True)
    return df

def _build_synthetic_micro_df(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    n = len(ohlcv_df)
    return pd.DataFrame({
        "vwap":               ohlcv_df["close"].values,
        "obi_10":             [0.0] * n,
        "cumulative_delta":   [0.0] * n,
        "atr":                [0.0] * n,
        "delta_divergence":   ["NEUTRAL"] * n,
        "regime":             ["LATERAL"] * n,
    })

def _generate_synthetic_klines(n: int) -> list[list]:
    import random
    now_ms   = int(time.time() * 1000)
    interval = 5 * 60 * 1000
    price    = 97_000.0
    klines = []
    for i in range(n):
        open_time = now_ms - (n - i) * interval
        close_time = open_time + interval - 1
        ret = random.gauss(0, 0.001)
        o = price
        c = round(price * (1 + ret), 2)
        h = round(max(o, c) * (1 + abs(random.gauss(0, 0.0005))), 2)
        lo = round(min(o, c) * (1 - abs(random.gauss(0, 0.0005))), 2)
        vol = round(random.uniform(5, 50), 4)
        price = c
        klines.append([open_time, str(o), str(h), str(lo), str(c), str(vol), close_time, str(vol * c), random.randint(100, 800), str(vol * 0.55), str(vol * 0.55 * c), "0"])
    return klines

def bootstrap_detector(detector, symbol: str = "BTCUSDT", interval: str = BOOTSTRAP_INTERVAL, limit: int = BOOTSTRAP_LIMIT, dry_run: bool = False) -> dict:
    market = os.environ.get("CGALPHA_BINANCE_MARKET", "futures").strip().lower()
    t0 = time.monotonic()
    logger.info(f"━━━ BOOTSTRAP START — {symbol} {interval} (limit={limit}) ━━━")
    if dry_run:
        logger.warning("Bootstrap DRY-RUN: usando datos sintéticos (sin Binance).")
        raw_klines = _generate_synthetic_klines(limit)
    else:
        raw_klines = _fetch_historical_klines(symbol, interval, limit)

    if not raw_klines:
        logger.error("Bootstrap abortado: no hay klines disponibles.")
        return {"candles_loaded": 0, "zones_created": 0, "error": "no_data"}

    ohlcv_df  = _klines_to_ohlcv_df(raw_klines)
    micro_df  = _build_synthetic_micro_df(ohlcv_df)
    from_ts = _ts_to_str(ohlcv_df["open_time"].iloc[0])
    to_ts   = _ts_to_str(ohlcv_df["close_time"].iloc[-1])
    zones_before = len(getattr(detector, "active_zones", {}))

    try:
        # 1. Hidratar el historial estadístico del detector para evitar Cold Start
        if hasattr(detector, "seed_history"):
            detector.seed_history(ohlcv_df)

        training_samples = detector.process_stream(ohlcv_df, micro_df)
        logger.info(f"Bootstrap: process_stream completado. Samples generados: {len(training_samples) if training_samples else 0}.")
    except Exception as exc:
        logger.error(f"Bootstrap: process_stream falló — {exc}", exc_info=True)
        return {"candles_loaded": len(ohlcv_df), "zones_created": 0, "error": str(exc)}

    active_zones = getattr(detector, "active_zones", [])
    zones_created = len(active_zones) - zones_before
    zone_ids = [f"{market}_{z}" for z in active_zones]
    duration = round(time.monotonic() - t0, 2)

    report = {"candles_loaded": len(ohlcv_df), "zones_created": zones_created, "zones_total": len(active_zones), "from_ts": from_ts, "to_ts": to_ts, "duration_s": duration, "active_zone_ids": zone_ids}

    logger.info(f"━━━ BOOTSTRAP COMPLETE ━━━\n"
                f"  Velas cargadas : {report['candles_loaded']}\n"
                f"  Zonas creadas  : {report['zones_created']}\n"
                f"  Zonas activas  : {report['zones_total']}\n"
                f"  Período        : {from_ts} → {to_ts}\n"
                f"  Duración       : {duration}s\n"
                f"  IDs de zonas   : {zone_ids or '(ninguna detectada aún)'}")

    if report["zones_total"] == 0:
        logger.warning("Bootstrap: 0 zonas activas tras el warm-up. El LiveAdapter esperará a que el detector las construya con los ticks del WebSocket.")
    else:
        # Fix: reset detection_timestamp to NOW so _cleanup_expired_zones
        # doesn't immediately expire zones that were detected on historical candles.
        now_ms = int(time.time() * 1000)
        for z in detector.active_zones:
            z.detection_timestamp = now_ms
        logger.info(f"Bootstrap: {report['zones_total']} zonas re-timestamped to NOW ({now_ms}) para evitar expiry inmediato.")

    return report

async def shutdown(loop, ws_manager, signal=None):
    """Cleanup gracefully."""
    global ACTIVE_WS_MANAGER
    if signal:
        logger.info(f"🛑 Señal recibida: {signal.name}...")
    
    logger.info("🔌 Cerrando conexiones...")
    manager = ws_manager if ws_manager is not None else ACTIVE_WS_MANAGER
    if manager is not None:
        try:
            await manager.stop()
        except Exception as exc:
            logger.warning(f"⚠️ Error al detener ws_manager: {exc}")
    else:
        logger.warning("⚠️ Shutdown sin ws_manager activo; se omite stop().")
    
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    
    logger.info(f"🧹 Cancelando {len(tasks)} tareas pendientes...")
    await asyncio.gather(*tasks, return_exceptions=True)
    ACTIVE_WS_MANAGER = None
    loop.stop()
    logger.info("👋 ShadowTrader detenido.")

async def main():
    global ACTIVE_WS_MANAGER
    MARKET   = os.environ.get("CGALPHA_BINANCE_MARKET", "futures").strip().lower()
    print("=" * 72)
    print("  🌌 CGAlpha v3 — MODO SHADOWTRADER LIVE")
    print(f"  Operando sobre: BTCUSDT (Binance {MARKET.upper()})")
    print("=" * 72)

    # 1. Cargar el Oracle entrenado (de Phase 10)
    logger.info("🧠 Cargando Oracle (RandomForest calibrado v5)...")
    oracle = OracleTrainer_v3.create_default()
    model_path = PROJECT_ROOT / "aipha_memory/data/models/oracle_v5.joblib"
    if model_path.exists():
        oracle.load_from_disk(str(model_path))
        logger.info(f"✅ Modelo Oracle v5 cargado exitosamente desde {model_path.name}")
    else:
        logger.warning(f"⚠️ No se encontró modelo en {model_path}. Usando placeholder.")
    
    # 1.5 Cargar el Oracle Regressor MAE (Capa 2)
    logger.info("🎯 Cargando Oracle Regressor MAE (Capa 2)...")
    regressor = OracleRegressor_MAE.create_default()
    mae_model_path = PROJECT_ROOT / "aipha_memory/data/models/oracle_mae_v1.joblib"
    if mae_model_path.exists():
        regressor.load_from_disk(str(mae_model_path))
        logger.info(f"✅ Modelo MAE Regressor cargado exitosamente desde {mae_model_path.name}")
    else:
        logger.warning(f"⚠️ No se encontró modelo en {mae_model_path}. Usando placeholder.")
    
    # 2. Inicializar componentes y Bootstrap
    detector = TripleCoincidenceDetector()
    
    SYMBOL   = os.environ.get("CGALPHA_SYMBOL", "BTCUSDT")
    MARKET   = os.environ.get("CGALPHA_BINANCE_MARKET", "futures").strip().lower()
    DRY_RUN  = os.environ.get("CGALPHA_BOOTSTRAP_DRY_RUN", "false").lower() == "true"
    WARMUP_N = int(os.environ.get("CGALPHA_BOOTSTRAP_LIMIT", "500"))
    logger.info(f"🌐 Mercado WebSocket/REST seleccionado: {MARKET.upper()}")

    # Bootstrapping (Arranque en Frío)
    bootstrap_report = bootstrap_detector(
        detector  = detector,
        symbol    = SYMBOL,
        interval  = BOOTSTRAP_INTERVAL,
        limit     = WARMUP_N,
        dry_run   = DRY_RUN,
    )

    if bootstrap_report.get("error"):
        logger.error(f"Bootstrap fallido: {bootstrap_report['error']}. Entorrno pasará a cold start ciego.")

    ws_manager = BinanceWebSocketManager.create_default()
    ACTIVE_WS_MANAGER = ws_manager
    adapter = LiveDataFeedAdapter.create_default(ws_manager, detector)
    adapter.inject_oracle(oracle)
    adapter.inject_regressor(regressor)

    # Conectar el stream de eventos al adapter
    ws_manager.add_callback(adapter.on_ws_message)

    logger.info("💾 Persistiendo zonas activas fundacionales (Bootstrap)...")
    if hasattr(adapter, "_persist_active_zones"):
        adapter._persist_active_zones()

    # 3. Iniciar bucle
    logger.info("🚀 Iniciando WebSocket Stream...")
    await ws_manager.start()
    
    print()
    print("  [MONITOR ACTIVO] Esperando datos de mercado...")
    print("  (Presiona Ctrl+C para detener)")
    print("-" * 72)

    # Mantener el script vivo
    try:
        while True:
            # Aquí podríamos imprimir estadísticas cada N segundos
            obi = ws_manager.get_current_obi("BTCUSDT")
            # Log sutil cada 10 segundos
            # logger.info(f"📊 Market Pulse: BTCUSDT | OBI: {obi:.4f}")
            await asyncio.sleep(10)
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    
    # Manejar señales de interrupción
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(
            s, lambda s=s: asyncio.create_task(shutdown(loop, None, signal=s))
        )

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
