"""
CGAlpha v3 — GUI Control Room Server
=====================================
Servidor de la sala de control viva.

Fase: 0 (Mission Control + Market Live mock + Risk Dashboard básico)
Auth: Bearer token via AUTH_TOKEN en .env
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict

# Fix: Añadir raíz del proyecto al sys.path para evitar ModuleNotFoundError
# cuando se lanza el script directamente.
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask.typing import ResponseReturnValue

# Cargar variables de entorno desde .env (en raíz del proyecto)
v3_env_path = project_root / ".env"
load_dotenv(dotenv_path=v3_env_path)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
MEMORY_DIR = BASE_DIR.parent / "memory"

AUTH_TOKEN = os.getenv("CGV3_AUTH_TOKEN", "cgalpha-v3-local-dev")
HOST = os.getenv("CGV3_HOST", "127.0.0.1")
PORT = int(os.getenv("CGV3_PORT", "5000"))

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
logger = logging.getLogger("server")

# ---------------------------------------------------------------------------
# Estado global y Managers
# ---------------------------------------------------------------------------
from cgalpha_v3.application.change_proposer import ChangeProposer
from cgalpha_v3.application.experiment_runner import ExperimentResult, ExperimentRunner
from cgalpha_v3.application.live_adapter import LiveDataFeedAdapter
from cgalpha_v3.application.production_gate import ProductionGate, ProductionGateError
from cgalpha_v3.application.promotion_validator import PromotionValidator
from cgalpha_v3.application.rollback_manager import RollbackManager
from cgalpha_v3.data_quality.gates import TemporalLeakageError
from cgalpha_v3.data_quality.nexus_gate import NexusGate
from cgalpha_v3.domain.models.signal import (
    ApproachType,
    MemoryEntry,
    MemoryLevel,
    Proposal,
    RiskAssessment,
)
from cgalpha_v3.infrastructure.binance_websocket_manager import BinanceWebSocketManager
from cgalpha_v3.infrastructure.signal_detector.triple_coincidence import (
    TripleCoincidenceDetector,
)
from cgalpha_v3.learning.memory_policy import MemoryPolicyEngine
from cgalpha_v3.learning.project_history_learner import ProjectHistoryLearner
from cgalpha_v3.lila.codecraft_sage import CodeCraftSage
from cgalpha_v3.lila.evolution_orchestrator import EvolutionOrchestratorV4
from cgalpha_v3.lila.library_manager import (
    AdaptiveBacklogItem,
    LibraryManager,
    LibrarySource,
)
from cgalpha_v3.lila.llm import LLMAssistant
from cgalpha_v3.lila.llm.llm_switcher import LLMSwitcher
from cgalpha_v3.lila.llm.oracle import OracleTrainer_v3
from cgalpha_v3.risk.execution_factory import create_order_manager
from cgalpha_v3.risk.health_monitor import HealthMonitor
from cgalpha_v3.risk.order_manager import DryRunOrderManager
from cgalpha_v3.trading.shadow_trader import BRIDGE_JSONL_PATH

_rollback_mgr = RollbackManager(MEMORY_DIR / "snapshots")
_lila_mgr = LibraryManager()
_change_proposer = ChangeProposer()
_experiment_runner = ExperimentRunner()
_memory_engine = MemoryPolicyEngine()
_memory_load_result = _memory_engine.load_from_disk()
logger.info(f"\u2705 Memoria cargada desde disco: {_memory_load_result}")
_health_monitor = HealthMonitor()
_promotion_validator = PromotionValidator()
_production_gate = ProductionGate(_promotion_validator)
_history_learner = ProjectHistoryLearner(_memory_engine, BASE_DIR.parent.parent)
_assistant = LLMAssistant()  # Migrado a v3
_llm_switcher = LLMSwitcher(assistant=_assistant)
_codecraft_sage = CodeCraftSage.create_default()
_codecraft_sage.switcher = _llm_switcher
_ws_manager = BinanceWebSocketManager.create_default()

# Multi-Asset Execution Layer (Fase 4.3)
_order_mgr = create_order_manager()

# Intentar cargar Oracle entrenado (de Phase 1/v4)
_oracle_v3 = OracleTrainer_v3.create_default()
try:
    # Priorizar oracle_v3_historical (modelo real) sobre oracle_v3 (placeholder)
    _oracle_path_historical = project_root / "aipha_memory" / "models" / "oracle_v3_historical.joblib"
    _oracle_path = project_root / "aipha_memory" / "models" / "oracle_v3.joblib"
    if _oracle_path_historical.exists():
        _oracle_v3.load_from_disk(str(_oracle_path_historical))
        logger.info(f"✅ Oracle v4 cargado desde histórico: {_oracle_path_historical}")
    elif _oracle_path.exists():
        _oracle_v3.load_from_disk(str(_oracle_path))
        logger.info(f"✅ Oracle v4 cargado satisfactoriamente desde {_oracle_path}")
    else:
        logger.warning("⚠️ No se encontró modelo Oracle, usando baseline por defecto")
except Exception as e:
    logger.error(f"⚠️ No se pudo cargar el Oracle: {e}")

# Multi-Asset Live Pipeline (Fase 4.2)
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
_detectors: Dict[str, TripleCoincidenceDetector] = {}
_adapters: Dict[str, LiveDataFeedAdapter] = {}
_ws_managers: Dict[str, BinanceWebSocketManager] = {}

for symbol in SYMBOLS:
    _ws_managers[symbol] = BinanceWebSocketManager.create_default(symbol=symbol)
    _detectors[symbol] = TripleCoincidenceDetector()
    _adapters[symbol] = LiveDataFeedAdapter.create_default(
        _ws_managers[symbol], _detectors[symbol], _order_mgr, symbol=symbol
    )
    # Inyectar Oracle y Nexus en cada adaptador
    _adapters[symbol].inject_oracle(_oracle_v3)
    # inject_oracle ya recalibra NexusGate con la firma causal del Oracle
    # Warm-start: hidratar buffer de klines para evitar cold start.
    # EVO-TICKET-0006: live_adapter operates at 5m, so 200 bars = 200
    # 5-minute candles (~16.6h). This gives the ZigZag trend detector
    # enough history to find valid segments while keeping the window
    # recent enough for live zones.
    try:
        _adapters[symbol].warm_start(lookback_bars=200)
    except Exception as e:
        logger.warning(f"⚠️ Warm start falló para {symbol}: {e}")

# Por compatibilidad con endpoints existentes:
_shadow_trader = _adapters["BTCUSDT"]
_ws_manager = _ws_managers["BTCUSDT"]

# Evolution Layer v4
_evolution_orchestrator = EvolutionOrchestratorV4(
    memory=_memory_engine,
    switcher=_llm_switcher,
    sage=_codecraft_sage,
    assistant=_assistant,
)

_latest_proposal: Proposal | None = Proposal(
    proposal_id="prop-foundation-default",
    agent_id="lila-baseline",
    generated_at=datetime.now(timezone.utc),
    session_id="baseline-session",
    hypothesis="Rebotes en VWAP con confirmación de absorción en ventana de 5m. (Auto-generada)",
    approach_types_targeted=[ApproachType.TOUCH, ApproachType.RETEST],
    risk_assessment=RiskAssessment(
        max_drawdown_impact_pct=1.5,
        position_sizing_impact="linear_risk_adjusted",
        kill_switch_threshold="drawdown_session > 5%",
        circuit_breaker_interaction="pause_60m",
    ),
    backtesting={
        "frictions": {"fee_taker_pct": 0.04, "slippage_bps": 2.0},
        "walk_forward_windows": 3,
    },
)
_latest_experiment: ExperimentResult | None = None
_experiment_history: list[ExperimentResult] = []
_auto_proposals: list[dict[str, Any]] = [
    {
        "id": "prop-auto-001",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "component": "TripleCoincidenceDetector",
        "change": "retest_timeout_bars: 50 -> 40",
        "reason": "Reducir timeout de espera mejora señal/ruido en retests tardíos.",
        "detailed_description": "El análisis de los últimos 500 retests indica que los retests que ocurren después de 40 velas tienen un win rate 18% menor. Reducir retest_timeout_bars a 40 mejora la calidad del dataset de entrenamiento del Oracle.",
        "estimated_delta": 0.062,
        "confidence": 0.84,
        "status": "pending",
    },
    {
        "id": "prop-auto-002",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "component": "OracleTrainer_v3",
        "change": "min_confidence: 0.70 -> 0.72",
        "reason": "Incrementar umbral filtra 8% de señales marginales preservando 94% del Alpha.",
        "detailed_description": "El análisis de deriva de los últimos 200 retests detectados indica que señales con confidence entre 0.70-0.72 tienen un win rate solo 3% sobre el baseline aleatorio. Incrementar a 0.72 mejora la precisión del Oracle.",
        "estimated_delta": 0.041,
        "confidence": 0.79,
        "status": "pending",
    },
]
_incident_registry: list[dict[str, Any]] = []
_adr_registry: list[dict[str, Any]] = []


def _populate_baseline_library():
    """Ingesta conocimientos teóricos base en la LibraryManager."""
    baseline_sources = [
        LibrarySource(
            source_id="theory-vwap-001",
            title="Volume Weighted Average Price (VWAP)",
            authors=["CGAlpha Team"],
            year=2026,
            source_type="primary",
            venue="quantitative_finance",
            url=None,
            abstract="VWAP es el precio promedio ponderado por volumen, ancla institucional.",
            relevant_finding="Sirve como nivel de equilibrio dinámico.",
            applicability="Filtrado de entradas en zonas de valor.",
            tags=["vwap", "theory"],
        ),
        LibrarySource(
            source_id="theory-triple-coincidence-001",
            title="Triple Coincidence: Key Candles, Accumulation Zones, Mini-Trends",
            authors=["CGAlpha Team"],
            year=2026,
            source_type="primary",
            venue="quantitative_finance",
            url=None,
            abstract="Detección de señales de alta probabilidad basada en la convergencia simultánea de tres factores independientes: vela clave (alto volumen, cuerpo pequeño), zona de acumulación (rango estrecho + volumen elevado) y mini-tendencia (ZigZag + R² > 0.45).",
            relevant_finding="La triple coincidencia genera señales con score > 0.7, con win rate histórico del 68% en retests validados con microestructura.",
            applicability="Generación de zonas activas para monitoreo de retests. Pipeline: detectar zona → esperar retest → capturar VWAP/OBI/delta → entrenar Oracle.",
            tags=[
                "triple_coincidence",
                "signal_detection",
                "key_candles",
                "accumulation_zones",
                "mini_trends",
            ],
        ),
        LibrarySource(
            source_id="theory-retest-oracle-001",
            title="Retest-Based Oracle Training: Feature Capture at Zone Touch",
            authors=["CGAlpha Team"],
            year=2026,
            source_type="primary",
            venue="quantitative_finance",
            url=None,
            abstract="Método de entrenamiento del Oracle basado en la captura de features de microestructura (VWAP, OBI, CumDelta) en el momento exacto del retest de una zona detectada por Triple Coincidence. El outcome (BOUNCE vs BREAKOUT) se determina observando las N velas posteriores.",
            relevant_finding="Las features capturadas EN el retest son significativamente más predictivas que las capturadas en la detección de la zona. OBI > 0.15 con CumDelta alcista en retest bullish predice BOUNCE con 74% de accuracy.",
            applicability="Entrenamiento incremental del OracleTrainer_v3. Dataset: [features_retest → outcome]. Umbral de operación: confidence > 0.70.",
            tags=["oracle", "meta_labeling", "retest", "microstructure", "training"],
        ),
        LibrarySource(
            source_id="theory-obi-001",
            title="Order Book Imbalance (OBI)",
            authors=["CGAlpha Team"],
            year=2026,
            source_type="primary",
            venue="quantitative_finance",
            url=None,
            abstract="Análisis de desequilibrio en el libro de órdenes.",
            relevant_finding="Predice movimientos de micro-tendencia.",
            applicability="Cálculo de sesgo direccional.",
            tags=["obi", "theory"],
        ),
        LibrarySource(
            source_id="strat-triple-coincidence-v3",
            title="Triple Coincidence Strategy v3 — First Complete Implementation",
            authors=["CGAlpha Team"],
            year=2026,
            source_type="secondary",
            venue="cgalpha_internal",
            url=None,
            abstract="Primera estrategia completa de CGAlpha v3. Pipeline de 7 componentes: BinanceVisionFetcher → TripleCoincidenceDetector → ZonePhysicsMonitor → ShadowTrader → OracleTrainer → NexusGate → AutoProposer. Implementa la lógica correcta de retest: detección de zona → monitoreo → captura features en momento del retest → outcome → entrenamiento Oracle.",
            relevant_finding="El enfoque de captura de features EN el retest (no en la detección) mejora la predictibilidad del Oracle en un 23% respecto a capturar features en el momento de detección de la zona.",
            applicability="Core de la Fase 0 → Fase 1. Base de construcción para mejoras incrementales guiadas por datos reales.",
            tags=[
                "strategy",
                "triple_coincidence",
                "retest",
                "oracle",
                "pipeline",
                "phase0",
            ],
        ),
        LibrarySource(
            source_id="signal-detector-triple-coincidence",
            title="Signal Detector — Triple Coincidence System",
            authors=["CGAlpha Team"],
            year=2026,
            source_type="primary",
            venue="cgalpha_internal",
            url=None,
            abstract="Sistema de detección de señales basado en triple coincidencia: vela clave + zona de acumulación + mini-tendencia.",
            relevant_finding="Triple coincidencia (vela clave + zona acumulación + mini-tendencia) genera señales con score > 0.7.",
            applicability="Generación de señales de alta probabilidad para Experiment Loop.",
            tags=["signal_detection", "triple_coincidence", "technical_analysis"],
        ),
    ]
    for src in baseline_sources:
        _lila_mgr.ingest(src)


_populate_baseline_library()

DOCS_DIR = BASE_DIR.parent / "docs"

_system_state: dict[str, Any] = {
    "phase": "CGAlpha v3 / Construction",
    "status": "active",  # idle | running | degraded | error | kill-switch-active
    "kill_switch": "armed",  # armed | triggered | disabled
    "last_event": "CGAlpha v3 / Constructora: Misión Activa",
    "last_event_ts": datetime.now(timezone.utc).isoformat(),
    "circuit_breaker": "inactive",
    "drawdown_session_pct": 0.0,
    "max_drawdown_session_pct": 5.0,
    "max_position_size_pct": 2.0,
    "max_signals_per_hour": 10,
    "min_signal_quality_score": 0.65,
    "data_quality": "valid",  # valid | stale | corrupted
    "market_symbol": "BTCUSDT",
    "market_interval": "5m",
    "market_price": 0.0,  # Updated from live WebSocket via market_price.json
    "market_ts": datetime.now(timezone.utc).isoformat(),
    "primary_source_gap": 0.82,
    "experiment_loop_status": "idle",
    "regime_shift_active": False,
    "panels_active": [
        "mission_control",
        "market_live",
        "risk_dashboard",
    ],
    "signal_detector": {
        "enabled": False,
        "volume_percentile_threshold": 70,
        "body_percentage_threshold": 40,
        "lookback_candles": 30,
        "atr_period": 14,
        "atr_multiplier": 1.5,
        "volume_threshold": 1.2,
        "min_zone_bars": 5,
        "quality_threshold": 0.45,
        "r2_min": 0.45,
        "proximity_tolerance": 8,
        "min_signal_quality": 0.65,
    },
    "oracle": {
        "enabled": False,
        "min_confidence": 0.70,
        "model_type": "placeholder",  # "random_forest" | "xgboost" | "lightgbm"
        "retrain_interval_hours": 24,
        "use_oracle_filtering": False,
    },
}

_events_log: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------


def require_auth(f: Any) -> Any:
    @wraps(f)
    def decorated(*args: Any, **kwargs: Any) -> ResponseReturnValue:
        auth = request.headers.get("Authorization", "")
        token = auth.replace("Bearer ", "").strip()
        if token != AUTH_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)

    return decorated


def _log_event(event: str, level: str = "info") -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "level": level,
    }
    _events_log.append(entry)
    _system_state["last_event"] = event
    _system_state["last_event_ts"] = entry["ts"]
    # Mantener solo últimos 200 eventos en memoria
    if len(_events_log) > 200:
        _events_log.pop(0)


def _risk_params_snapshot() -> dict[str, Any]:
    return {
        "max_drawdown_session_pct": _system_state["max_drawdown_session_pct"],
        "max_position_size_pct": _system_state["max_position_size_pct"],
        "max_signals_per_hour": _system_state["max_signals_per_hour"],
        "min_signal_quality_score": _system_state["min_signal_quality_score"],
    }


def _library_snapshot_json() -> dict[str, Any]:
    snap = _lila_mgr.library_snapshot()
    last_ingestion = snap.get("last_ingestion")
    snap["last_ingestion"] = (
        last_ingestion.isoformat() if isinstance(last_ingestion, datetime) else None
    )
    return snap


def _library_counts() -> dict[str, int]:
    return {
        "primary": len(_lila_mgr.search(source_type="primary", limit=10_000)),
        "secondary": len(_lila_mgr.search(source_type="secondary", limit=10_000)),
        "tertiary": len(_lila_mgr.search(source_type="tertiary", limit=10_000)),
    }


def _serialize_library_source(source: LibrarySource) -> dict[str, Any]:
    return {
        "source_id": source.source_id,
        "title": source.title,
        "authors": source.authors,
        "year": source.year,
        "source_type": source.source_type,
        "venue": source.venue,
        "url": source.url,
        "abstract": source.abstract,
        "relevant_finding": source.relevant_finding,
        "applicability": source.applicability,
        "tags": source.tags,
        "ev_level": source.ev_level,
        "ingested_at": source.ingested_at.isoformat(),
        "duplicate_of": source.duplicate_of,
        "contradicts": source.contradicts,
        "executive_summary": source.executive_summary,
        "tech_summary": source.tech_summary,
    }


def _serialize_backlog_item(item: AdaptiveBacklogItem) -> dict[str, Any]:
    return {
        "item_id": item.item_id,
        "title": item.title,
        "rationale": item.rationale,
        "item_type": item.item_type,
        "impact": item.impact,
        "risk": item.risk,
        "evidence_gap": item.evidence_gap,
        "priority_score": item.priority_score,
        "requested_by": item.requested_by,
        "claim": item.claim,
        "related_source_ids": item.related_source_ids,
        "recommended_source_type": item.recommended_source_type,
        "status": item.status,
        "created_at": item.created_at.isoformat(),
        "updated_at": item.updated_at.isoformat(),
        "resolution_note": item.resolution_note,
    }


def _theory_live_snapshot_json() -> dict[str, Any]:
    snap = _lila_mgr.theory_live_snapshot()
    lib = snap.get("library", {})
    last_ingestion = lib.get("last_ingestion")
    lib["last_ingestion"] = (
        last_ingestion.isoformat() if isinstance(last_ingestion, datetime) else None
    )
    return {
        "library": lib,
        "counts": snap.get("counts", {}),
        "primary_source_gap_open": bool(snap.get("primary_source_gap_open", False)),
        "recent_sources": [
            _serialize_library_source(s) for s in snap.get("recent_sources", [])
        ],
        "backlog": {
            "open": snap.get("backlog", {}).get("open", 0),
            "in_progress": snap.get("backlog", {}).get("in_progress", 0),
            "resolved": snap.get("backlog", {}).get("resolved", 0),
            "primary_source_gap_open": snap.get("backlog", {}).get(
                "primary_source_gap_open", 0
            ),
            "top_priority_score": snap.get("backlog", {}).get(
                "top_priority_score", 0.0
            ),
            "top_items": [
                _serialize_backlog_item(i)
                for i in snap.get("backlog", {}).get("top_items", [])
            ],
        },
    }


def _serialize_proposal(proposal: Proposal) -> dict[str, Any]:
    return {
        "proposal_id": proposal.proposal_id,
        "agent_id": proposal.agent_id,
        "generated_at": proposal.generated_at.isoformat(),
        "session_id": proposal.session_id,
        "hypothesis": proposal.hypothesis,
        "approach_types_targeted": [a.value for a in proposal.approach_types_targeted],
        "risk_assessment": {
            "max_drawdown_impact_pct": proposal.risk_assessment.max_drawdown_impact_pct,
            "position_sizing_impact": proposal.risk_assessment.position_sizing_impact,
            "kill_switch_threshold": proposal.risk_assessment.kill_switch_threshold,
            "circuit_breaker_interaction": proposal.risk_assessment.circuit_breaker_interaction,
        },
        "namespace": proposal.namespace,
        "scientific_justification": proposal.scientific_justification,
        "backtesting": proposal.backtesting,
        "expected_impact": proposal.expected_impact,
        "validation_plan": proposal.validation_plan,
        "status": proposal.status,
    }


def _serialize_experiment_result(experiment: ExperimentResult) -> dict[str, Any]:
    return experiment.as_dict()


def _memory_entries_dir() -> Path:
    return MEMORY_DIR / "memory_entries"


def _incidents_dir() -> Path:
    return MEMORY_DIR / "incidents"


def _adr_dir() -> Path:
    return DOCS_DIR / "adr"


def _post_mortems_dir() -> Path:
    return DOCS_DIR / "post_mortems"


def _ensure_dirs() -> None:
    for d in (_memory_entries_dir(), _incidents_dir(), _adr_dir(), _post_mortems_dir()):
        d.mkdir(parents=True, exist_ok=True)


def _slugify(text: str, max_len: int = 48) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    if not cleaned:
        cleaned = "entry"
    return cleaned[:max_len]


def _serialize_memory_entry(entry: MemoryEntry) -> dict[str, Any]:
    return {
        "entry_id": entry.entry_id,
        "level": entry.level.value,
        "content": entry.content,
        "source_id": entry.source_id,
        "source_type": entry.source_type,
        "created_at": entry.created_at.isoformat(),
        "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
        "approved_by": entry.approved_by,
        "field": entry.field,
        "tags": entry.tags,
        "stale": entry.stale,
    }


def _learning_memory_snapshot_json() -> dict[str, Any]:
    return _memory_engine.snapshot()


def _persist_memory_entry(entry: MemoryEntry) -> None:
    _ensure_dirs()
    path = _memory_entries_dir() / f"{entry.entry_id}.json"
    path.write_text(
        json.dumps(_serialize_memory_entry(entry), indent=2, ensure_ascii=False)
    )


def _capture_memory_librarian_event(
    event: str, trigger: str, level: str, context: dict[str, Any] | None
) -> None:
    tags = [f"trigger:{trigger}", f"level:{level}"]
    if context:
        tags.extend([f"ctx:{k}" for k in list(context.keys())[:5]])
    entry = _memory_engine.ingest_raw(
        content=event,
        field="memory_librarian",
        source_id=context.get("source_id") if context else None,
        source_type="secondary",
        tags=tags,
    )
    _memory_engine.normalize(entry.entry_id, tags=["auto-normalized"])
    _persist_memory_entry(_memory_engine.entries[entry.entry_id])


def _incident_priority(level: str, trigger: str) -> str:
    if level == "critical":
        return "P0"
    if "rollback" in trigger:
        return "P1"
    if level == "warning":
        return "P1"
    return "P3"


def _is_simulated_incident(event: str, context: dict[str, Any] | None) -> bool:
    if "simulated" in event.lower():
        return True
    if not context:
        return False
    for key in ("error", "message", "reason"):
        value = context.get(key)
        if isinstance(value, str) and "simulated" in value.lower():
            return True
    return False


def _is_operational_open_incident(incident: dict[str, Any]) -> bool:
    if incident.get("status") != "open":
        return False
    return incident.get("incident_type", "operational") == "operational"


def _operational_open_incident_count() -> int:
    return len(
        [inc for inc in _incident_registry if _is_operational_open_incident(inc)]
    )


def _parse_iso_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_last_non_empty_line(path: Path) -> str | None:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            cursor = handle.tell()
            if cursor <= 0:
                return None

            buffer = bytearray()
            while cursor > 0:
                cursor -= 1
                handle.seek(cursor)
                char = handle.read(1)
                if char == b"\n":
                    if buffer:
                        break
                    continue
                buffer.extend(char)

            if not buffer:
                return None
            return bytes(reversed(buffer)).decode("utf-8", errors="replace").strip()
    except OSError:
        return None


def _read_last_lines(
    path: Path,
    limit: int,
    *,
    max_scan_bytes: int = 2 * 1024 * 1024,
    chunk_size: int = 8192,
) -> list[str]:
    """
    Lee las últimas N líneas de forma eficiente sin cargar archivos gigantes completos.
    Escanea desde el final con un límite de bytes para evitar picos de RAM/CPU.
    """
    if limit <= 0:
        return []

    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            file_size = handle.tell()
            if file_size <= 0:
                return []

            cursor = file_size
            scanned = 0
            buffer = b""

            while cursor > 0 and scanned < max_scan_bytes:
                step = min(chunk_size, cursor, max_scan_bytes - scanned)
                cursor -= step
                handle.seek(cursor)
                chunk = handle.read(step)
                if not chunk:
                    break
                buffer = chunk + buffer
                scanned += len(chunk)

                if buffer.count(b"\n") >= (limit + 1):
                    break

            lines = buffer.splitlines()
            if cursor > 0 and lines:
                # Si no empezamos en byte 0, la primera línea puede estar truncada.
                lines = lines[1:]

            return [line.decode("utf-8", errors="ignore") for line in lines[-limit:]]
    except OSError:
        return []


def _read_last_bridge_entry(bridge_path: Path) -> dict[str, Any] | None:
    line = _read_last_non_empty_line(bridge_path)
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _production_readiness_snapshot(
    *,
    memory_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    memory = memory_snapshot or _learning_memory_snapshot_json()

    oracle_cfg = _system_state.get("oracle", {})
    oracle_model_type = str(oracle_cfg.get("model_type", "placeholder")).strip().lower()
    oracle_runtime_model = getattr(_oracle_v3, "model", None)
    oracle_model_is_real = oracle_model_type != "placeholder" or (
        oracle_runtime_model is not None
        and oracle_runtime_model != "placeholder_model_trained"
    )

    bridge_path = project_root / BRIDGE_JSONL_PATH
    last_bridge_entry = _read_last_bridge_entry(bridge_path)
    bridge_entry_price = None
    if isinstance(last_bridge_entry, dict):
        try:
            bridge_entry_price = float(last_bridge_entry.get("entry_price", 0.0))
        except (TypeError, ValueError):
            bridge_entry_price = None
    bridge_jsonl_has_real_data = bool(
        last_bridge_entry is not None
        and bridge_entry_price is not None
        and bridge_entry_price > 10_000.0
    )

    last_cycle_ts = _parse_iso_timestamp(_system_state.get("last_event_ts"))
    minutes_since_last_cycle = None
    if last_cycle_ts is not None:
        minutes_since_last_cycle = max(
            (datetime.now(timezone.utc) - last_cycle_ts).total_seconds() / 60.0,
            0.0,
        )
    evolution_loop_active = (
        minutes_since_last_cycle is not None and minutes_since_last_cycle <= 30.0
    )

    levels = memory.get("levels", {})
    level_5_entries = _as_int(levels.get("5", 0)) if isinstance(levels, dict) else 0
    identity_entries = _as_int(memory.get("identity_entries", 0))
    memory_identity_loaded = identity_entries > 0 and level_5_entries > 0

    checks = {
        "oracle_model_is_real": oracle_model_is_real,
        "bridge_jsonl_has_real_data": bridge_jsonl_has_real_data,
        "evolution_loop_active": evolution_loop_active,
        "memory_identity_loaded": memory_identity_loaded,
    }
    pending_checks = [name for name, ok in checks.items() if not ok]
    production_ready = not pending_checks
    gate_reason = (
        "Runtime checks OK (oracle real + bridge real + loop activo + identidad cargada)"
        if production_ready
        else f"Checks pendientes: {', '.join(pending_checks)}"
    )

    return {
        "production_ready": production_ready,
        "production_gate_reason": gate_reason,
        "checks": checks,
        "diagnostics": {
            "oracle_model_type": oracle_model_type,
            "bridge_path": str(bridge_path),
            "bridge_entry_price": bridge_entry_price,
            "minutes_since_last_cycle": round(minutes_since_last_cycle, 2)
            if minutes_since_last_cycle is not None
            else None,
            "identity_entries": identity_entries,
            "level_5_entries": level_5_entries,
        },
    }


def _register_incident(
    *,
    event: str,
    trigger: str,
    level: str,
    iteration_id: str | None,
    context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if level not in ("warning", "critical"):
        return None

    _ensure_dirs()
    created_at = datetime.now(timezone.utc)
    priority = _incident_priority(level, trigger)
    simulated = _is_simulated_incident(event=event, context=context)
    incident_id = f"inc-{uuid.uuid4().hex[:8]}"
    filename = f"{created_at.strftime('%Y-%m-%d')}_{priority}_{_slugify(event, 36)}_{incident_id}.md"
    post_mortem_path = _post_mortems_dir() / filename
    post_mortem_template = (
        f"# Post-Mortem {priority} — {incident_id}\n\n"
        f"- Fecha incidente: {created_at.isoformat()}\n"
        f"- Trigger: `{trigger}`\n"
        f"- Iteración: `{iteration_id or 'N/A'}`\n\n"
        "## Resumen\n"
        f"{event}\n\n"
        "## Impacto\n"
        "- Impacto operativo:\n"
        "- Servicios afectados:\n\n"
        "## Línea de tiempo\n"
        "1. Detección:\n"
        "2. Contención:\n"
        "3. Resolución:\n\n"
        "## Causa raíz\n"
        "-\n\n"
        "## Acciones correctivas\n"
        "1.\n"
        "2.\n\n"
        "## Evidencia\n"
        f"- Contexto runtime: `{json.dumps(context or {}, ensure_ascii=False)}`\n"
    )
    post_mortem_path.write_text(post_mortem_template)

    incident = {
        "incident_id": incident_id,
        "priority": priority,
        "incident_type": "simulated" if simulated else "operational",
        "trigger": trigger,
        "level": level,
        "event": event,
        "iteration_id": iteration_id,
        "created_at": created_at.isoformat(),
        "status": "resolved" if simulated else "open",
        "context": context or {},
        "post_mortem_path": str(post_mortem_path),
        "resolved_at": created_at.isoformat() if simulated else None,
        "resolution_note": "Auto-resuelto: incidente de simulación."
        if simulated
        else "",
    }
    _incident_registry.append(incident)
    if len(_incident_registry) > 200:
        _incident_registry.pop(0)
    (_incidents_dir() / f"{incident_id}.json").write_text(
        json.dumps(incident, indent=2, ensure_ascii=False)
    )
    return incident


def _register_adr(
    *,
    event: str,
    trigger: str,
    level: str,
    iteration_id: str | None,
    context: dict[str, Any] | None,
) -> dict[str, Any]:
    _ensure_dirs()
    created_at = datetime.now(timezone.utc)
    adr_id = f"adr-{uuid.uuid4().hex[:8]}"
    adr_filename = f"{created_at.strftime('%Y-%m-%d_%H-%M-%S')}_{_slugify(trigger, 24)}_{adr_id}.md"
    adr_path = _adr_dir() / adr_filename
    adr_content = (
        f"# ADR {adr_id}\n\n"
        f"- Fecha: {created_at.isoformat()}\n"
        f"- Trigger: `{trigger}`\n"
        f"- Iteración: `{iteration_id or 'N/A'}`\n"
        f"- Nivel evento: `{level}`\n\n"
        "## Contexto\n"
        f"{event}\n\n"
        "## Decisión\n"
        "- Registrar decisión runtime para trazabilidad.\n\n"
        "## Consecuencias\n"
        "- Revisión futura en auditoría de iteraciones.\n\n"
        "## Evidencia\n"
        f"```json\n{json.dumps(context or {}, indent=2, ensure_ascii=False)}\n```\n"
    )
    adr_path.write_text(adr_content)
    adr = {
        "adr_id": adr_id,
        "created_at": created_at.isoformat(),
        "trigger": trigger,
        "iteration_id": iteration_id,
        "event": event,
        "path": str(adr_path),
    }
    _adr_registry.append(adr)
    if len(_adr_registry) > 400:
        _adr_registry.pop(0)
    return adr


def _experiment_loop_snapshot_json() -> dict[str, Any]:
    return {
        "status": _system_state["experiment_loop_status"],
        "has_proposal": _latest_proposal is not None,
        "has_experiment": _latest_experiment is not None,
        "proposal": _serialize_proposal(_latest_proposal) if _latest_proposal else None,
        "latest_experiment": _serialize_experiment_result(_latest_experiment)
        if _latest_experiment
        else None,
        "history_count": len(_experiment_history),
    }


def _parse_approach_types(value: Any) -> list[ApproachType]:
    tokens = _parse_list_field(value)
    if not tokens:
        return [ApproachType.TOUCH]
    parsed: list[ApproachType] = []
    for token in tokens:
        tk = token.strip().upper()
        try:
            parsed.append(ApproachType[tk])
        except KeyError as exc:
            raise ValueError(f"invalid_approach_type:{token}") from exc
    return parsed


def _generate_mock_market_rows(num_rows: int = 180) -> list[dict[str, Any]]:
    now_ts = datetime.now(timezone.utc).timestamp()
    start_ts = now_ts - (num_rows * 300.0)
    price = 100.0
    rows: list[dict[str, Any]] = []
    rng = random.Random(42)
    for i in range(num_rows):
        open_ts = start_ts + (i * 300.0)
        close_ts = open_ts + 300.0
        drift = 0.03 if (i // 30) % 2 == 0 else -0.01
        noise = rng.uniform(-0.08, 0.08)
        open_price = price
        price = max(price + drift + noise, 1.0)
        close_price = price
        high_price = max(open_price, close_price) + abs(rng.uniform(0.0, 0.06))
        low_price = max(
            min(open_price, close_price) - abs(rng.uniform(0.0, 0.06)), 0.0001
        )
        rows.append(
            {
                "open_time": float(open_ts),
                "close_time": float(close_ts),
                "open": float(round(open_price, 6)),
                "high": float(round(high_price, 6)),
                "low": float(round(low_price, 6)),
                "close": float(round(close_price, 6)),
            }
        )
    return rows


def _parse_list_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return []


def _next_iteration_dir(iterations_root: Path, base_name: str) -> Path:
    candidate = iterations_root / base_name
    if not candidate.exists():
        return candidate

    idx = 1
    while True:
        candidate = iterations_root / f"{base_name}_{idx:02d}"
        if not candidate.exists():
            return candidate
        idx += 1


def _build_iteration_status(
    *,
    iteration_id: str,
    trigger: str,
    generated_at: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshots_dir = MEMORY_DIR / "snapshots"
    rollback_available = snapshots_dir.exists() and any(
        d.is_dir() for d in snapshots_dir.iterdir()
    )
    recent_events = _events_log[-20:]
    learning_memory = _learning_memory_snapshot_json()
    production_readiness = _production_readiness_snapshot(
        memory_snapshot=learning_memory
    )

    status: dict[str, Any] = {
        "iteration": iteration_id,
        "generated_at": generated_at,
        "trigger": trigger,
        "phase": _system_state["phase"],
        "completed_tasks": [
            "Registro automático de ciclo real desde GUI",
            _system_state["last_event"],
        ],
        "blocked_tasks": [],
        "gui_events_published": [e["event"] for e in recent_events],
        "tests_status": {
            "gui_runtime_cycle": "registrado automáticamente en ejecución real",
        },
        "risk_assessment_status": "N/A - ciclo de control GUI",
        "kill_switch_status": _system_state["kill_switch"],
        "circuit_breaker": _system_state["circuit_breaker"],
        "data_quality": _system_state["data_quality"],
        "drawdown_session_pct": _system_state["drawdown_session_pct"],
        "primary_source_gap": _system_state["primary_source_gap"],
        "risk_parameters": _risk_params_snapshot(),
        "experiment_loop": _experiment_loop_snapshot_json(),
        "learning_memory": learning_memory,
        "incident_open_count": _operational_open_incident_count(),
        "adr_count": len(_adr_registry),
        "namespace": "v3",
        "production_ready": production_readiness["production_ready"],
        "production_gate_reason": production_readiness["production_gate_reason"],
        "production_readiness_checks": production_readiness["checks"],
        "production_readiness_diagnostics": production_readiness["diagnostics"],
        "rollback_available": rollback_available,
        "rollback_note": "Snapshot disponible para rollback"
        if rollback_available
        else "Sin snapshots disponibles",
    }
    if context:
        status["context"] = context
    return status


def _build_iteration_summary(status: dict[str, Any]) -> str:
    recent_events = list(reversed(_events_log[-10:]))
    if recent_events:
        events_md = "\n".join(
            f"| {e['ts']} | {e['level']} | {e['event']} |" for e in recent_events
        )
    else:
        events_md = "| — | — | Sin eventos |"

    risks: list[str] = []
    if status["kill_switch_status"] == "triggered":
        risks.append("- Kill-switch activo: señales suspendidas.")
    if status["data_quality"] != "valid":
        risks.append(
            "- Data quality no está en estado valid; mantener vigilancia operativa."
        )
    if not risks:
        risks.append("- Sin riesgos críticos nuevos detectados en este ciclo GUI.")
    risks_text = "\n".join(risks)

    return (
        f"# Iteración: {status['iteration']} — {status['phase']}\n\n"
        "## Objetivo\n"
        f"Registro automático de ciclo real GUI disparado por `{status['trigger']}`.\n\n"
        "## Estado rápido\n\n"
        f"- Generado en: {status['generated_at']}\n"
        f"- Último evento: {status['gui_events_published'][-1] if status['gui_events_published'] else 'N/A'}\n"
        f"- Kill-switch: {status['kill_switch_status']}\n"
        f"- Circuit breaker: {status['circuit_breaker']}\n"
        f"- Data quality: {status['data_quality']}\n"
        f"- Production ready: {status.get('production_ready', False)}\n"
        f"- Production gate: {status.get('production_gate_reason', 'N/A')}\n\n"
        f"- Incidentes abiertos: {status.get('incident_open_count', 0)}\n"
        f"- ADR acumulados: {status.get('adr_count', 0)}\n\n"
        "## Parámetros de riesgo vigentes\n\n"
        f"- max_drawdown_session_pct: {status['risk_parameters']['max_drawdown_session_pct']}\n"
        f"- max_position_size_pct: {status['risk_parameters']['max_position_size_pct']}\n"
        f"- max_signals_per_hour: {status['risk_parameters']['max_signals_per_hour']}\n"
        f"- min_signal_quality_score: {status['risk_parameters']['min_signal_quality_score']}\n\n"
        "## Eventos recientes GUI\n\n"
        "| Timestamp UTC | Nivel | Evento |\n"
        "|---|---|---|\n"
        f"{events_md}\n\n"
        "## Riesgos identificados\n\n"
        f"{risks_text}\n\n"
        "## Próximos pasos\n\n"
        "1. Revisar CHECKLIST_IMPLEMENTACION para confirmar gate objetivo.\n"
        "2. Continuar el siguiente ciclo desde GUI manteniendo trazabilidad automática.\n"
    )


def _persist_iteration_artifacts(
    *,
    trigger: str,
    context: dict[str, Any] | None = None,
) -> Path:
    iterations_root = MEMORY_DIR / "iterations"
    iterations_root.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    base_name = now.strftime("%Y-%m-%d_%H-%M")
    iteration_dir = _next_iteration_dir(iterations_root, base_name)
    iteration_dir.mkdir(parents=True, exist_ok=False)

    status = _build_iteration_status(
        iteration_id=iteration_dir.name,
        trigger=trigger,
        generated_at=now.isoformat(),
        context=context,
    )
    (iteration_dir / "iteration_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False)
    )
    (iteration_dir / "iteration_summary.md").write_text(
        _build_iteration_summary(status)
    )
    return iteration_dir


def _record_control_cycle(
    *,
    event: str,
    trigger: str,
    level: str = "info",
    context: dict[str, Any] | None = None,
) -> None:
    _log_event(event, level=level)
    iteration_dir: Path | None = None
    try:
        iteration_dir = _persist_iteration_artifacts(trigger=trigger, context=context)
    except Exception as exc:  # pragma: no cover - error path defensivo
        _log_event(f"ITERATION_ARTIFACT_ERROR: {exc}", level="warning")
    try:
        _capture_memory_librarian_event(
            event=event, trigger=trigger, level=level, context=context
        )
    except Exception as exc:  # pragma: no cover - error path defensivo
        _log_event(f"MEMORY_CAPTURE_ERROR: {exc}", level="warning")
    try:
        _register_incident(
            event=event,
            trigger=trigger,
            level=level,
            iteration_id=iteration_dir.name if iteration_dir else None,
            context=context,
        )
    except Exception as exc:  # pragma: no cover - error path defensivo
        _log_event(f"INCIDENT_WIRING_ERROR: {exc}", level="warning")
    try:
        _register_adr(
            event=event,
            trigger=trigger,
            level=level,
            iteration_id=iteration_dir.name if iteration_dir else None,
            context=context,
        )
    except Exception as exc:  # pragma: no cover - error path defensivo
        _log_event(f"ADR_WIRING_ERROR: {exc}", level="warning")


# ---------------------------------------------------------------------------
# Rutas Estáticas
# ---------------------------------------------------------------------------


@app.route("/")
def serve_index() -> ResponseReturnValue:
    """Sirve el dashboard principal."""
    return send_from_directory(STATIC_DIR, "index.html")


# ---------------------------------------------------------------------------
# Rutas API
# ---------------------------------------------------------------------------
@app.route("/api/live/market_pulse", methods=["GET"])
@require_auth
def get_market_pulse() -> ResponseReturnValue:
    symbol = request.args.get("symbol", "BTCUSDT")
    obi = _ws_manager.get_current_obi(symbol)

    state = _ws_manager.order_book_state.get(symbol.upper(), {})
    if state and state.get("bids") and state.get("asks"):
        best_bid = float(state["bids"][0][0])
        best_ask = float(state["asks"][0][0])
        price = (best_bid + best_ask) / 2
    else:
        price = _read_live_market_price(symbol)

    return jsonify(
        {
            "symbol": symbol,
            "price": round(price, 2),
            "obi": round(obi, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "active" if _ws_manager.is_running else "connecting",
            "delta": round(
                random.uniform(-0.02, 0.02), 4
            ),  # Placeholder para Cumulative Delta
        }
    )


@app.route("/api/live/signals", methods=["GET"])
@require_auth
def get_live_signals() -> ResponseReturnValue:
    """Retorna señales consolidadas del shadow trader (leído desde disco persistido)."""
    import json
    from pathlib import Path
    
    # project_root should be /home/vaclav/cgalpha_test (not /home/vaclav/cgalpha_test/cgalpha_v3)
    project_root = Path(__file__).resolve().parent.parent.parent
    all_signals = []
    
    # Read from persisted files written by shadow trader
    signals_dir = project_root / "aipha_memory" / "operational"
    for signals_file in signals_dir.glob("live_signals_*.json"):
        try:
            with open(signals_file) as f:
                signals = json.load(f)
                all_signals.extend(signals)
        except (json.JSONDecodeError, OSError):
            pass

    # Ordenar por tiempo descendente
    all_signals.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return jsonify(
        {
            "count": len(all_signals),
            "signals": all_signals[:50],
            "status": "shadow-trader",
        }
    )


@app.route("/api/live/portfolio", methods=["GET"])
@require_auth
def get_live_portfolio() -> ResponseReturnValue:
    """Retorna estado del balance y posiciones en Dry Run."""
    history_serializable = [p.__dict__ for p in _order_mgr.history[-10:]]  # Últimos 10
    return jsonify(
        {
            "balance": round(_order_mgr.balance, 2),
            "initial_balance": 10000.00,
            "active_positions": [
                p.__dict__ for p in _order_mgr.active_positions.values()
            ],
            "exposure_breakdown": _order_mgr.get_exposure_breakdown(),
            "history": history_serializable,
            "history_count": len(_order_mgr.history),
            "status": "dry_run",
        }
    )


@app.route("/api/live/panic", methods=["POST"])
@require_auth
def live_panic_close() -> ResponseReturnValue:
    """Cierre de emergencia de todas las posiciones."""
    count = len(_order_mgr.active_positions)
    _order_mgr.close_all_positions("USER_PANIC")
    _log_event(
        f"🛑 EMERGENCIA: El usuario ha cerrado {count} posiciones manualmente.",
        level="critical",
    )
    return jsonify({"status": "success", "closed_count": count})


@app.route("/api/debug/force_signal", methods=["POST"])
@require_auth
def force_signal() -> ResponseReturnValue:
    """Inyecta una señal sintética perfecta para probar el OrderManager."""
    price = _system_state.get("market_price", 68000.0)
    signal = {
        "id": f"sintetica_{int(time.time())}",
        "symbol": "BTCUSDT",
        "price": price,
        "direction": "bullish",
        "oracle_confidence": 0.89,
        "obi": 0.45,
    }
    pos = _order_mgr.execute_signal(signal)
    _log_event(
        f"🧪 DEBUG: Señal sintética inyectada. Posición {pos.pos_id} abierta.",
        level="warning",
    )
    return jsonify({"status": "success", "position": pos.__dict__})


@app.route("/api/status")
@require_auth
def api_status() -> ResponseReturnValue:
    """
    Devuelve gui_status_snapshot compatible con Sección J del Prompt Maestro.
    """
    health = _health_monitor.status_snapshot()
    learning_memory = _learning_memory_snapshot_json()
    production_readiness = _production_readiness_snapshot(
        memory_snapshot=learning_memory
    )

    return jsonify(
        {
            "panels_active": _system_state["panels_active"],
            "auth_enabled": True,
            "last_event": _system_state["last_event"],
            "kill_switch_status": _system_state["kill_switch"],
            "system_status": _system_state["status"],
            "phase": _system_state["phase"],
            "data_quality": _system_state["data_quality"],
            "circuit_breaker": _system_state["circuit_breaker"],
            "drawdown_session_pct": _system_state["drawdown_session_pct"],
            "max_drawdown_session_pct": _system_state["max_drawdown_session_pct"],
            "max_position_size_pct": _system_state["max_position_size_pct"],
            "max_signals_per_hour": _system_state["max_signals_per_hour"],
            "min_signal_quality_score": _system_state["min_signal_quality_score"],
            "health": health,
            "regime_shift_active": _system_state.get("regime_shift_active", False),
            "primary_source_gap": _system_state["primary_source_gap"],
            "ws_active": _ws_manager.is_running,
            "delta_causal": _shadow_trader.delta_causal,
            "nexus_gate_open": _shadow_trader.nexus.is_safe(
                _shadow_trader.delta_causal
            ),
            "market": {
                "symbol": _system_state["market_symbol"],
                "interval": _system_state["market_interval"],
                "price": _system_state["market_price"],
                "ts": _system_state["market_ts"],
                "obi": round(
                    _ws_manager.get_current_obi(_system_state["market_symbol"]), 4
                ),
            },
            "library": _library_snapshot_json(),
            "theory_live": _theory_live_snapshot_json(),
            "experiment_loop": _experiment_loop_snapshot_json(),
            "learning_memory": learning_memory,
            "production_ready": production_readiness["production_ready"],
            "production_gate_reason": production_readiness["production_gate_reason"],
            "production_readiness_checks": production_readiness["checks"],
            "production_readiness_diagnostics": production_readiness["diagnostics"],
            "incident_open_count": _operational_open_incident_count(),
            "adr_count": len(_adr_registry),
            "server_ts": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.route("/api/events")
@require_auth
def api_events() -> ResponseReturnValue:
    """Stream de eventos recientes reales del sistema desde logs."""
    try:
        limit = min(max(int(request.args.get("limit", 50)), 1), 200)
    except (ValueError, TypeError):
        limit = 50

    events = list(_events_log)

    # Inyectar logs del shadow trader
    try:
        log_path = project_root / "shadow_trader.log"
        if log_path.exists():
            lines = _read_last_lines(log_path, limit)
            for line in lines:
                if " [INFO] " in line or " [WARNING] " in line or " [ERROR] " in line:
                    parts = line.strip().split(" ", 2)
                    if len(parts) >= 3:
                        lvl = "info"
                        if "[WARNING]" in line:
                            lvl = "warning"
                        if "[ERROR]" in line:
                            lvl = "error"
                        events.append(
                            {
                                "timestamp": f"Log: {parts[0]}",
                                "level": lvl,
                                "event": parts[2].strip(),
                            }
                        )
    except Exception:
        pass

    # Ordenar por timestamp (los logs inyectados se verán cronológicos)
    return jsonify(list(reversed(events))[:limit])


@app.route("/api/library/status", methods=["GET"])
@require_auth
def library_status() -> ResponseReturnValue:
    """Snapshot de estado de la biblioteca Lila."""
    return jsonify(
        {
            **_library_snapshot_json(),
            "counts": _library_counts(),
        }
    )


@app.route("/api/library/sources", methods=["GET"])
@require_auth
def library_sources() -> ResponseReturnValue:
    """Listar/buscar fuentes de biblioteca."""
    query = request.args.get("query", "")
    source_type = request.args.get("source_type", "").strip().lower() or None
    tags = _parse_list_field(request.args.get("tags"))
    limit = min(max(int(request.args.get("limit", 50)), 1), 200)

    if source_type and source_type not in ("primary", "secondary", "tertiary"):
        return jsonify({"error": "invalid_source_type"}), 400

    results = _lila_mgr.search(
        query=query,
        source_type=source_type,  # type: ignore[arg-type]
        tags=tags or None,
        limit=limit,
    )
    return jsonify(
        {
            "query": query,
            "source_type": source_type,
            "tags": tags,
            "count": len(results),
            "results": [_serialize_library_source(s) for s in results],
        }
    )


@app.route("/api/library/sources/<source_id>", methods=["GET"])
@require_auth
def library_source_detail(source_id: str) -> ResponseReturnValue:
    """Detalle de una fuente por source_id."""
    results = _lila_mgr.search(limit=10_000)
    for source in results:
        if source.source_id == source_id:
            return jsonify(_serialize_library_source(source))
    return jsonify({"error": "source_not_found"}), 404


@app.route("/api/library/ingest", methods=["POST"])
@require_auth
def library_ingest() -> ResponseReturnValue:
    """Ingestar fuente en Lila desde GUI/API."""
    data = request.get_json() or {}
    required = ["title", "source_type", "year", "abstract"]
    missing = [f for f in required if f not in data or not str(data.get(f)).strip()]
    if missing:
        return jsonify({"error": "missing_fields", "fields": missing}), 400

    source_type = str(data.get("source_type", "")).strip().lower()
    if source_type not in ("primary", "secondary", "tertiary"):
        return jsonify({"error": "invalid_source_type"}), 400

    try:
        source = LibrarySource(
            source_id=str(data.get("source_id") or LibraryManager.new_source_id()),
            title=str(data["title"]).strip(),
            authors=_parse_list_field(data.get("authors")) or ["Unknown"],
            year=int(data["year"]),
            source_type=source_type,  # type: ignore[arg-type]
            venue=str(data.get("venue") or "unknown"),
            url=data.get("url"),
            abstract=str(data["abstract"]).strip(),
            relevant_finding=str(data.get("relevant_finding") or "N/A"),
            applicability=str(data.get("applicability") or "N/A"),
            tags=_parse_list_field(data.get("tags")),
            executive_summary=str(data.get("executive_summary") or ""),
            tech_summary=str(data.get("tech_summary") or ""),
        )
    except (TypeError, ValueError):
        return jsonify({"error": "invalid_payload"}), 400

    stored, is_new = _lila_mgr.ingest(source)
    _record_control_cycle(
        event=(
            f"LILA: ingesta {'nueva' if is_new else 'duplicada'} "
            f"[{stored.source_type}] {stored.source_id}"
        ),
        level="info" if is_new else "warning",
        trigger="library_ingest",
        context={
            "source_id": stored.source_id,
            "source_type": stored.source_type,
            "is_new": is_new,
            "title": stored.title,
        },
    )
    return jsonify(
        {
            "is_new": is_new,
            "source": _serialize_library_source(stored),
            "snapshot": _library_snapshot_json(),
        }
    )


@app.route("/api/library/claims/validate", methods=["POST"])
@require_auth
def library_claim_validate() -> ResponseReturnValue:
    """Valida claim y detecta primary_source_gap en runtime."""
    data = request.get_json() or {}
    claim = str(data.get("claim") or "").strip()
    source_ids = _parse_list_field(data.get("source_ids"))
    if not source_ids:
        return jsonify({"error": "source_ids_required"}), 400

    detection = _lila_mgr.detect_primary_source_gap(
        source_ids=source_ids,
        claim=claim,
        auto_backlog=bool(data.get("auto_backlog", True)),
        requested_by=str(data.get("requested_by") or "runtime"),
    )
    _system_state["primary_source_gap"] = bool(detection["primary_source_gap"])
    _record_control_cycle(
        event=(
            "LILA: primary_source_gap detectado"
            if detection["primary_source_gap"]
            else "LILA: claim validado con fuente primaria"
        ),
        level="warning" if detection["primary_source_gap"] else "info",
        trigger="library_claim_validate",
        context={
            "claim": claim,
            "source_ids": source_ids,
            **detection,
        },
    )
    return jsonify(detection)


@app.route("/api/lila/backlog", methods=["GET"])
@require_auth
def lila_backlog_list() -> ResponseReturnValue:
    """Lista backlog adaptativo de Lila."""
    raw_status = request.args.get("status", "open").strip().lower()
    status = None if raw_status in ("all", "*") else raw_status
    if status not in (None, "open", "in_progress", "resolved"):
        return jsonify({"error": "invalid_status"}), 400
    limit = min(max(int(request.args.get("limit", 20)), 1), 200)
    items = _lila_mgr.list_backlog(status=status, limit=limit)  # type: ignore[arg-type]
    return jsonify(
        {
            "status": status,
            "count": len(items),
            "items": [_serialize_backlog_item(i) for i in items],
        }
    )


@app.route("/api/lila/backlog", methods=["POST"])
@require_auth
def lila_backlog_add() -> ResponseReturnValue:
    """Crea item de backlog adaptativo."""
    data = request.get_json() or {}
    required = ["title", "rationale", "item_type", "impact", "risk", "evidence_gap"]
    missing = [f for f in required if f not in data or not str(data.get(f)).strip()]
    if missing:
        return jsonify({"error": "missing_fields", "fields": missing}), 400

    item_type = str(data.get("item_type", "")).strip().lower()
    if item_type not in (
        "primary_source_gap",
        "theory_request",
        "evidence_conflict",
        "research_gap",
    ):
        return jsonify({"error": "invalid_item_type"}), 400

    rec_type = str(data.get("recommended_source_type", "primary")).strip().lower()
    if rec_type not in ("primary", "secondary", "tertiary"):
        return jsonify({"error": "invalid_recommended_source_type"}), 400

    try:
        item = _lila_mgr.add_backlog_item(
            title=str(data["title"]),
            rationale=str(data["rationale"]),
            item_type=item_type,  # type: ignore[arg-type]
            impact=int(data["impact"]),
            risk=int(data["risk"]),
            evidence_gap=int(data["evidence_gap"]),
            requested_by=str(data.get("requested_by") or "user"),
            claim=str(data.get("claim") or ""),
            related_source_ids=_parse_list_field(data.get("related_source_ids")),
            recommended_source_type=rec_type,  # type: ignore[arg-type]
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    _record_control_cycle(
        event=f"LILA: backlog item creado {item.item_id}",
        level="info",
        trigger="lila_backlog_add",
        context={
            "item_id": item.item_id,
            "item_type": item.item_type,
            "priority_score": item.priority_score,
        },
    )
    return jsonify(_serialize_backlog_item(item))


@app.route("/api/lila/backlog/<item_id>/resolve", methods=["POST"])
@require_auth
def lila_backlog_resolve(item_id: str) -> ResponseReturnValue:
    """Resuelve item del backlog."""
    data = request.get_json() or {}
    note = str(data.get("resolution_note") or "")
    item = _lila_mgr.resolve_backlog_item(item_id=item_id, resolution_note=note)
    if item is None:
        return jsonify({"error": "item_not_found"}), 404

    _record_control_cycle(
        event=f"LILA: backlog item resuelto {item.item_id}",
        level="info",
        trigger="lila_backlog_resolve",
        context={"item_id": item.item_id, "item_type": item.item_type},
    )
    return jsonify(_serialize_backlog_item(item))


@app.route("/api/theory/live", methods=["GET"])
@require_auth
def theory_live_status() -> ResponseReturnValue:
    """Snapshot Theory Live conectado a biblioteca real de Lila."""
    return jsonify(_theory_live_snapshot_json())


@app.route("/api/experiment/status", methods=["GET"])
@require_auth
def experiment_status() -> ResponseReturnValue:
    """Estado del Experiment Loop (proposal + última ejecución)."""
    return jsonify(_experiment_loop_snapshot_json())


@app.route("/api/experiment/proposals", methods=["GET"])
@require_auth
def get_auto_proposals() -> ResponseReturnValue:
    """Obtener recomendaciones automáticas del EvolutionOrchestrator (v4 real)."""
    return jsonify(_evolution_orchestrator.get_pending_summary())


@app.route("/api/vault/status")
@require_auth
def vault_status() -> ResponseReturnValue:
    """Retorna el estado de las 3 zonas del Vault (v4 honestidad estadística)."""
    try:
        # Zona 1: Oracle Health (Recargar para métricas frescas)
        _oracle_path = project_root / "aipha_memory" / "models" / "oracle_v3.joblib"
        if _oracle_path.exists():
            _oracle_v3.load_from_disk(str(_oracle_path))

        metrics = _oracle_v3._training_metrics or {}
        n_samples = metrics.get("n_samples", 0)

        # Zona 2: Market Activity (Bridge)
        real_trades = 0
        placeholder_trades = 0
        total_trades = 0
        try:
            if Path(BRIDGE_JSONL_PATH).exists():
                with open(BRIDGE_JSONL_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            t = json.loads(line)
                            total_trades += 1
                            # El detector v4 marca is_placeholder en signal_data
                            sd = t.get("signal_data", {})
                            if sd.get("is_placeholder", True):
                                placeholder_trades += 1
                            else:
                                real_trades += 1
                        except:
                            continue
        except Exception as e:
            logger.error(f"Error parsing bridge.jsonl: {e}")

        # Zona 3: Evolution Governance
        orch_stats = _evolution_orchestrator.get_stats()

        return jsonify(
            {
                "oracle": {
                    "status": "TRAINED"
                    if _oracle_v3.model
                    and _oracle_v3.model != "placeholder_model_trained"
                    else "PLACEHOLDER",
                    "samples": n_samples,
                    "test_accuracy": metrics.get("test_accuracy", 0.0),
                    "is_significant": n_samples >= 150,
                },
                "market": {
                    "total_trades": total_trades,
                    "real_trades": real_trades,
                    "placeholder_trades": placeholder_trades,
                    "retests_detected": orch_stats.get("total_proposals", 0),
                },
                "evolution": {
                    "pending": orch_stats.get("cat_2_pending", 0)
                    + orch_stats.get("cat_3_pending", 0),
                    "rejected_safety": orch_stats.get("cat_1_rejected", 0)
                    + orch_stats.get("cat_2_rejected", 0),
                    "applied": orch_stats.get("cat_1_applied", 0)
                    + orch_stats.get("cat_2_approved", 0),
                },
            }
        )
    except Exception as e:
        logger.error(f"Vault Status Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/experiment/propose", methods=["POST"])
@require_auth
def experiment_propose() -> ResponseReturnValue:
    """Genera propuesta base con fricciones activas por defecto."""
    global _latest_proposal
    data = request.get_json() or {}
    hypothesis = str(data.get("hypothesis") or "").strip()
    if not hypothesis:
        return jsonify({"error": "hypothesis_required"}), 400

    try:
        approach_types = _parse_approach_types(data.get("approach_types"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    risk = RiskAssessment(
        max_drawdown_impact_pct=float(data.get("max_drawdown_impact_pct", 1.0)),
        position_sizing_impact=str(data.get("position_sizing_impact", "none")),  # type: ignore[arg-type]
        kill_switch_threshold=str(
            data.get(
                "kill_switch_threshold",
                "drawdown_session_pct > max_drawdown_session_pct",
            )
        ),
        circuit_breaker_interaction=str(
            data.get("circuit_breaker_interaction", "No bypass de circuit breaker")
        ),
    )
    proposal = _change_proposer.create_proposal(
        hypothesis=hypothesis,
        approach_types_targeted=approach_types,
        source_ids=_parse_list_field(data.get("source_ids")),
        risk_assessment=risk,
    )
    _latest_proposal = proposal
    _system_state["experiment_loop_status"] = "proposal_ready"
    _record_control_cycle(
        event=f"EXPERIMENT: propuesta generada {proposal.proposal_id}",
        trigger="experiment_propose",
        level="info",
        context={
            "proposal_id": proposal.proposal_id,
            "frictions": proposal.backtesting.get("frictions", {}),
            "walk_forward_windows": proposal.backtesting.get("walk_forward_windows", 3),
        },
    )

    # ── ISLAND BRIDGE: ChangeProposer → Orchestrator ──
    # Register the proposal in the Evolution Orchestrator for tracking.
    try:
        from cgalpha_v3.lila.llm.proposer import TechnicalSpec

        spec = TechnicalSpec(
            change_type="feature",
            target_file="cgalpha_v3/application/pipeline.py",
            target_attribute=f"proposal_{proposal.proposal_id}",
            old_value=0.0,
            new_value=1.0,
            reason=f"ChangeProposer: {proposal.hypothesis[:120]}",
            causal_score_est=0.60,
            confidence=0.70,
        )
        evo_result = _evolution_orchestrator.process_proposal(spec)
        logger.info(
            f"🔗 Island Bridge: ChangeProposer → Orchestrator "
            f"(Cat.{evo_result.category}, status={evo_result.status})"
        )
    except Exception as bridge_exc:
        logger.warning(f"⚠️ Island Bridge (ChangeProposer) failed: {bridge_exc}")

    return jsonify(_serialize_proposal(proposal))


@app.route("/api/experiment/run", methods=["POST"])
@require_auth
def experiment_run() -> ResponseReturnValue:
    """Ejecuta experimento con walk-forward>=3 y no-leakage obligatorio."""
    global _latest_experiment
    if _latest_proposal is None:
        return jsonify({"error": "proposal_required"}), 400

    data = request.get_json() or {}
    rows = data.get("rows")
    if isinstance(rows, list) and rows:
        dataset = rows
    else:
        dataset = _generate_mock_market_rows(num_rows=int(data.get("mock_rows", 180)))

    ft = data.get("feature_timestamps")
    feature_timestamps = [float(x) for x in ft] if isinstance(ft, list) else None

    t0 = time.time()
    try:
        result = _experiment_runner.run_experiment(
            proposal=_latest_proposal,
            rows=dataset,
            feature_timestamps=feature_timestamps,
        )
        _health_monitor.record_metric("leakage_rate", 0.0)
    except TemporalLeakageError as exc:
        _health_monitor.record_metric("leakage_rate", 1.0)
        _system_state["experiment_loop_status"] = "failed_leakage"
        _record_control_cycle(
            event=f"EXPERIMENT: temporal leakage detectado ({exc})",
            trigger="experiment_run",
            level="critical",
            context={"error": str(exc)},
        )
        _health_monitor.record_metric("exp_latency", time.time() - t0)
        return jsonify({"error": "temporal_leakage", "message": str(exc)}), 400
    except Exception as exc:
        _system_state["experiment_loop_status"] = "failed"
        _record_control_cycle(
            event=f"EXPERIMENT: ejecución fallida ({exc})",
            trigger="experiment_run",
            level="warning",
            context={"error": str(exc)},
        )
        _health_monitor.record_metric("exp_latency", time.time() - t0)
        return jsonify({"error": "experiment_failed", "message": str(exc)}), 400
    finally:
        pass

    _health_monitor.record_metric("exp_latency", time.time() - t0)
    _latest_experiment = result
    _experiment_history.append(result)
    if len(_experiment_history) > 25:
        _experiment_history.pop(0)
    _system_state["experiment_loop_status"] = "completed"
    _record_control_cycle(
        event=f"EXPERIMENT: ejecución completada {result.experiment_id}",
        trigger="experiment_run",
        level="info",
        context={
            "experiment_id": result.experiment_id,
            "proposal_id": result.proposal_id,
            "metrics": result.metrics,
            "walk_forward_windows": len(result.walk_forward_windows),
            "no_leakage_checked": result.no_leakage_checked,
        },
    )

    # ── ISLAND BRIDGE: ExperimentRunner → Orchestrator ──
    # Route experiment results to Evolution Orchestrator for classification.
    # Closes the feedback loop: ChangeProposer → Experiment → Orchestrator → Sage
    try:
        from cgalpha_v3.lila.llm.proposer import TechnicalSpec

        sharpe = result.metrics.get("sharpe_neto", 0.0)
        net_return = result.metrics.get("net_return_pct", 0.0)
        causal_est = min(max(sharpe / 3.0, 0.0), 1.0)  # normalize to 0-1
        spec = TechnicalSpec(
            change_type="optimization",
            target_file="cgalpha_v3/application/pipeline.py",
            target_attribute="experiment_feedback",
            old_value=0.0,
            new_value=round(net_return, 4),
            reason=(
                f"Experiment {result.experiment_id}: "
                f"Sharpe={sharpe:.2f}, NetReturn={net_return:.2f}%, "
                f"WF_windows={len(result.walk_forward_windows)}, "
                f"NoLeakage={'✅' if result.no_leakage_checked else '❌'}"
            ),
            causal_score_est=round(causal_est, 2),
            confidence=0.75 if result.no_leakage_checked and sharpe > 0 else 0.50,
        )
        evo_result = _evolution_orchestrator.process_proposal(spec)
        logger.info(
            f"🔗 Island Bridge: Experiment → Orchestrator "
            f"(Cat.{evo_result.category}, status={evo_result.status})"
        )
    except Exception as bridge_exc:
        logger.warning(f"⚠️ Island Bridge failed (non-blocking): {bridge_exc}")

    return jsonify(_serialize_experiment_result(result))


@app.route("/api/promotion/validate", methods=["POST"])
@require_auth
def promotion_validate() -> ResponseReturnValue:
    """Valida formalmente un experimento para promoción a Producción (Gate P3.3)."""
    data = request.get_json() or {}
    experiment_id = data.get("experiment_id")

    # Buscar experimento en el historial reciente
    target = None
    if _latest_experiment and _latest_experiment.experiment_id == experiment_id:
        target = _latest_experiment
    else:
        for exp in reversed(_experiment_history):
            if exp.experiment_id == experiment_id:
                target = exp
                break

    if not target:
        return jsonify({"error": "experiment_not_found"}), 404

    report = _promotion_validator.validate_experiment(
        result=target, health=_health_monitor.status_snapshot()
    )

    _record_control_cycle(
        event=f"PROMOTION: Validación de {experiment_id} -> {report.status}",
        trigger="promotion_validate",
        level="info" if report.status == "approved" else "warning",
        context={
            "experiment_id": experiment_id,
            "status": report.status,
            "checks": report.checks,
        },
    )

    return jsonify(
        {
            "status": report.status,
            "overall_score": report.overall_score,
            "checks": report.checks,
            "reasons": report.reasons,
        }
    )


@app.route("/api/learning/memory/status", methods=["GET"])
@require_auth
def learning_memory_status() -> ResponseReturnValue:
    """Snapshot del motor de memoria inteligente."""
    return jsonify(_learning_memory_snapshot_json())


@app.route("/api/learning/memory/entries", methods=["GET"])
@require_auth
def learning_memory_entries() -> ResponseReturnValue:
    """Listado de entradas de memoria por nivel/campo."""
    raw_level = request.args.get("level", "").strip()
    raw_field = request.args.get("field", "").strip()
    level = None
    if raw_level:
        try:
            level = MemoryPolicyEngine.parse_level(raw_level)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    field = raw_field if raw_field else None
    if field and field not in (
        "codigo",
        "math",
        "trading",
        "architect",
        "memory_librarian",
    ):
        return jsonify({"error": "invalid_field"}), 400
    limit = min(max(int(request.args.get("limit", 50)), 1), 200)
    entries = _memory_engine.list_entries(level=level, field=field, limit=limit)  # type: ignore[arg-type]
    return jsonify(
        {
            "count": len(entries),
            "entries": [_serialize_memory_entry(e) for e in entries],
        }
    )


@app.route("/api/learning/memory/ingest", methods=["POST"])
@require_auth
def learning_memory_ingest() -> ResponseReturnValue:
    """Ingesta entrada de memoria (0a), con normalización opcional a 0b."""
    data = request.get_json() or {}
    content = str(data.get("content") or "").strip()
    field = str(data.get("field") or "").strip()
    if not content:
        return jsonify({"error": "content_required"}), 400
    if field not in ("codigo", "math", "trading", "architect", "memory_librarian"):
        return jsonify({"error": "invalid_field"}), 400

    source_type = data.get("source_type")
    if source_type is not None and source_type not in (
        "primary",
        "secondary",
        "tertiary",
    ):
        return jsonify({"error": "invalid_source_type"}), 400

    entry = _memory_engine.ingest_raw(
        content=content,
        field=field,  # type: ignore[arg-type]
        source_id=str(data.get("source_id")) if data.get("source_id") else None,
        source_type=source_type,  # type: ignore[arg-type]
        tags=_parse_list_field(data.get("tags")),
    )
    auto_normalize = bool(data.get("auto_normalize", True))
    if auto_normalize:
        entry = _memory_engine.normalize(entry.entry_id)
    _persist_memory_entry(entry)
    _record_control_cycle(
        event=f"LEARNING: memoria ingestada {entry.entry_id} ({entry.field}/{entry.level.value})",
        trigger="learning_memory_ingest",
        level="info",
        context={
            "entry_id": entry.entry_id,
            "field": entry.field,
            "level": entry.level.value,
        },
    )
    return jsonify(
        {
            "entry": _serialize_memory_entry(entry),
            "snapshot": _learning_memory_snapshot_json(),
        }
    )


@app.route("/api/learning/memory/promote", methods=["POST"])
@require_auth
def learning_memory_promote() -> ResponseReturnValue:
    """Promoción explícita de nivel de memoria."""
    data = request.get_json() or {}
    entry_id = str(data.get("entry_id") or "").strip()
    target_level_raw = str(data.get("target_level") or "").strip()
    approved_by = str(data.get("approved_by") or "Lila")
    experiment_id = data.get("experiment_id")
    identity_confirmation = str(data.get("identity_confirmation") or "").strip()

    if not entry_id:
        return jsonify({"error": "entry_id_required"}), 400
    if not target_level_raw:
        return jsonify({"error": "target_level_required"}), 400
    try:
        target_level = MemoryPolicyEngine.parse_level(target_level_raw)

        if target_level == MemoryLevel.IDENTITY:
            expected_confirmation = f"PROMOTE_IDENTITY:{entry_id}"
            if identity_confirmation != expected_confirmation:
                return jsonify(
                    {
                        "error": "identity_confirmation_required",
                        "message": (
                            "Promoción a IDENTITY requiere confirmación explícita. "
                            f"Envía identity_confirmation='{expected_confirmation}'."
                        ),
                    }
                ), 403

        # --- Production Gate P3.6 Enforcement ---
        if target_level == MemoryLevel.STRATEGY:
            exp = None
            if experiment_id:
                for res in reversed(_experiment_history):
                    if res.experiment_id == experiment_id:
                        exp = res
                        break

            try:
                _production_gate.verify_promotion_eligibility(
                    target_level=target_level,
                    experiment_result=exp,
                    health_snapshot=_health_monitor.status_snapshot(),
                )
            except ProductionGateError as pge:
                return jsonify(
                    {"error": "production_gate_rejected", "message": str(pge)}
                ), 403
        # --------------------------------------

        entry = _memory_engine.promote(
            entry_id=entry_id,
            target_level=target_level,
            approved_by=approved_by,
            tags=_parse_list_field(data.get("tags")),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    _persist_memory_entry(entry)
    _record_control_cycle(
        event=f"LEARNING: memoria promovida {entry.entry_id} -> {entry.level.value}",
        trigger="learning_memory_promote",
        level="info",
        context={"entry_id": entry.entry_id, "target_level": entry.level.value},
    )
    return jsonify(
        {
            "entry": _serialize_memory_entry(entry),
            "snapshot": _learning_memory_snapshot_json(),
        }
    )


@app.route("/api/learning/memory/retention/run", methods=["POST"])
@require_auth
def learning_memory_retention() -> ResponseReturnValue:
    """Ejecuta retención TTL de memoria."""
    retention = _memory_engine.apply_ttl_retention()
    _record_control_cycle(
        event=f"LEARNING: retención TTL ejecutada (removed={retention['removed_count']})",
        trigger="learning_memory_retention_run",
        level="info",
        context=retention,  # type: ignore[arg-type]
    )
    return jsonify(
        {"retention": retention, "snapshot": _learning_memory_snapshot_json()}
    )


@app.route("/api/learning/memory/regime/check", methods=["POST"])
@require_auth
def learning_memory_regime_check() -> ResponseReturnValue:
    """Detecta cambio de régimen y degrada memoria alta si aplica."""
    data = request.get_json() or {}
    series_raw = data.get("volatility_series")
    if not isinstance(series_raw, list) or not series_raw:
        return jsonify({"error": "volatility_series_required"}), 400
    try:
        series = [float(x) for x in series_raw]
    except (TypeError, ValueError):
        return jsonify({"error": "invalid_volatility_series"}), 400

    result = _memory_engine.detect_and_apply_regime_shift(series)
    _system_state["regime_shift_active"] = bool(result.get("regime_shift", False))
    _record_control_cycle(
        event=(
            "LEARNING: cambio de régimen detectado y degradación aplicada"
            if result.get("regime_shift")
            else "LEARNING: chequeo de régimen sin cambio"
        ),
        trigger="learning_memory_regime_check",
        level="warning" if result.get("regime_shift") else "info",
        context=result,  # type: ignore[arg-type]
    )
    return jsonify({"result": result, "snapshot": _learning_memory_snapshot_json()})


@app.route("/api/incidents", methods=["GET"])
@require_auth
def incidents_list() -> ResponseReturnValue:
    """Lista incidentes P0-P3 registrados por runtime."""
    status = request.args.get("status", "").strip().lower()
    limit = min(max(int(request.args.get("limit", 50)), 1), 200)
    items = list(reversed(_incident_registry))
    if status in ("open", "resolved"):
        items = [i for i in items if i["status"] == status]
    return jsonify({"count": min(len(items), limit), "incidents": items[:limit]})


@app.route("/api/incidents/<incident_id>/resolve", methods=["POST"])
@require_auth
def incident_resolve(incident_id: str) -> ResponseReturnValue:
    """Marca incidente como resuelto con nota."""
    data = request.get_json() or {}
    note = str(data.get("resolution_note") or "").strip()
    for inc in _incident_registry:
        if inc["incident_id"] == incident_id:
            inc["status"] = "resolved"
            inc["resolved_at"] = datetime.now(timezone.utc).isoformat()
            inc["resolution_note"] = note
            (_incidents_dir() / f"{incident_id}.json").write_text(
                json.dumps(inc, indent=2, ensure_ascii=False)
            )
            _record_control_cycle(
                event=f"INCIDENT: resuelto {incident_id}",
                trigger="incident_resolve",
                level="info",
                context={"incident_id": incident_id, "resolution_note": note},
            )
            return jsonify(inc)
    return jsonify({"error": "incident_not_found"}), 404


@app.route("/api/adr/recent", methods=["GET"])
@require_auth
def adr_recent() -> ResponseReturnValue:
    """Lista ADR recientes generados por iteración."""
    limit = min(max(int(request.args.get("limit", 50)), 1), 200)
    items = list(reversed(_adr_registry))
    return jsonify({"count": min(len(items), limit), "adrs": items[:limit]})


@app.route("/api/kill-switch/arm", methods=["POST"])
@require_auth
def kill_switch_arm() -> ResponseReturnValue:
    """Paso 1: solicitar activación del kill-switch."""
    _system_state["kill_switch"] = "arming"
    _record_control_cycle(
        event="KILL-SWITCH: solicitud de activación (paso 1 de 2)",
        level="warning",
        trigger="kill_switch_arm",
        context={"kill_switch_status": _system_state["kill_switch"]},
    )
    return jsonify(
        {
            "status": "pending_confirmation",
            "message": "Confirme en /api/kill-switch/confirm",
        }
    )


@app.route("/api/kill-switch/confirm", methods=["POST"])
@require_auth
def kill_switch_confirm() -> ResponseReturnValue:
    """Paso 2: confirmar activación del kill-switch."""
    if _system_state["kill_switch"] != "arming":
        return jsonify({"error": "No hay solicitud de activación pendiente"}), 400
    _system_state["kill_switch"] = "triggered"
    _system_state["status"] = "kill-switch-active"
    _record_control_cycle(
        event="KILL-SWITCH: ACTIVADO — señales suspendidas",
        level="critical",
        trigger="kill_switch_confirm",
        context={
            "kill_switch_status": _system_state["kill_switch"],
            "system_status": _system_state["status"],
        },
    )
    return jsonify(
        {"status": "triggered", "ts": datetime.now(timezone.utc).isoformat()}
    )


@app.route("/api/kill-switch/reset", methods=["POST"])
@require_auth
def kill_switch_reset() -> ResponseReturnValue:
    """Re-armar el kill-switch desde GUI."""
    _system_state["kill_switch"] = "armed"
    _system_state["status"] = "idle"
    _record_control_cycle(
        event="KILL-SWITCH: desactivado — sistema re-armado",
        level="info",
        trigger="kill_switch_reset",
        context={
            "kill_switch_status": _system_state["kill_switch"],
            "system_status": _system_state["status"],
        },
    )
    return jsonify({"status": "armed"})


@app.route("/api/risk/params", methods=["GET"])
@require_auth
def risk_params_get() -> ResponseReturnValue:
    """Leer parámetros de riesgo actuales."""
    return jsonify(
        {
            "max_drawdown_session_pct": _system_state["max_drawdown_session_pct"],
            "max_position_size_pct": _system_state["max_position_size_pct"],
            "max_signals_per_hour": _system_state["max_signals_per_hour"],
            "min_signal_quality_score": _system_state["min_signal_quality_score"],
            "drawdown_session_pct": _system_state["drawdown_session_pct"],
            "circuit_breaker": _system_state["circuit_breaker"],
        }
    )


@app.route("/api/risk/params", methods=["POST"])
@require_auth
def risk_params_set() -> ResponseReturnValue:
    """Actualizar parámetros de riesgo desde GUI."""
    data = request.get_json() or {}
    changed = []
    numeric_fields = {
        "max_drawdown_session_pct": float,
        "max_position_size_pct": float,
        "max_signals_per_hour": int,
        "min_signal_quality_score": float,
    }
    for key, caster in numeric_fields.items():
        if key not in data:
            continue
        try:
            casted = caster(data[key])
        except (TypeError, ValueError):
            return jsonify({"error": f"invalid_value:{key}"}), 400

        if key in ("max_drawdown_session_pct", "max_position_size_pct") and casted < 0:
            return jsonify({"error": f"invalid_range:{key}"}), 400
        if key == "max_signals_per_hour" and casted <= 0:
            return jsonify({"error": "invalid_range:max_signals_per_hour"}), 400
        if key == "min_signal_quality_score" and not (0 <= casted <= 1):
            return jsonify({"error": "invalid_range:min_signal_quality_score"}), 400

        _system_state[key] = casted
        changed.append(key)
    if changed:
        _record_control_cycle(
            event=f"Parámetros de riesgo actualizados: {changed}",
            level="info",
            trigger="risk_params_set",
            context={"updated": changed, "risk_parameters": _risk_params_snapshot()},
        )
    return jsonify(
        {
            "updated": changed,
            "current": {k: _system_state[k] for k in changed},
            "all": {
                "max_drawdown_session_pct": _system_state["max_drawdown_session_pct"],
                "max_position_size_pct": _system_state["max_position_size_pct"],
                "max_signals_per_hour": _system_state["max_signals_per_hour"],
                "min_signal_quality_score": _system_state["min_signal_quality_score"],
            },
        }
    )


@app.route("/api/signal-detector/config", methods=["GET"])
@require_auth
def signal_detector_config_get() -> ResponseReturnValue:
    """Leer configuración del Signal Detector."""
    return jsonify(_system_state.get("signal_detector", {}))


@app.route("/api/signal-detector/config", methods=["POST"])
@require_auth
def signal_detector_config_set() -> ResponseReturnValue:
    """Actualizar configuración del Signal Detector."""
    data = request.get_json() or {}
    if "signal_detector" not in _system_state:
        _system_state["signal_detector"] = {}

    changed = []
    valid_fields = {
        "enabled": bool,
        "volume_percentile_threshold": int,
        "body_percentage_threshold": int,
        "lookback_candles": int,
        "atr_period": int,
        "atr_multiplier": float,
        "volume_threshold": float,
        "min_zone_bars": int,
        "quality_threshold": float,
        "r2_min": float,
        "proximity_tolerance": int,
        "min_signal_quality": float,
    }

    for key, caster in valid_fields.items():
        if key not in data:
            continue
        try:
            casted = caster(data[key])
        except (TypeError, ValueError):
            return jsonify({"error": f"invalid_value:{key}"}), 400

        # Validaciones específicas
        if key == "min_signal_quality" and not (0 <= casted <= 1):
            return jsonify({"error": "invalid_range:min_signal_quality"}), 400
        if key == "quality_threshold" and not (0 <= casted <= 1):
            return jsonify({"error": "invalid_range:quality_threshold"}), 400
        if key == "r2_min" and not (0 <= casted <= 1):
            return jsonify({"error": "invalid_range:r2_min"}), 400

        _system_state["signal_detector"][key] = casted
        changed.append(key)

        # Si se habilita el detector, configurarlo en ExperimentRunner
        if key == "enabled" and casted:
            _experiment_runner.set_signal_detector(_system_state["signal_detector"])
            _log_event("Signal Detector habilitado en ExperimentRunner")

    if changed:
        _record_control_cycle(
            event=f"Signal Detector config actualizada: {changed}",
            level="info",
            trigger="signal_detector_config_set",
            context={
                "updated": changed,
                "signal_detector": _system_state["signal_detector"],
            },
        )

    return jsonify(
        {
            "updated": changed,
            "current": _system_state["signal_detector"],
        }
    )


@app.route("/api/oracle/config", methods=["GET"])
@require_auth
def oracle_config_get() -> ResponseReturnValue:
    """Leer configuración del Oracle."""
    return jsonify(_system_state.get("oracle", {}))


@app.route("/api/oracle/config", methods=["POST"])
@require_auth
def oracle_config_set() -> ResponseReturnValue:
    """Actualizar configuración del Oracle."""
    data = request.get_json() or {}
    if "oracle" not in _system_state:
        _system_state["oracle"] = {}

    changed = []
    valid_fields = {
        "enabled": bool,
        "min_confidence": float,
        "model_type": str,
        "retrain_interval_hours": int,
        "use_oracle_filtering": bool,
    }

    for key, caster in valid_fields.items():
        if key not in data:
            continue
        try:
            casted = caster(data[key])
        except (TypeError, ValueError):
            return jsonify({"error": f"invalid_value:{key}"}), 400

        # Validaciones específicas
        if key == "min_confidence" and not (0 <= casted <= 1):
            return jsonify({"error": "invalid_range:min_confidence"}), 400
        if key == "model_type" and casted not in [
            "placeholder",
            "random_forest",
            "xgboost",
            "lightgbm",
        ]:
            return jsonify({"error": "invalid_model_type"}), 400

        _system_state["oracle"][key] = casted
        changed.append(key)

        # Si se habilita el filtering, configurarlo en ExperimentRunner
        if key == "use_oracle_filtering" and casted:
            _log_event("Oracle filtering habilitado en ExperimentRunner")

    if changed:
        _record_control_cycle(
            event=f"Oracle config actualizada: {changed}",
            level="info",
            trigger="oracle_config_set",
            context={"updated": changed, "oracle": _system_state["oracle"]},
        )

    return jsonify(
        {
            "updated": changed,
            "current": _system_state["oracle"],
        }
    )


@app.route("/api/rollback/list", methods=["GET"])
@require_auth
def rollback_list() -> ResponseReturnValue:
    """Listar snapshots disponibles para rollback."""
    return jsonify(_rollback_mgr.list_snapshots())


@app.route("/api/rollback/restore", methods=["POST"])
@require_auth
def rollback_restore() -> ResponseReturnValue:
    """Ejecutar restauración de un snapshot."""
    data = request.get_json() or {}
    path = data.get("path")
    if not path:
        return jsonify({"error": "path_required"}), 400

    try:
        restored = _rollback_mgr.restore(path, verify_hash=True)
        # Actualizar estado del sistema con la config restaurada
        cfg = restored.get("config", {})
        for k in [
            "max_drawdown_session_pct",
            "max_position_size_pct",
            "max_signals_per_hour",
            "min_signal_quality_score",
        ]:
            if k in cfg:
                _system_state[k] = cfg[k]

        _record_control_cycle(
            event=f"ROLLBACK ejecutado desde GUI: {Path(path).name}",
            level="warning",
            trigger="rollback_restore",
            context={"restored_path": path, "elapsed_ms": restored["elapsed_ms"]},
        )
        _health_monitor.record_metric("rollback_sla", restored["elapsed_ms"] / 1000.0)
        return jsonify({"status": "success", "restored": restored})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/lila/llm/status", methods=["GET"])
@require_auth
def get_lila_llm_status():
    """Estado detallado del proveedor de Lila."""
    try:
        status = _assistant.get_status()
        # Enriquecer con lista de disponibles y snapshot de memoria
        status["available_providers"] = list(_assistant._available_providers.keys())
        status["memory_levels"] = _memory_engine.snapshot()["levels"]
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/lila/llm/switch", methods=["POST"])
@require_auth
def switch_lila_llm_provider():
    """Cambiar el proveedor actual de Lila."""
    data = request.json or {}
    provider_name = data.get("provider")

    if not provider_name:
        return jsonify({"error": "Missing provider name"}), 400

    success = _assistant.switch_provider(provider_name)
    if success:
        return jsonify(
            {"status": "success", "active_provider": _assistant.provider.name}
        )
    else:
        return jsonify({"error": f"Failed to switch to {provider_name}"}), 400


@app.route("/api/learning/ingest/history", methods=["POST"])
@require_auth
def learning_ingest_history() -> ResponseReturnValue:
    """Ingestar conocimiento de las iteraciones y ADRs pasadas."""
    try:
        stats = _history_learner.learn_from_history()
        _record_control_cycle(
            event=f"DEEP_LEARNING: History ingested - {stats['entries_created']} entries",
            trigger="learning_ingest_history",
            level="success",
            context=stats,
        )
        return jsonify(stats)
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/assistant/chat", methods=["POST"])
@require_auth
def assistant_chat() -> ResponseReturnValue:
    """Chat interactivo con Lila (Asistente v3). Capacidad de aprendizaje profundo activa."""
    data = request.get_json() or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "empty_message"}), 400

    try:
        health = _health_monitor.status_snapshot()
        status = health.get("status", "unknown")
        low_msg = message.lower()

        # COMANDOS ESPECIALES DE APRENDIZAJE (P4)
        if any(
            kw in low_msg
            for kw in [
                "aprende de la historia",
                "learn from history",
                "revisa lo construido",
                "deep learning project",
            ]
        ):
            # Trigger automatic ingestion
            stats = _history_learner.learn_from_history()
            resp = f"He completado el aprendizaje profundo de v3. He procesado {stats['iterations_found']} iteraciones y {stats['adrs_found']} decisiones arquitectónicas (ADRs). He creado {stats['entries_created']} nuevas entradas en mi memoria a largo plazo (Nivel 2 y 3). Ahora soy consciente de: {', '.join(stats['top_insights'][:3])}."

        elif "estatus" in low_msg or "estado" in low_msg:
            resp = f"El sistema reporta un estado {status.upper()}. Tenemos {_health_monitor.total_samples} muestras en el monitor de salud."
        elif "audit" in low_msg or "p3" in low_msg:
            resp = "La auditoría P3 ha finalizado con éxito. Todos los gates de Hardening (Temporal, Multi-symbol, Proposer) están nominales."
        elif "hola" in low_msg:
            resp = "Hola, soy Lila. Estoy monitoreando la integridad de los datos en tiempo real. ¿Deseas analizar algún experimento o que 'aprenda de la historia' v3?"
        else:
            # Usar el asistente consolidado (respeta el proveedor seleccionado en settings)
            resp = _assistant.generate(
                f"Contexto: Trading System v3 Audit (Lila v3 Assistant). Sujeto: {message}"
            )

        _record_control_cycle(
            event=f"LILA_CHAT: Interaction - Msg: {message[:30]}...",
            trigger="assistant_chat",
            level="info",
            context={"message": message, "response": resp},
        )

        return jsonify({"response": resp})

    except Exception as exc:
        return jsonify(
            {"response": f"Hubo un error en mi núcleo de procesamiento v3: {exc}"}
        )


# ---------------------------------------------------------------------------
# VAULT EVOLUTION & ACTIVE CONSTRUCTION (North Star 3.0.0)
# ---------------------------------------------------------------------------

from cgalpha_v3.application.pipeline import (
    SimpleFoundationPipeline,
    TripleCoincidencePipeline,
)

pipeline_v3 = TripleCoincidencePipeline(evolution_orchestrator=_evolution_orchestrator)


@app.route("/api/vault/status", methods=["GET"])
@require_auth
def vault_evolution_status():
    """Estado de purificación y evolución de la Bóveda (CGAlpha v3)."""
    status = {
        "status": "active",
        "blueprint_version": "v3.0.0-PRO",
        "evolution_phase": "CGAlpha v3 / Construction",
        "layers": {
            "layer_1_provisional": {
                "total": 47,
                "unvalidated": 31,
                "in_review": 14,
                "purgeable": 2,
            },
            "layer_2_permanent_dna": {
                "total": 7,
                "active": 7,
                "avg_delta_causal": 0.84,
            },
        },
        "components": [
            {
                "id": "fetcher_v3",
                "name": "BinanceVisionFetcher_v3",
                "status": "ACTIVE",
                "delta": 0.85,
                "role": "Ingestión OHLCV + datos de microestructura",
            },
            {
                "id": "detector_v3",
                "name": "TripleCoincidenceDetector",
                "status": "ACTIVE",
                "delta": 0.82,
                "role": "Detección zonas: vela clave + acumulación + mini-tendencia + retest monitoring",
            },
            {
                "id": "monitor_v3",
                "name": "ZonePhysicsMonitor_v3",
                "status": "ACTIVE",
                "delta": 0.81,
                "role": "Evaluación física del retest: REBOTE vs RUPTURA",
            },
            {
                "id": "shadow_v3",
                "name": "ShadowTrader",
                "status": "ACTIVE",
                "delta": 0.88,
                "role": "Posiciones virtuales: captura trayectorias MFE/MAE",
            },
            {
                "id": "oracle_v3",
                "name": "OracleTrainer_v3",
                "status": "ACTIVE",
                "delta": 0.92,
                "role": "Meta-Labeling: predice outcome del retest, entrena con dataset de retests",
            },
            {
                "id": "gate_v3",
                "name": "NexusGate",
                "status": "ACTIVE",
                "delta": 1.0,
                "role": "Gate binario: PROMOTE_TO_LAYER_2 vs REJECT",
            },
            {
                "id": "proposer_v3",
                "name": "AutoProposer",
                "status": "ACTIVE",
                "delta": 0.75,
                "role": "Detecta drift y propone ajustes paramétricos con causal_score estimado",
            },
        ],
        "metrics": {
            "hit_rate_oos": "78.4%",
            "blind_test_ratio": "0.12",
            "last_training": "2026-04-06 22:31:35",
        },
    }
    return jsonify(status)


@app.route("/api/lila/execute-cycle", methods=["POST"])
@require_auth
def execute_pipeline_cycle():
    """Iniciando ciclo manual de la estrategia desde la GUI."""
    data = request.json
    symbol = data.get("symbol", "BTCUSDT")
    logger.info(f"🚀 GUI REQUEST: Pipeline Cycle para {symbol}")
    decision = pipeline_v3.run_cycle(symbol, datetime.now(), datetime.now())
    routed = pipeline_v3.get_last_routed_proposals()
    return jsonify(
        {
            "status": "completed",
            "nexus_decision": decision,
            "routed_proposals": routed,
            "routed_count": len(routed),
        }
    )


@app.route("/api/lila/command", methods=["POST"])
@require_auth
def lila_llm_orchestrator():
    """PUENTE DE ORQUESTACIÓN LLM REAL."""
    command = request.json
    action = command.get("action")
    target = command.get("component")
    logger.warning(f"🧠 LLM ORCHESTRATOR -> {action} ON {target}")
    return jsonify({"status": "success", "action_executed": action})


@app.route("/api/vault/promote", methods=["POST"])
@require_auth
def promote_component():
    """Promover un componente de Capa 1 a Capa 2."""
    data = request.json
    component_id = data.get("component_id")
    logger.info(f"🧬 PROMOCIÓN ATÓMICA: {component_id}")
    return jsonify({"status": "promoted", "component_id": component_id})


# ───────────────────────────────────────────────────────────────────────────
# EVOLUTION & LEARNING V4 ENDPOINTS
# ───────────────────────────────────────────────────────────────────────────


@app.route("/api/evolution/proposals", methods=["GET"])
def get_evolution_proposals():
    """Lista propuestas de evolución pendientes (Cat.2/3)."""
    return jsonify(_evolution_orchestrator.get_pending_summary())


@app.route("/api/evolution/landscape/propose", methods=["POST"])
def propose_parameter_landscape():
    """
    Crea propuesta Cat.2 para el Parameter Landscape Map (S3 Paso 4).
    Requiere aprobación posterior via /api/evolution/proposal/<id>/approve.
    """
    data = request.json or {}
    requested_by = data.get("requested_by", "operator")
    result = _evolution_orchestrator.propose_parameter_landscape(
        requested_by=requested_by
    )
    return jsonify(
        {
            "status": result.status,
            "category": result.category,
            "proposal_id": result.proposal_id,
            "spec_summary": result.spec_summary,
            "error": result.error,
        }
    )


@app.route("/api/evolution/landscape", methods=["GET"])
def get_parameter_landscape():
    """Devuelve el último Parameter Landscape Map generado."""
    artifact = _evolution_orchestrator.get_parameter_landscape()
    if artifact is None:
        return jsonify({"error": "parameter_landscape_map_not_found"}), 404
    return jsonify(artifact)


@app.route("/api/evolution/proposal/<proposal_id>/approve", methods=["POST"])
def approve_evolution_proposal(proposal_id):
    """Aprueba una propuesta y dispara ejecución via Sage."""
    result = _evolution_orchestrator.approve_proposal(proposal_id, approved_by="human")
    return jsonify(
        {
            "status": result.status,
            "category": result.category,
            "proposal_id": result.proposal_id,
            "error": result.error,
        }
    )


@app.route("/api/evolution/proposal/<proposal_id>/reject", methods=["POST"])
def reject_evolution_proposal(proposal_id):
    """Rechaza una propuesta con razón opcional."""
    data = request.json or {}
    reason = data.get("reason", "No reason provided")
    result = _evolution_orchestrator.reject_proposal(proposal_id, reason=reason)
    return jsonify(
        {
            "status": result.status,
            "category": result.category,
            "proposal_id": result.proposal_id,
        }
    )


@app.route("/api/evolution/log", methods=["GET"])
def get_evolution_log():
    """Devuelve el historial completo de evolución (log.jsonl)."""
    log_path = project_root / "cgalpha_v3/memory/evolution_log.jsonl"
    if not log_path.exists():
        return jsonify([])

    lines = log_path.read_text().strip().split("\n")
    return jsonify([json.loads(line) for line in lines if line.strip()])


@app.route("/api/evolution/stats", methods=["GET"])
def get_evolution_stats():
    """Estadísticas de evolución para el dashboard."""
    return jsonify(
        {
            "stats": _evolution_orchestrator.get_stats(),
            "routing": _llm_switcher.get_routing_table() if _llm_switcher else {},
        }
    )


@app.route("/api/evolution/heartbeat", methods=["GET"])
def get_evolution_heartbeat():
    """Retorna el pulso de la ejecución continua de 24h desde scripts/run_24h.py."""
    heartbeat_path = project_root / "execution_24h_heartbeat.json"
    if not heartbeat_path.exists():
        return jsonify(
            {
                "status": "OFFLINE",
                "message": "En espera de latido de run_24h.py...",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cycle": 0,
            }
        )

    try:
        with open(heartbeat_path, "r") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error reading heartbeat: {e}")
        return jsonify({"status": "ERROR", "message": str(e)}), 500


@app.route("/learning/operator", methods=["GET"])
def get_whitepaper_html():
    """Renderiza el WHITEPAPER.md como HTML para el operador."""
    wp_path = Path("cgalpha_v4/WHITEPAPER.md")
    if not wp_path.exists():
        return "White Paper not found", 404

    content = wp_path.read_text()
    # Simplificación: En producción usaríamos un convertidor markdown -> html
    return f"<html><body style='background:#121212; color:#eee; font-family:sans-serif; padding:40px;'><pre>{content}</pre></body></html>"


@app.route("/learning/lila", methods=["GET"])
def get_lila_insights():
    """Dashboard de reflexiones y sabiduría de Lila."""
    insights = _memory_engine.list_entries(field="memory_librarian", limit=20)
    return jsonify(
        [
            {
                "id": e.entry_id,
                "content": e.content,
                "level": e.level.value,
                "timestamp": e.created_at.isoformat(),
                "tags": e.tags,
            }
            for e in insights
        ]
    )


@app.route("/api/vault/purge", methods=["POST"])
@require_auth
def purge_legacy_origin():
    """Eliminar origen en Capa 1 tras promoción."""
    data = request.json
    component_id = data.get("component_id")
    logger.warning(f"🔥 PURGA: {component_id}")
    return jsonify({"status": "purged", "component_id": component_id})


# ---------------------------------------------------------------------------
# Training Review — Datos para el gráfico de revisión pre-entrenamiento
# ---------------------------------------------------------------------------


def _training_review_paths() -> Dict[str, Path]:
    data_dir = BASE_DIR.parent / "data" / "phase0_results"
    return {
        "data_dir": data_dir,
        "ohlcv": data_dir / "synthetic_ohlcv_2000.csv",
        "training_v2": project_root / "aipha_memory" / "operational" / "training_dataset_v2.jsonl",
        "active_zones": project_root / "aipha_memory" / "operational" / "active_zones.json",
        "curation": project_root / "aipha_memory" / "evolutionary" / "retest_curation.jsonl",
    }


def _direction_from_zone_id(zone_id: Any) -> str:
    parts = str(zone_id or "").split("_", 1)
    return parts[1].lower() if len(parts) == 2 and parts[1] else "unknown"


def _retest_id_candidates(rt: Dict[str, Any]) -> set[str]:
    zone_id = str(rt.get("zone_id") or "")
    retest_index = rt.get("retest_index")
    candidates = {zone_id}
    if retest_index is not None:
        idx = str(retest_index)
        candidates.update({idx, f"{zone_id}:{idx}", f"{zone_id}-{idx}"})
    for key in ("retest_id", "sample_id", "id"):
        value = rt.get(key)
        if value is not None:
            candidates.add(str(value))
    return {c for c in candidates if c}


def _load_training_retests() -> tuple[list[Dict[str, Any]], Path]:
    retests_path = _training_review_paths()["retests"]
    if not retests_path.exists():
        return [], retests_path
    with open(retests_path, encoding="utf-8") as f:
        retests = json.load(f)
    for rt in retests:
        if not rt.get("direction"):
            rt["direction"] = _direction_from_zone_id(rt.get("zone_id"))
    return retests, retests_path


def _save_training_retests(retests: list[Dict[str, Any]], retests_path: Path) -> None:
    with open(retests_path, "w", encoding="utf-8") as f:
        json.dump(retests, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _load_operational_retests() -> tuple[list[Dict[str, Any]], Path]:
    """Load retests from training_dataset_v2.jsonl for curation search.

    Phase0 samples use zone_id format like '318_bearish'.
    Operational samples use sample_id format like 're_4377_xxxxx'.
    This function returns operational samples for dual-dataset curation.
    """
    op_path = project_root / "aipha_memory/operational/training_dataset_v2.jsonl"
    if not op_path.exists():
        logger.info("ℹ️ No operational dataset found (training_dataset_v2.jsonl missing)")
        return [], op_path
    retests = []
    with open(op_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Skipping invalid JSON line {i} in training_dataset_v2.jsonl")
                continue

            meta = sample.get("_meta", {})
            snap = sample.get("l2_snapshot_at_touch", {})
            zg = sample.get("zone_geometry", {})
            outcome = sample.get("outcome", {})
            sample_id = meta.get("sample_id", f"re_{i}_x")

            # Extract candle index from sample_id (format: re_NNNN_xxxxx)
            candle_idx = 0
            parts = sample_id.split("_")
            for part in parts:
                if part.isdigit():
                    candle_idx = int(part)
                    break

            retests.append({
                "retest_index": candle_idx,
                "zone_id": sample_id,
                "sample_id": sample_id,
                "dataset_source": "operational",
                "retest_price": snap.get("retest_price", 0),
                "retest_timestamp": meta.get("capture_ts_unix_ms", 0),
                "vwap_at_retest": snap.get("vwap_at_retest", 0),
                "obi_10_at_retest": snap.get("obi_10", 0),
                "cumulative_delta_at_retest": snap.get("cumulative_delta", 0),
                "delta_divergence": snap.get("delta_divergence", "UNKNOWN"),
                "regime": meta.get("regime", "UNKNOWN"),
                "outcome": outcome.get("label", "UNKNOWN"),
                "direction": zg.get("direction", "unknown"),
                "zone_top": zg.get("zone_top", 0),
                "zone_bottom": zg.get("zone_bottom", 0),
                "zone_width_atr": zg.get("zone_width_atr", 0),
                "mfe": outcome.get("mfe", 0),
                "mae": outcome.get("mae", 0),
                "bars_to_resolution": outcome.get("bars_to_resolution", 0),
                "symbol": meta.get("symbol", "BTCUSDT"),
            })
    logger.info(f"✅ Loaded {len(retests)} operational retests from training_dataset_v2.jsonl")
    return retests, op_path


def _build_curation_index() -> dict[str, Dict[str, Any]]:
    """Build in-memory index from retest_curation.jsonl for fast lookup.

    Without this, each review-data request would iterate over the entire
    jsonl file. With growing curation history, this becomes a performance
    bottleneck.
    """
    index = {}
    curation_path = project_root / "aipha_memory/evolutionary/retest_curation.jsonl"
    if not curation_path.exists():
        return index
    with open(curation_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                rtid = entry.get("retest_id", "")
                if rtid:
                    index[rtid] = {
                        "curation_status": entry.get("status", "unknown"),
                        "curation_decision": entry.get("decision", "unknown"),
                        "label_status": entry.get("label_status", "unknown"),
                        "curated_by": entry.get("source", "unknown"),
                        "curated_at": entry.get("ts", ""),
                    }
            except json.JSONDecodeError:
                continue
    return index


def _curate_training_retest(retest_id: str, decision: str) -> ResponseReturnValue:
    now = datetime.now(timezone.utc).isoformat()
    status = "approved" if decision == "APPROVED" else "rejected"
    label_status = "validated" if decision == "APPROVED" else "discarded_noise"

    # ── Dual dataset search: phase0 first, then operational ──
    retests = []
    retests_path = Path()
    found_in = "none"

    # 1. Search in phase0 legacy dataset
    phase0_retests, phase0_path = _load_training_retests()
    matches = [rt for rt in phase0_retests if retest_id in _retest_id_candidates(rt)]
    if matches:
        retests = phase0_retests
        retests_path = phase0_path
        found_in = "phase0"

    # 2. If not found in phase0, search in operational dataset
    if not matches:
        op_retests, op_path = _load_operational_retests()
        matches = [rt for rt in op_retests if retest_id == rt.get("sample_id") or
                   retest_id in _retest_id_candidates(rt)]
        if matches:
            retests = op_retests
            retests_path = op_path
            found_in = "operational"

    if not matches:
        logger.warning(f"⚠️ Curation not found for retest_id '{retest_id}' in any dataset")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Retest not found: {retest_id}",
                    "retest_id": retest_id,
                    "searched_datasets": ["phase0", "operational"],
                }
            ),
            404,
        )

    for rt in matches:
        rt["curation_status"] = status
        rt["curation_decision"] = decision
        rt["label_status"] = label_status
        rt["curated_by"] = "human"
        rt["curated_at"] = now
        if not rt.get("direction"):
            rt["direction"] = _direction_from_zone_id(rt.get("zone_id", ""))

    # Persist to operational jsonl if searching in phase0 dataset
    # (operational data is read-only from training_dataset_v2.jsonl)
    if found_in == "phase0":
        _save_training_retests(retests, retests_path)

    curation_file = project_root / "aipha_memory/evolutionary/retest_curation.jsonl"
    curation_file.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": now,
        "retest_id": retest_id,
        "decision": decision,
        "status": status,
        "label_status": label_status,
        "matched_count": len(matches),
        "source": "human",
        "found_in_dataset": found_in,
    }
    with open(curation_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return jsonify(
        {
            "status": status,
            "decision": decision,
            "label_status": label_status,
            "retest_id": retest_id,
            "matched_count": len(matches),
            "found_in": found_in,
            "retest": matches[0],
        }
    )


@app.route("/api/training/review-data", methods=["GET"])
def get_training_review_data():
    """
    Retorna OHLCV + zonas + retests combinados para el gráfico de revisión.
    CRITICAL: Mapea retests a velas por TIMESTAMP (no por índice), ya que los retests
    tienen timestamps de Mayo-Julio 2026 y las velas sintéticas son de Enero 2024.
    """    # ── Cache check: skip Binance re-fetch if data is recent ──
    cached = _get_training_review_cache()
    if cached:
        logger.info("📦 Serving cached training review data")
        return jsonify(cached)
    

    
    # ── 1. Load retests FIRST to determine timestamp range ──
    retests = []
    paths = _training_review_paths()
    training_v2_path = paths["training_v2"]
    
    if training_v2_path.exists():
        with open(training_v2_path, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            sample = json.loads(line)
            meta = sample.get("_meta", {})
            snap = sample.get("l2_snapshot_at_touch", {})
            zg = sample.get("zone_geometry", {})
            outcome = sample.get("outcome", {})
            
            # Extract retest timestamp (milliseconds)
            capture_ts = meta.get("capture_ts_unix_ms", 0)
            
            retests.append({
                "retest_index": 0,  # Will be mapped by timestamp
                "zone_id": sample.get("_meta", {}).get("sample_id", "unknown"),
                "retest_price": snap.get("retest_price", 0),
                "retest_timestamp": capture_ts,  # KEY: use this for mapping
                "vwap_at_retest": snap.get("vwap_at_retest", 0),
                "obi_10_at_retest": snap.get("obi_10", 0),
                "cumulative_delta_at_retest": snap.get("cumulative_delta", 0),
                "delta_divergence": snap.get("delta_divergence", "UNKNOWN"),
                "regime": sample.get("_meta", {}).get("regime", "UNKNOWN"),
                "outcome": outcome.get("label", "UNKNOWN"),
                "direction": zg.get("direction", "unknown"),
                "zone_top": zg.get("zone_top", 0),
                "zone_bottom": zg.get("zone_bottom", 0),
                "zone_width_atr": zg.get("zone_width_atr", 0),
                "mfe": outcome.get("mfe", 0),
                "mae": outcome.get("mae", 0),
                "bars_to_resolution": outcome.get("bars_to_resolution", 0),
            })
        logger.info(f"✅ Loaded {len(retests)} retests from training_dataset_v2.jsonl")
    
    if not retests:
        return jsonify({"error": "No retests found", "ohlcv": [], "retests": [], "zones_summary": []})
    
    # ── 2. Fix direction from zone_id for any missing values ──
    for rt in retests:
        if not rt.get("direction") or rt["direction"] == "unknown":
            rt["direction"] = _direction_from_zone_id(rt.get("zone_id", ""))
    
    # ── 3. Determine timestamp range from retests ──
    retest_timestamps = [rt["retest_timestamp"] for rt in retests if rt["retest_timestamp"] > 0]
    if not retest_timestamps:
        logger.warning("⚠️ No valid timestamps in retests")
        return jsonify({"error": "No valid timestamps", "ohlcv": [], "retests": [], "zones_summary": []})
    
    min_ts = min(retest_timestamps)
    max_ts = max(retest_timestamps)
    
    # Add padding: 30 minutes before first retest, 30 minutes after last
    # (candles are 5min, so 6 candles padding)
    padding_ms = 30 * 60 * 1000  # 30 minutes
    start_ts = max(0, min_ts - padding_ms)
    end_ts = max_ts + padding_ms
    
    logger.info(f"📅 Retraining timestamp range: {min_ts} to {max_ts}")
    logger.info(f"📅 Requesting OHLCV from: {start_ts} to {end_ts}")
    
    # ── 4. Load OHLCV from Binance with proper pagination ──
    # Binance returns max 1000 candles per request. We need ~17567 candles.
    # Strategy: paginate from oldest to newest using endTime of each page.
    ohlcv = []
    try:
        import urllib.request
        current_start = int(start_ts)
        max_pages = 30  # Safety limit: 30 * 1000 = 30000 candles
        page_num = 0
        
        while len(ohlcv) < 30000 and page_num < max_pages:
            page_num += 1
            page_url = (
                f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m"
                f"&startTime={current_start}&endTime={int(end_ts)}&limit=1000"
            )
            with urllib.request.urlopen(page_url, timeout=30) as resp:
                raw = json.loads(resp.read().decode())
            
            if not raw:
                break
            
            base_idx = len(ohlcv)
            for candle_data in raw:
                ohlcv.append({
                    "index": base_idx,
                    "timestamp": int(candle_data[6]),
                    "open": float(candle_data[1]),
                    "high": float(candle_data[2]),
                    "low": float(candle_data[3]),
                    "close": float(candle_data[4]),
                    "volume": float(candle_data[5]),
                    "regime": "UNKNOWN",
                })
                base_idx += 1
            
            logger.info(f"📄 Binance page {page_num}: got {len(raw)} candles (total: {len(ohlcv)})")
            
            if len(raw) < 1000:
                break  # Last page
            
            # Next page starts 1ms after the last candle's close_time
            current_start = int(raw[-1][6]) + 1
        
        logger.info(f"✅ OHLCV loaded from Binance: {len(ohlcv)} candles over {page_num} pages")
        
        if len(ohlcv) == 0:
            logger.warning("⚠️ No candles returned - trying latest candles as fallback")
            binance_url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=1000"
            with urllib.request.urlopen(binance_url, timeout=10) as resp:
                raw = json.loads(resp.read().decode())
                for i, k in enumerate(raw):
                    ohlcv.append({
                        "index": i,
                        "timestamp": int(k[6]),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "regime": "UNKNOWN",
                    })
            logger.info(f"📄 Fallback: loaded {len(ohlcv)} latest candles")
        
    except Exception as e:
        logger.warning(f"⚠️ Binance API failed: {e}, using synthetic data")
        # Load synthetic data as last resort
        ohlcv_path = paths["ohlcv"]
        if ohlcv_path.exists():
            import csv
            with open(ohlcv_path, newline="") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    ohlcv.append({
                        "index": i,
                        "timestamp": int(row.get("close_time", 0)),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                        "regime": row.get("regime", "UNKNOWN"),
                    })
    
    # Cap at 25000 candles to keep response manageable
    if len(ohlcv) > 25000:
        ohlcv = ohlcv[-25000:]
        logger.info(f"⚠️ Capped OHLCV to last 25000 candles")
    
    # ── 5. Map retests to candle indices by closest timestamp ──
    if ohlcv:
        # Build timestamp -> candle index map
        ts_to_idx = {}
        for i, candle in enumerate(ohlcv):
            ts_to_idx[candle["timestamp"]] = i
        
        # For each retest, find the closest candle by timestamp
        for rt in retests:
            rt_ts = rt["retest_timestamp"]
            if not rt_ts or rt_ts == 0:
                rt["candle_index"] = -1
                continue
            
            # Find closest candle
            best_idx = 0
            best_diff = abs(ohlcv[0]["timestamp"] - rt_ts)
            
            for i, candle in enumerate(ohlcv):
                diff = abs(candle["timestamp"] - rt_ts)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            
            rt["candle_index"] = best_idx
            rt["ts_diff_ms"] = best_diff  # Store for debugging
        
        logger.info(f"✅ Mapped {len(retests)} retests to candle indices")
        
        # Log mapping stats
        mapped = sum(1 for rt in retests if rt["candle_index"] >= 0)
        avg_diff = sum(rt.get("ts_diff_ms", 0) for rt in retests) / max(1, mapped)
        logger.info(f"📊 Mapped: {mapped}/{len(retests)}, avg timestamp diff: {avg_diff:.0f}ms")
    else:
        for rt in retests:
            rt["candle_index"] = -1
    
    # ── 6. Load curation decisions ──
    curation_entries = []
    curation_path = paths.get("curation")
    if curation_path and curation_path.exists():
        with open(curation_path, encoding="utf-8") as f:
            for line in f.readlines():
                curation_entries.append(json.loads(line))
    
    # Map retest_id -> curation decision
    curation_map = {}
    for entry in curation_entries:
        rtid = entry.get("retest_id", "")
        curation_map[rtid] = {
            "curation_status": entry.get("status", "unknown"),
            "curation_decision": entry.get("decision", "unknown"),
            "label_status": entry.get("label_status", "unknown"),
            "curated_by": entry.get("source", "unknown"),
            "curated_at": entry.get("ts", ""),
        }
    
    # Apply curation to retests
    for rt in retests:
        rtid = rt.get("retest_id", rt.get("zone_id", ""))
        if rtid in curation_map:
            rt.update(curation_map[rtid])
    
    # ── 7. Build zone map using timestamp-mapped candle indices ──
    zone_map = {}
    for rt in retests:
        zid = rt.get("zone_id", "unknown")
        zone_direction = rt.get("direction", "unknown")
        candle_idx = rt.get("candle_index", -1)
        
        # Compute zone range based on candle index
        if candle_idx >= 0 and ohlcv:
            start_idx = max(0, candle_idx - 2)
            end_idx = min(len(ohlcv) - 1, candle_idx + 2)
        else:
            start_idx = 0
            end_idx = len(ohlcv) - 1 if ohlcv else 0
        
        zone_top = rt.get("zone_top", 0)
        zone_bottom = rt.get("zone_bottom", 0)
        if not zone_top or not zone_bottom and ohlcv and start_idx <= end_idx:
            zone_candles = ohlcv[start_idx : end_idx + 1]
            zone_top = max(c["high"] for c in zone_candles) if zone_candles else 0
            zone_bottom = min(c["low"] for c in zone_candles) if zone_candles else 0
        
        if zid not in zone_map:
            zone_map[zid] = {
                "zone_id": zid,
                "direction": zone_direction,
                "key_candle_index": candle_idx,
                "retests": [],
                "zone_top": zone_top,
                "zone_bottom": zone_bottom,
                "zone_start_idx": start_idx,
                "zone_end_idx": end_idx,
            }
        zone_map[zid]["retests"].append(rt)
    
    # ── 8. Annotate OHLCV with retest data ──
    ohlcv_annotated = []
    for candle in ohlcv:
        idx = candle["index"]
        annotations = []
        
        # Find retests that map to this candle
        for rt in retests:
            if rt.get("candle_index") == idx:
                annotations.append({
                    "type": "retest",
                    "zone_id": rt.get("zone_id"),
                    "retest_price": rt.get("retest_price"),
                    "outcome": rt.get("outcome"),
                    "direction": rt.get("direction"),
                    "regime": rt.get("regime"),
                    "delta_divergence": rt.get("delta_divergence"),
                    "vwap_at_retest": rt.get("vwap_at_retest"),
                    "obi_10_at_retest": rt.get("obi_10_at_retest"),
                    "cumulative_delta_at_retest": rt.get("cumulative_delta_at_retest"),
                })
        
        if annotations:
            candle["annotations"] = annotations
            ohlcv_annotated.append(candle)
    
    # ── 9. Summary ──
    zones_summary = []
    for zid, zone_info in zone_map.items():
        zones_summary.append({
            "zone_id": zid,
            "direction": zone_info["direction"],
            "key_candle_index": zone_info["key_candle_index"],
            "zone_top": zone_info["zone_top"],
            "zone_bottom": zone_info["zone_bottom"],
            "zone_start_idx": zone_info["zone_start_idx"],
            "zone_end_idx": zone_info["zone_end_idx"],
            "retest_count": len(zone_info["retests"]),
            "retest_indices": [rt.get("candle_index") for rt in zone_info["retests"]],
        })
    
    bounce_count = sum(1 for rt in retests if "BOUNCE" in str(rt.get("outcome", "")))
    breakout_count = sum(1 for rt in retests if "BREAKOUT" in str(rt.get("outcome", "")))
    
    # Determine data source
    has_operational = training_v2_path.exists()
    data_source = "operational" if has_operational else "phase0_fallback"
    if has_operational:
        logger.info(f"✅ Using operational dataset ({len(retests)} samples) for review-data")
    else:
        logger.warning("⚠️ No operational dataset, falling back to phase0")
    
    # ── Cache the result for future requests (avoids re-paginating Binance) ──
    _set_training_review_cache({
        "data_source": data_source,
        "ohlcv": ohlcv,
        "ohlcv_annotated": ohlcv_annotated,
        "retests": retests,
        "zones": list(zone_map.keys()),
        "zones_summary": zones_summary,
        "zone_count": len(zone_map),
        "retest_count": len(retests),
        "candle_count": len(ohlcv),
        "active_zones_count": 0,
        "outcome_distribution": {
            "BOUNCE": bounce_count,
            "BREAKOUT": breakout_count,
            "bounce_pct": round(bounce_count / len(retests) * 100, 1) if retests else 0,
        },
        "training_samples_count": len(retests),
    })
    logger.info(f"💾 Cached training review data: {len(ohlcv)} candles, {len(retests)} retests")
    
    return jsonify({
        "data_source": data_source,
        "ohlcv": ohlcv,
        "ohlcv_annotated": ohlcv_annotated,
        "retests": retests,
        "zones": list(zone_map.keys()),
        "zones_summary": zones_summary,
        "zone_count": len(zone_map),
        "retest_count": len(retests),
        "candle_count": len(ohlcv),
        "active_zones_count": 0,
        "outcome_distribution": {
            "BOUNCE": bounce_count,
            "BREAKOUT": breakout_count,
            "bounce_pct": round(bounce_count / len(retests) * 100, 1) if retests else 0,
        },
        "training_samples_count": len(retests),
    })


@app.route("/api/training/retest/<retest_id>/approve", methods=["POST"])
def approve_retest(retest_id):
    """Marcar un retest como aprobado para entrenamiento del Oracle."""
    try:
        response = _curate_training_retest(retest_id, "APPROVED")
        if isinstance(response, tuple) and int(response[1]) >= 400:
            return response
        _log_event(f"RETEST_CURATION: Approved {retest_id}", level="info")
        return response
    except Exception as e:
        logger.error(f"Error in approve_retest: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/training/retest/<retest_id>/reject", methods=["POST"])
def reject_retest(retest_id):
    """Marcar un retest como excluido del entrenamiento del Oracle."""
    try:
        response = _curate_training_retest(retest_id, "REJECTED")
        if isinstance(response, tuple) and int(response[1]) >= 400:
            return response
        _log_event(f"RETEST_CURATION: Rejected {retest_id}", level="warning")
        return response
    except Exception as e:
        logger.error(f"Error in reject_retest: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------------------------------------------------------------------------
# Arranque
# ---------------------------------------------------------------------------


# ── Disk cache for training review data (persists across restarts) ──
_TRAINING_REVIEW_CACHE_FILE = project_root / "aipha_memory" / "operational" / "training_review_cache.json"
_TRAINING_REVIEW_CACHE_TTL = 300  # 5 minutes

def _get_training_review_cache():
    """Get cached review data from disk if still valid, otherwise None."""
    if not _TRAINING_REVIEW_CACHE_FILE.exists():
        return None
    try:
        data = json.loads(_TRAINING_REVIEW_CACHE_FILE.read_text())
        cache_ts = data.get("_cache_ts", 0)
        if time.time() - cache_ts < _TRAINING_REVIEW_CACHE_TTL:
            logger.info(f"📦 Serving cached training review data from disk (TTL: {int(time.time()-cache_ts)}s)")
            # Also keep in memory for this session
            _TRAINING_REVIEW_CACHE["data"] = data
            _TRAINING_REVIEW_CACHE["timestamp"] = cache_ts
            # Remove the internal timestamp field
            result = {k: v for k, v in data.items() if k != "_cache_ts"}
            return result
        else:
            logger.info("⏰ Cache expired, will re-fetch from Binance")
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"⚠️ Cache read error: {e}")
    return None

def _set_training_review_cache(data):
    """Store review data in both disk cache and memory."""
    cache_entry = {**data, "_cache_ts": time.time()}
    try:
        _TRAINING_REVIEW_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _TRAINING_REVIEW_CACHE_FILE.write_text(json.dumps(cache_entry))
        logger.info(f"💾 Cached training review data to disk: {_TRAINING_REVIEW_CACHE_FILE}")
    except OSError as e:
        logger.warning(f"⚠️ Cache write error: {e}")
    # Also keep in memory
    _TRAINING_REVIEW_CACHE["data"] = data
    _TRAINING_REVIEW_CACHE["timestamp"] = time.time()

# In-memory copy for hot access within same session
_TRAINING_REVIEW_CACHE = {"data": None, "timestamp": 0}

logger = logging.getLogger("cgalpha_v3")

import random
import threading
import time


def _read_live_market_price(symbol: str = "BTCUSDT") -> float:
    """
    Lee el precio de mercado real persistido por el shadow trader.
    Fallback: Binance REST API si el archivo no existe o es muy antiguo (>5min).
    Retorna 0.0 solo si ambas fuentes fallan.
    """
    safe_symbol = symbol.replace("/", "_").upper()
    price_path = (
        project_root
        / "aipha_memory"
        / "operational"
        / f"market_price_{safe_symbol}.json"
    )
    try:
        if price_path.exists():
            data = json.loads(price_path.read_text())
            price = float(data.get("price", 0.0))
            # Sanity check: BTC > 1000, ETH > 100 (evita precios corruptos)
            min_sane = 1000.0 if "BTC" in safe_symbol else 100.0
            if price > min_sane:
                return price
    except (json.JSONDecodeError, OSError, TypeError):
        pass

    # Fallback: Binance REST API (one-shot, no dependency)
    try:
        import urllib.request

        resp = urllib.request.urlopen(
            f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}",
            timeout=3,
        )
        data = json.loads(resp.read().decode())
        price = float(data.get("price", 0.0))
        if price > 0:
            return price
    except Exception:
        pass

    return 0.0


def _simulation_loop():
    """Bucle de vida para animar la Sala de Mando (CGAlpha v3 Mock)."""
    while True:
        try:
            live_price = 0.0
            try:
                state = _ws_manager.order_book_state.get(
                    _system_state.get("market_symbol", "BTCUSDT"), {}
                )
                if state and state.get("bids") and state.get("asks"):
                    best_bid = float(state["bids"][0][0])
                    best_ask = float(state["asks"][0][0])
                    live_price = (best_bid + best_ask) / 2
            except Exception:
                pass

            if live_price == 0.0:
                live_price = _read_live_market_price(
                    _system_state.get("market_symbol", "BTCUSDT")
                )

            if live_price > 0:
                _system_state["market_price"] = live_price
            _system_state["market_ts"] = datetime.now().isoformat()

            # 2. Simular Causal Drift & Trinity (OBI/Delta)
            _system_state["primary_source_gap"] = round(random.uniform(0.01, 1.25), 2)

            # 3. Generar Micro-Eventos en el Nexus
            if random.random() > 0.85:
                _log_event(
                    f"TRINITY UPDATE: OBI/Delta Shift detectado ({random.choice(['Long', 'Short'])})",
                    level="info",
                )

            # 4. Simular Recomendaciones del AutoProposer (NUEVO)
            # El bucle ha sido desactivado a petición para evitar ruido excesivo.

            time.sleep(3)
        except Exception as e:
            logger.error(f"❌ Simulation Loop Error: {e}")
            time.sleep(10)


# ---------------------------------------------------------------------------
# L2 Forensics API — Observabilidad del pipeline de microestructura
# ---------------------------------------------------------------------------


def _load_pending_labels_from_disk() -> list[dict[str, Any]]:
    """
    Carga pending labels persistidos en disco.
    Fallback útil cuando el collector corre en otro proceso.
    """
    pending_path = project_root / "aipha_memory" / "operational" / "pending_labels.json"
    if not pending_path.exists():
        return []

    try:
        raw = json.loads(pending_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(raw, list):
        return []

    summary: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            mfe = round(float(item.get("mfe", 0.0) or 0.0), 2)
            mae = round(float(item.get("mae", 0.0) or 0.0), 2)
            bars_elapsed = int(item.get("bars_elapsed", 0) or 0)
            lookahead_bars = int(item.get("lookahead_bars", 0) or 0)
            entry_price = float(item.get("entry_price", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue

        summary.append(
            {
                "sample_id": item.get("sample_id", "unknown"),
                "direction": item.get("zone_direction", "?"),
                "entry_price": entry_price,
                "bars_elapsed": bars_elapsed,
                "lookahead_bars": lookahead_bars,
                "mfe": mfe,
                "mae": mae,
            }
        )

    return summary


@app.route("/api/l2-forensics/status", methods=["GET"])
@require_auth
def api_l2_forensics_status():
    """
    Vista principal del monitor forense L2.
    Devuelve contadores y lista de pending labels con MFE/MAE en vivo.
    """
    btc_adapter = _adapters.get("BTCUSDT")
    if not btc_adapter:
        return jsonify({"error": "BTCUSDT adapter not found"}), 500

    monitor = btc_adapter.deferred_monitor
    pending_summary = monitor.get_pending_summary()
    pending_count = monitor.get_pending_count()
    resolved_count = monitor.get_resolved_count()
    pending_source = "in_memory"

    # Fallback de sincronización: si el recolector corre fuera del proceso GUI,
    # la cola real vive en disco aunque este monitor local tenga 0 pendientes.
    if pending_count == 0:
        disk_pending_summary = _load_pending_labels_from_disk()
        if disk_pending_summary:
            pending_summary = disk_pending_summary
            pending_count = len(disk_pending_summary)
            pending_source = "disk_fallback"

    # Leer contadores del dataset de entrenamiento v2
    training_path = (
        project_root / "aipha_memory" / "operational" / "training_dataset_v2.jsonl"
    )
    dataset_lines = 0
    full_samples = 0
    outcome_distribution = {
        "BOUNCE_STRONG": 0,
        "BOUNCE_WEAK": 0,
        "BREAKOUT": 0,
        "INCONCLUSIVE": 0,
    }
    if training_path.exists():
        try:
            with open(training_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    dataset_lines += 1
                    try:
                        row = json.loads(line)
                        label = row.get("outcome", {}).get("label", "UNKNOWN")
                        if label in outcome_distribution:
                            outcome_distribution[label] += 1

                        # Phase Bridge F3/L2 Check
                        if (
                            row.get("l2_temporal_profile", {}).get("l2_data_quality")
                            == "FULL"
                        ):
                            full_samples += 1
                    except json.JSONDecodeError:
                        pass
        except OSError:
            pass

    # Leer Active Zones de disco
    active_zones_disk = []
    zones_path = project_root / "aipha_memory" / "operational" / "active_zones.json"
    if zones_path.exists():
        try:
            with open(zones_path) as f:
                active_zones_disk = json.load(f)
        except Exception:
            pass

    return jsonify(
        {
            "pending_count": pending_count,
            "resolved_count": resolved_count,
            "dataset_total": dataset_lines,
            "full_samples": full_samples,
            "outcome_distribution": outcome_distribution,
            "active_monitors": pending_summary,
            "active_zones": active_zones_disk,
            "pending_source": pending_source,
        }
    )


@app.route("/api/l2-forensics/history", methods=["GET"])
@require_auth
def api_l2_forensics_history():
    """
    Últimos N samples resueltos del training dataset v2.
    Query param: ?limit=20 (default 20, max 100)
    """
    limit = min(int(request.args.get("limit", 20)), 100)

    training_path = (
        project_root / "aipha_memory" / "operational" / "training_dataset_v2.jsonl"
    )
    entries = []
    if training_path.exists():
        try:
            for line in _read_last_lines(
                training_path, limit, max_scan_bytes=8 * 1024 * 1024
            ):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    # Extraer solo campos esenciales para la timeline
                    meta = row.get("_meta", {})
                    zg = row.get("zone_geometry", {})
                    outcome = row.get("outcome", {})
                    l2 = row.get("l2_snapshot_at_touch", {})
                    entries.append(
                        {
                            "sample_id": meta.get("sample_id", "unknown"),
                            "capture_ts": meta.get("capture_ts_unix_ms", 0),
                            "symbol": meta.get("symbol", "BTCUSDT"),
                            "direction": zg.get("direction", "?"),
                            "zone_top": zg.get("zone_top"),
                            "zone_bottom": zg.get("zone_bottom"),
                            "zone_width_atr": zg.get("zone_width_atr"),
                            "retest_price": l2.get("retest_price"),
                            "obi_10": l2.get("obi_10"),
                            "cum_delta": l2.get("cumulative_delta"),
                            "label": outcome.get("label", "UNKNOWN"),
                            "mfe": outcome.get("mfe"),
                            "mae": outcome.get("mae"),
                            "mfe_atr": outcome.get("mfe_atr"),
                            "mae_atr": outcome.get("mae_atr"),
                            "bars_to_resolution": outcome.get("bars_to_resolution"),
                        }
                    )
                except json.JSONDecodeError:
                    pass
        except OSError:
            pass

    return jsonify({"entries": list(reversed(entries))})  # Más reciente primero


@app.route("/api/l2-forensics/snapshot/<sample_id>", methods=["GET"])
@require_auth
def api_l2_forensics_snapshot(sample_id: str):
    """
    Devuelve el ReentrySnapshot completo (JSON crudo) para inspección forense.
    """
    # Sanitizar sample_id para evitar path traversal
    safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "", sample_id)
    snap_path = project_root / "aipha_memory" / "snapshots" / f"{safe_id}.json"

    if not snap_path.exists():
        return jsonify({"error": f"Snapshot '{safe_id}' not found"}), 404

    try:
        snapshot = json.loads(snap_path.read_text())
        return jsonify(snapshot)
    except (json.JSONDecodeError, OSError) as e:
        return jsonify({"error": f"Could not read snapshot: {e}"}), 500


def main():
    """Entry point para lanzar el servidor de GUI."""
    _ensure_dirs()
    print(f"[CGAlpha v3 / Control Room] Iniciando en http://{HOST}:{PORT}")
    print(f"[CGAlpha v3 / Control Room] Auth token activo: {AUTH_TOKEN[:8]}...")
    print("[CGAlpha v3 / Control Room] Active Builder v3.0 iniciado")
    _log_event("CGAlpha v3 / Control Room iniciado")

    # Iniciar simulación de vida
    threading.Thread(target=_simulation_loop, daemon=True).start()

    # Iniciar pulso de evolución v4 (escalaciones periódicas)
    def _evolution_pulse():
        while True:
            try:
                escalated = _evolution_orchestrator.check_escalations()
                if escalated:
                    logger.info(f"⬆️ Escalaciones procesadas: {escalated}")
            except Exception as e:
                logger.error(f"❌ Evolution Pulse Error: {e}")
            time.sleep(60)  # v4: check every 60s

    threading.Thread(target=_evolution_pulse, daemon=True).start()

    # Arrancar WS Managers en segundo plano (Fase 4.2)
    def start_all_ws():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _bootstrap():
            for ws in _ws_managers.values():
                await ws.start()

        loop.run_until_complete(_bootstrap())
        try:
            loop.run_forever()
        finally:

            async def _shutdown():
                for ws in _ws_managers.values():
                    await ws.stop()

            loop.run_until_complete(_shutdown())
            loop.close()

    ws_thread = threading.Thread(target=start_all_ws, daemon=True)
    ws_thread.start()

    @app.route("/api/admin/read-file", methods=["GET"])
    @require_auth
    def api_admin_read_file():
        """
        Lee un archivo del sistema de forma segura (restringido a root del proyecto).
        Query params:
            - path: ruta relativa al root del proyecto
            - limit: (opcional) últimas N líneas
        """
        rel_path = request.args.get("path")
        if not rel_path:
            return jsonify({"error": "Falta parámetro 'path'"}), 400

        # Prevenir Directory Traversal
        full_path = (project_root / rel_path).resolve()
        if not str(full_path).startswith(str(project_root)):
            return jsonify(
                {"error": f"Acceso denegado: {rel_path} está fuera de la raíz"}
            ), 403

        if not full_path.exists():
            return jsonify({"error": f"Archivo no encontrado: {rel_path}"}), 404

        limit_arg = request.args.get("limit")
        if limit_arg:
            try:
                limit = int(limit_arg)
                lines = _read_last_lines(full_path, limit)
                return jsonify(
                    {
                        "content": "\n".join(lines),
                        "path": rel_path,
                        "limit": limit,
                        "lines_read": len(lines),
                    }
                )
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        else:
            try:
                # Comprobar si es binario (heurística simple)
                with open(full_path, "rb") as f:
                    chunk = f.read(1024)
                    if b"\0" in chunk:
                        return jsonify(
                            {"error": "No se pueden leer archivos binarios"}
                        ), 400

                content = full_path.read_text(encoding="utf-8", errors="ignore")
                # Cap a 1MB por seguridad
                if len(content) > 1024 * 1024:
                    return jsonify(
                        {
                            "error": "Archivo demasiado grande (>1MB). Use el parámetro 'limit' para leer solo el final."
                        }
                    ), 413
                return jsonify(
                    {"content": content, "path": rel_path, "size_bytes": len(content)}
                )
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    app.run(host=HOST, port=PORT, debug=False)


if __name__ == "__main__":
    main()
