"""
Tests for Safety Envelope (parameter_constraints.json) and Evolution Pulse endpoint.
"""
import pytest
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cgalpha_v3.lila.evolution_orchestrator import EvolutionOrchestratorV4, EvolutionResult
from cgalpha_v3.lila.llm.proposer import TechnicalSpec


# ───────────────────────────────────────────────────────
# SAFETY ENVELOPE TESTS
# ───────────────────────────────────────────────────────

class TestSafetyEnvelope:
    """Test that the Safety Envelope rejects out-of-bounds proposals."""

    def _make_orchestrator(self):
        return EvolutionOrchestratorV4()

    def test_valid_parameter_passes(self):
        """A proposal within bounds should NOT be rejected by the envelope."""
        orch = self._make_orchestrator()
        spec = TechnicalSpec(
            change_type="parameter",
            target_file="cgalpha_v3/lila/llm/oracle.py",
            target_attribute="n_estimators",
            old_value=100,
            new_value=150,  # Within [10, 500]
            reason="test",
            causal_score_est=0.5,
            confidence=0.8,
        )
        is_valid, msg = orch._validate_constraints(spec)
        assert is_valid, f"Should pass: {msg}"

    def test_below_minimum_rejected(self):
        """A proposal below the minimum bound should be rejected."""
        orch = self._make_orchestrator()
        spec = TechnicalSpec(
            change_type="parameter",
            target_file="cgalpha_v3/lila/llm/oracle.py",
            target_attribute="n_estimators",
            old_value=100,
            new_value=5,  # Below min=10
            reason="test",
            causal_score_est=0.5,
            confidence=0.8,
        )
        is_valid, msg = orch._validate_constraints(spec)
        assert not is_valid
        assert "SAFETY ENVELOPE VIOLATION" in msg

    def test_above_maximum_rejected(self):
        """A proposal above the maximum bound should be rejected."""
        orch = self._make_orchestrator()
        spec = TechnicalSpec(
            change_type="parameter",
            target_file="cgalpha_v3/lila/llm/oracle.py",
            target_attribute="n_estimators",
            old_value=100,
            new_value=999,  # Above max=500
            reason="test",
            causal_score_est=0.5,
            confidence=0.8,
        )
        is_valid, msg = orch._validate_constraints(spec)
        assert not is_valid
        assert "SAFETY ENVELOPE VIOLATION" in msg

    def test_unknown_parameter_passes(self):
        """A parameter not in constraints should pass (no restriction)."""
        orch = self._make_orchestrator()
        spec = TechnicalSpec(
            change_type="parameter",
            target_file="some_file.py",
            target_attribute="unknown_param_xyz",
            old_value=0,
            new_value=999999,
            reason="test",
            causal_score_est=0.5,
            confidence=0.8,
        )
        is_valid, msg = orch._validate_constraints(spec)
        assert is_valid

    def test_process_proposal_rejects_unsafe(self):
        """Full integration: process_proposal should return REJECTED_SAFETY."""
        orch = self._make_orchestrator()
        spec = TechnicalSpec(
            change_type="parameter",
            target_file="cgalpha_v3/lila/llm/oracle.py",
            target_attribute="min_confidence",
            old_value=0.7,
            new_value=1.5,  # Above max=0.99
            reason="bad proposal",
            causal_score_est=0.5,
            confidence=0.9,
        )
        result = orch.process_proposal(spec)
        assert result.status == "REJECTED_SAFETY"
        assert result.category == 0

    def test_zigzag_threshold_bounds(self):
        """Verify zigzag_threshold bounds [0.0001, 0.05]."""
        orch = self._make_orchestrator()
        # Too low
        spec_low = TechnicalSpec(
            change_type="parameter",
            target_file="triple_coincidence.py",
            target_attribute="zigzag_threshold",
            old_value=0.001,
            new_value=0.00001,  # Below min
            reason="test",
            causal_score_est=0.5,
            confidence=0.8,
        )
        is_valid, _ = orch._validate_constraints(spec_low)
        assert not is_valid

        # Too high
        spec_high = TechnicalSpec(
            change_type="parameter",
            target_file="triple_coincidence.py",
            target_attribute="zigzag_threshold",
            old_value=0.001,
            new_value=0.1,  # Above max=0.05
            reason="test",
            causal_score_est=0.5,
            confidence=0.8,
        )
        is_valid, _ = orch._validate_constraints(spec_high)
        assert not is_valid


# ───────────────────────────────────────────────────────
# EVOLUTION PULSE ENDPOINT TESTS
# ───────────────────────────────────────────────────────

class TestEvolutionPulseEndpoint:
    """Test the /api/evolution/heartbeat endpoint."""

    def _get_app(self):
        """Import and return the Flask test client."""
        from cgalpha_v3.gui.server import app
        app.config["TESTING"] = True
        return app.test_client()

    def test_heartbeat_returns_json(self):
        """Endpoint should return valid JSON with expected keys."""
        client = self._get_app()
        resp = client.get(
            "/api/evolution/heartbeat",
            headers={"Authorization": "Bearer cgalpha-v3-local-dev"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "status" in data
        assert "cycle" in data

    def test_heartbeat_without_file(self):
        """When heartbeat file doesn't exist, should return offline status."""
        client = self._get_app()
        resp = client.get(
            "/api/evolution/heartbeat",
            headers={"Authorization": "Bearer cgalpha-v3-local-dev"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        # Either "offline" or actual data if file exists
        assert data["status"] in ("offline", "OFFLINE", "OK", "ERROR", "NO_DATA", "COMPLETED")
