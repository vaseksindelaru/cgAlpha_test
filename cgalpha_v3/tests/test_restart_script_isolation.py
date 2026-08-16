"""Regression tests for SDT-safe restart process isolation."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESTART_SCRIPT = PROJECT_ROOT / "restart-cgalpha-test"


def _script() -> str:
    return RESTART_SCRIPT.read_text(encoding="utf-8")


def test_restart_stops_only_processes_owned_by_test_repo():
    script = _script()

    assert "stop_repo_processes" in script
    assert 'readlink "/proc/$pid/cwd"' in script
    assert '"$cwd" = "$REPO_DIR"' in script

    forbidden_global_kills = (
        "pkill -9 -f harness_watchdog.py",
        "pkill -9 -f cgalpha_v3.gui.server",
        "pkill -9 -f launch_shadow_live.py",
        'pkill -9 -f "CGAlpha_0.0.1-Aipha_0.0.3',
    )
    for command in forbidden_global_kills:
        assert command not in script


def test_restart_never_force_kills_an_unowned_port_listener():
    script = _script()

    assert "fuser -k 8080/tcp" not in script
    assert "Port 8080 is occupied after stopping TEST processes" in script


def test_restart_does_not_embed_r2_credentials():
    script = _script()

    assert "R2_ACCESS_KEY_ID must be set" in script
    assert "R2_SECRET_ACCESS_KEY must be set" in script
    assert "d498edd635c3ee894993700637376906" not in script
    assert "f9aebd0b4173905721cd741c46895670e1aab42531f4eb76aa18f7e6a7b72549" not in script
