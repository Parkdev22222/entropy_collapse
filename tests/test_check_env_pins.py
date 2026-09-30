"""scripts/check_env_pins.py: vllm's declared-only opentelemetry range is an
accepted exception (ray 2.58 needs a newer one), and must never be "fixed"."""
import importlib.util
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cep", ROOT / "scripts" / "check_env_pins.py")
cep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cep)


def fake(monkeypatch, installed, reqs, enforced):
    def version(name):
        if name not in installed:
            raise cep.md.PackageNotFoundError(name)
        return installed[name]
    monkeypatch.setattr(cep.md, "version", version)
    monkeypatch.setattr(cep, "_requirements", lambda d: [Requirement(r) for r in reqs.get(d, [])])
    monkeypatch.setattr(cep, "_enforced_at_import", lambda h: enforced.get(h, set()))


def test_vllm_opentelemetry_is_noted_not_fixed(monkeypatch, capsys):
    fake(monkeypatch, {"vllm": "0.8.4", "ray": "2.58.0", "opentelemetry-sdk": "1.45.0"},
         {"vllm": ["opentelemetry-sdk<1.27.0,>=1.26.0"]}, {"vllm": set()})
    assert cep.main([]) == 3
    out = capsys.readouterr().out
    assert "NOTE" in out and "fix:" not in out


def test_an_enforced_or_other_violation_still_fails(monkeypatch, capsys):
    fake(monkeypatch, {"vllm": "0.8.4", "opentelemetry-sdk": "1.45.0",
                       "transformers": "4.57.6", "huggingface-hub": "1.30.0"},
         {"vllm": ["opentelemetry-sdk<1.27.0,>=1.26.0"],
          "transformers": ["huggingface-hub<1.0,>=0.34.0"]},
         {"vllm": set(), "transformers": {"huggingface-hub"}})
    assert cep.main([]) == 1
    out = capsys.readouterr().out
    assert '"huggingface-hub' in out and "opentelemetry" not in out.split("fix:")[1]


def test_if_vllm_ever_enforces_it_the_exception_lapses(monkeypatch):
    fake(monkeypatch, {"vllm": "0.9.0", "opentelemetry-sdk": "1.45.0"},
         {"vllm": ["opentelemetry-sdk<1.27.0"]}, {"vllm": {"opentelemetry-sdk"}})
    assert cep.main([]) == 1
