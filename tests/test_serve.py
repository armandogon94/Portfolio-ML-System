"""Host-port guardrails for the native FastAPI launcher."""

from __future__ import annotations

import sys

import pytest

from scripts import serve

DOCUMENTED_RESERVED_PORTS = {80, 443, 3000, 5000, 5432, 6379, 7000, 8000, 11434}


def test_refused_port_set_matches_the_documented_reserved_ports():
    assert serve.REFUSED_PORTS == DOCUMENTED_RESERVED_PORTS


@pytest.mark.parametrize("port", sorted(DOCUMENTED_RESERVED_PORTS))
def test_serve_refuses_every_documented_reserved_port(monkeypatch, capsys, port):
    monkeypatch.setattr(sys, "argv", ["serve.py", "--port", str(port)])

    with pytest.raises(SystemExit) as excinfo:
        serve.main()

    assert excinfo.value.code == 2
    error = capsys.readouterr().err
    assert f"Port {port} is reserved" in error
    if port in {80, 443}:
        assert "root" in error
        assert "reverse proxy" in error
