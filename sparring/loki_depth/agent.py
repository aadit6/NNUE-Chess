"""Sparring partner: Loki 3.0.0, an open-source C++ engine, spoken to over UCI.

Loki exposes no UCI_LimitStrength, so unlike the Stockfish partner this one only plays at full
strength. It runs as a separate process and never enters the submission: the packager walks
`*.py` beside agent.py and the packages those import, and nothing here is imported by any of it.
"""

import atexit
import os
import subprocess
import sys
from pathlib import Path

ENGINE_PATH = Path(__file__).resolve().parent.parent.parent / "engines" / "Loki3.0.0-x64.exe"
FIXED_DEPTH = int(os.environ.get("FIXED_DEPTH", "6"))
RESERVE_MS = 300
HANDSHAKE_LINES = 400


def _launch() -> subprocess.Popen[str]:
    if not ENGINE_PATH.is_file():
        raise RuntimeError(f"no engine binary at {ENGINE_PATH}")
    return subprocess.Popen(
        [str(ENGINE_PATH)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )


ENGINE = _launch()


def _send(line: str) -> None:
    assert ENGINE.stdin is not None
    ENGINE.stdin.write(line + "\n")
    ENGINE.stdin.flush()


def _await(token: str) -> None:
    assert ENGINE.stdout is not None
    for _ in range(HANDSHAKE_LINES):
        line = ENGINE.stdout.readline()
        if not line:
            raise RuntimeError("engine closed during handshake")
        if line.split(" ")[0].strip() == token:
            return
    raise RuntimeError(f"engine never answered {token}")


def _quit() -> None:
    if ENGINE.poll() is None:
        try:
            _send("quit")
            ENGINE.wait(timeout=3)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            ENGINE.kill()


atexit.register(_quit)

_send("uci")
_await("uciok")
_send("setoption name Threads value 1")
_send("setoption name Hash value 16")
_send("isready")
_await("readyok")
print(f"sparring: Loki 3.0.0 pinned to depth {FIXED_DEPTH}", file=sys.stderr, flush=True)


def get_move(fen: str, time_left_ms: int) -> str:
    _send(f"position fen {fen}")
    _send(f"go depth {FIXED_DEPTH}")
    assert ENGINE.stdout is not None
    while True:
        line = ENGINE.stdout.readline()
        if not line:
            raise RuntimeError("engine closed while thinking")
        if line.startswith("bestmove"):
            return line.split()[1]
