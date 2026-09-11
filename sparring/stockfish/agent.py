"""A sparring partner, not a submission. Speaks the harness protocol, plays through UCI.

This exists so local results have a reference point outside this repo. It shells out to an
installed engine binary, which the competition forbids inside a submission, so it lives in a
subdirectory that `make zip` never walks: the packager globs `*.py` beside agent.py and the
packages those import, and nothing here is imported by anything there.

Strength is capped through UCI_LimitStrength so the games are informative rather than a
100-0 formality. Set the target with SPAR_ELO.
"""

import atexit
import os
import shutil
import subprocess
import sys

BINARY = os.environ.get("SPAR_ENGINE", "stockfish")
TARGET_ELO = int(os.environ.get("SPAR_ELO", "1500"))
MOVE_DIVISOR = int(os.environ.get("SPAR_DIVISOR", "30"))
RESERVE_MS = 300
HANDSHAKE_LINES = 400


def _launch() -> subprocess.Popen[str]:
    path = shutil.which(BINARY)
    if path is None:
        raise RuntimeError(f"no engine binary named {BINARY!r} on PATH")
    return subprocess.Popen(
        [path],
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
_send("setoption name UCI_LimitStrength value true")
_send(f"setoption name UCI_Elo value {TARGET_ELO}")
_send("isready")
_await("readyok")
print(f"sparring: {BINARY} capped at UCI_Elo {TARGET_ELO}", file=sys.stderr, flush=True)


def get_move(fen: str, time_left_ms: int) -> str:
    budget = max(20, (time_left_ms - RESERVE_MS) // MOVE_DIVISOR)
    _send(f"position fen {fen}")
    _send(f"go movetime {budget}")
    assert ENGINE.stdout is not None
    while True:
        line = ENGINE.stdout.readline()
        if not line:
            raise RuntimeError("engine closed while thinking")
        if line.startswith("bestmove"):
            return line.split()[1]
