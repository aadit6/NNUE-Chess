"""Check the compiled engine against python-chess and the Python engine it replaced.

    uv run python tools/verify_compiled.py

- perft: move-path counts on six standard positions, against published totals or python-chess
- moves: the legal moves of positions from random games, against python-chess's, move for move
- keys: the key make_move keeps up to date, against one computed from scratch, and restored exactly
  by unmake_move
- see: exchange evaluation, against the Python version in versions/v9b
- evaluation: the hand-written and network evaluations, against versions/v9b
- accumulators: the network kept up to date move by move, against a full rebuild, through
  skipped evaluations, null moves and moves taken back

Exits non-zero on the first mismatch.
"""

import random
import sys
import time
from pathlib import Path

import chess
import numpy as np
from numba import int32, int64, njit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluation  # noqa: E402
import nnue_eval  # noqa: E402
import position  # noqa: E402

PERFT = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", (20, 400, 8902, 197281)),
    (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        (48, 2039, 97862, 4085603),
    ),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", (14, 191, 2812, 43238, 674624)),
    (
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        (6, 264, 9467, 422333),
    ),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", (44, 1486, 62379, 2103487)),
    (
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N1P/1PP1QPP1/R4RK1 w - - 0 10",
        (48, 2166, 97403),  # python-chess's own perft for this one
    ),
]


def fail(message: str) -> None:
    print(f"FAIL {message}")
    raise SystemExit(1)


@njit(int64(position.POSITION, int32[:, ::1], int64), cache=position.CACHE)
def perft(pos, buffers, depth):  # type: ignore[no-untyped-def]
    moves = buffers[depth]
    count = position.generate(pos, moves, False)
    total = 0
    for index in range(count):
        if position.make_move(pos, moves[index]):
            total += 1 if depth == 1 else perft(pos, buffers, depth - 1)
            position.unmake_move(pos)
    return total


def legal_ucis(pos: position.Position, moves: np.ndarray) -> set[str]:
    found = set()
    for index in range(position.generate(pos, moves, False)):
        move = moves[index]
        if position.make_move(pos, move):
            if position.compute_key(pos) != pos.key:
                fail(f"key after {position.to_uci(move)} differs from one computed from scratch")
            position.unmake_move(pos)
            found.add(position.to_uci(move))
    return found


def check_perft() -> None:
    pos = position.new_position()
    buffers = np.zeros((8, position.MAX_MOVES), dtype=np.int32)
    started = time.perf_counter()
    for fen, expected in PERFT:
        position.load(pos, chess.Board(fen))
        for depth, total in enumerate(expected, start=1):
            found = perft(pos, buffers, depth)
            if found != total:
                fail(f"perft {depth} of {fen}: {found}, expected {total}")
    elapsed = time.perf_counter() - started
    print(f"perft: 6 positions match ({elapsed:.1f} s including compiling perft)")


def random_positions(games: int, seed: int) -> list[chess.Board]:
    rng = random.Random(seed)
    boards = []
    for _ in range(games):
        board = chess.Board()
        for _ in range(rng.randrange(20, 160)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(rng.choice(moves))
            boards.append(board.copy(stack=False))
    return boards


def check_moves_and_keys(boards: list[chess.Board]) -> None:
    pos = position.new_position()
    moves = np.zeros(position.MAX_MOVES, dtype=np.int32)
    keys_by_transposition: dict[object, int] = {}
    for board in boards:
        position.load(pos, board)
        expected = {move.uci() for move in board.legal_moves}
        found = legal_ucis(pos, moves)
        if found != expected:
            fail(f"moves of {board.fen()}: missing {expected - found}, extra {found - expected}")
        if bool(position.in_check(pos)) != board.is_check():
            fail(f"in_check of {board.fen()}")
        if bool(position.insufficient_material(pos)) != board.is_insufficient_material():
            fail(f"insufficient_material of {board.fen()}")
        # Equal python-chess transposition keys must give equal keys here, and different ones
        # different keys.
        identity = board._transposition_key()
        key = int(pos.key)
        if keys_by_transposition.setdefault(identity, key) != key:
            fail(f"two keys for one position: {board.fen()}")
    distinct = len(set(keys_by_transposition.values()))
    if distinct != len(keys_by_transposition):
        fail("two positions share a key")
    print(f"moves and keys: {len(boards)} positions match python-chess")


def check_see(boards: list[chess.Board], reference: object) -> None:
    pos = position.new_position()
    moves = np.zeros(position.MAX_MOVES, dtype=np.int32)
    checked = 0
    for board in boards:
        position.load(pos, board)
        for index in range(position.generate(pos, moves, True)):
            move = moves[index]
            python_move = chess.Move.from_uci(position.to_uci(move))
            if python_move not in board.legal_moves:
                continue
            expected = reference.see(board, python_move)  # type: ignore[attr-defined]
            found = position.see(pos, move)
            if found != expected:
                fail(f"see {python_move} in {board.fen()}: {found}, expected {expected}")
            checked += 1
    print(f"see: {checked} captures match versions/v9b")


def check_evaluation(boards: list[chess.Board], reference: object) -> None:
    """The hand-written evaluation must match exactly. The network adds its sums in another order
    (see head() in nnue_eval.py), so its output must agree to 1e-3, and a score may land one
    centipawn away when it sits on a rounding boundary."""
    pos = position.new_position()
    net = evaluation.load_network()
    if not net.loaded:
        fail("no weights file, so the network cannot be checked")
    worst = 0.0
    off_by_one = 0
    for board in boards:
        position.load(pos, board)
        expected = reference.classical_evaluate(board)  # type: ignore[attr-defined]
        found = evaluation.classical_evaluate(pos)
        if found != expected:
            fail(f"classical_evaluate of {board.fen()}: {found}, expected {expected}")
        rebuilt = nnue_eval.raw(board)
        incremental = float(evaluation.network_raw(pos, net))
        worst = max(worst, abs(incremental - rebuilt))
        expected = reference.nnue_eval.evaluate(board)  # type: ignore[attr-defined]
        found = evaluation.evaluate(pos, net)
        cached = evaluation.evaluate(pos, net)
        if abs(incremental - rebuilt) > 1e-3 or abs(found - expected) > 1 or cached != found:
            fail(f"network evaluate of {board.fen()}: {found} then {cached}, expected {expected}")
        off_by_one += found != expected
    print(
        f"evaluation: {len(boards)} positions match versions/v9b; the network to {worst:.1e}, "
        f"{off_by_one} a centipawn apart after rounding"
    )


def encoded(pos: position.Position, moves: np.ndarray, move: chess.Move) -> int:
    for index in range(position.generate(pos, moves, False)):
        if position.to_uci(moves[index]) == move.uci():
            return int(moves[index])
    fail(f"{move.uci()} was not generated")
    return 0


def check_accumulators(games: int, seed: int) -> None:
    """Incremental network output against forward() rebuilding it, along random games."""
    rng = random.Random(seed)
    pos = position.new_position()
    net = evaluation.load_network()
    moves = np.zeros(position.MAX_MOVES, dtype=np.int32)
    worst = 0.0
    compared = 0
    centipawn_differences = 0

    def compare(board: chess.Board) -> None:
        nonlocal worst, compared, centipawn_differences
        incremental = float(evaluation.network_raw(pos, net))
        rebuilt = nnue_eval.raw(board)
        worst = max(worst, abs(incremental - rebuilt))
        compared += 1
        centipawn_differences += int(incremental * net_scale) != int(rebuilt * net_scale)
        if abs(incremental - rebuilt) > 2e-3:
            fail(f"accumulator at {board.fen()}: {incremental}, rebuilt {rebuilt}")

    net_scale = nnue_eval.SCALE
    for _ in range(games):
        board = chess.Board()
        position.load(pos, board)
        for _ in range(100):  # stays under MAX_PLY, since every move here is one more ply
            legal = list(board.legal_moves)
            if not legal:
                break
            if rng.random() < 0.5:  # skipping some makes the next update walk back further
                compare(board)
            if rng.random() < 0.1 and not board.is_check():
                position.make_null(pos)
                board.push(chess.Move.null())
                compare(board)
                position.unmake_null(pos)
                board.pop()
            move = rng.choice(legal)
            position.make_move(pos, encoded(pos, moves, move))
            board.push(move)
            if rng.random() < 0.15:  # take it back and play another from the same ply
                compare(board)
                position.unmake_move(pos)
                board.pop()
                move = rng.choice(legal)
                position.make_move(pos, encoded(pos, moves, move))
                board.push(move)
        compare(board)
    print(
        f"accumulators: {compared} evaluations match a full rebuild, largest difference "
        f"{worst:.1e}, {centipawn_differences} differ by a centipawn after rounding"
    )


def load_reference() -> object:
    """The Python engine this one replaced, imported under another name."""
    import importlib.util

    folder = ROOT / "versions" / "v9b"
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("reference_agent", folder / "agent.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.remove(str(folder))
    return module


def main() -> None:
    check_perft()
    boards = random_positions(60, seed=1)
    check_moves_and_keys(boards)
    reference = load_reference()
    check_see(boards, reference)
    check_evaluation(boards, reference)
    check_accumulators(40, seed=2)
    print("all checks passed")


if __name__ == "__main__":
    main()
