"""Chess engine entry point: a compiled alpha-beta search over a neural evaluation.

The work happens in compiled code, split across four modules:

- position.py: bitboards, legal moves, make and unmake, hashing, static exchange evaluation
- evaluation.py: the network (nnue_eval.py) with a score cache, and a hand-written fallback
- search.py: principal variation search with a transposition table kept across moves, move
  ordering by hash move, MVV-LVA with static exchange evaluation, killers and history,
  null-move, reverse futility, futility and late move pruning, late move reductions, internal
  iterative deepening, mate distance pruning, check extensions and a quiescence search

This file keeps what runs once a move: the clock, the opening book, iterative deepening with
aspiration windows, and a legal answer whatever goes wrong. python-chess reads the position and
checks the reply; the search itself never touches it.

Evaluation: a HalfKAv2_hm network trained from scratch (see nnue_eval.py). The hand-written
evaluation stays as the fallback if the weights file is missing.

Opening: book/hiarcs_ref.bin, the HIARCS Reference Book Lite (human games with both players rated
2550 or more up to 2009, no computer or correspondence games) converted from Arena's .abk to
Polyglot, keeping moves played in at least 3 games. Its most played move is played without search.
The rules allow opening tables only for positions whose move number is 20 or lower, so the book
holds no position past move 20 and is never consulted after it.
"""

import hashlib
import os
import time
from operator import attrgetter
from pathlib import Path

# numba reads this when it is first imported. At LLVM's -O2 the compiled search measured 12%
# faster than at numba's default -O3 (86-90k against 77-78k nodes a second, tools/bench.py at
# depth 7), with the same nodes and moves.
os.environ.setdefault("NUMBA_OPT", "2")

# Local testing only: with NNUE_CHESS_CACHE_DIR set, compiled code is cached between games (see
# CACHE in position.py). numba notices when a function's own file changes but not the files it
# calls, so the directory is named after a hash of every source file here and the opt level.
if cache_root := os.environ.get("NNUE_CHESS_CACHE_DIR"):
    sources = sorted(Path(__file__).resolve().parent.glob("*.py"))
    digest = hashlib.sha256(os.environ["NUMBA_OPT"].encode())
    for source in sources:
        digest.update(source.read_bytes())
    os.environ["NUMBA_CACHE_DIR"] = str(Path(cache_root).resolve() / digest.hexdigest()[:16])

import chess
import chess.polyglot
import numpy as np

import evaluation
import position
import search

CONTRACT_INCREMENT_MS = 500
RESERVE_MS = 250  # the referee's watchdog grace and pipe latency
PANIC_MS = 400
# A rated game ran 69 moves and ended with 10 s left, so the clock is spread wider than the
# ~30 moves a middlegame position suggests, and the increment is only mostly spent.
MOVE_DIVISOR = 34.0
INCREMENT_SHARE = 0.85
LOW_CLOCK_MS = 25_000
LOW_DIVISOR = 45.0

MAX_DEPTH = 64
ASPIRATION = 40
INF = search.INF
MATE_BOUND = search.MATE_BOUND

# The network replaces the hand-written evaluation whenever its weights ship beside this file.
NETWORK = evaluation.load_network()
print(f"evaluation: {'network' if NETWORK.loaded else 'classical (no weights file)'}", flush=True)


class Searcher:
    """Iterative deepening over the compiled search, with state that lasts the whole game."""

    def __init__(self) -> None:
        self.position = position.new_position()
        self.state = search.new_state()
        self.root_moves = np.zeros(position.MAX_MOVES, dtype=np.int32)
        self.nodes = 0

    def choose(self, board: chess.Board, optimum_ms: float, maximum_ms: float) -> chess.Move:
        started = time.perf_counter()
        position.load(self.position, board)
        moves = self.root_moves
        count = search.begin(self.position, self.state, maximum_ms / 1000.0, moves)
        self.nodes = 0
        if count == 1:
            return self.to_move(moves[0])

        best_move = moves[0]
        best_score = 0
        completed = 0
        depth = 1
        while depth <= MAX_DEPTH:
            alpha, beta = -INF, INF
            if depth >= 4:
                alpha, beta = best_score - ASPIRATION, best_score + ASPIRATION
            while True:
                score, index, stopped = search.search_root(
                    self.position, NETWORK, self.state, moves, count, depth, alpha, beta
                )
                if stopped or alpha < score < beta:
                    break
                if score <= alpha:
                    alpha = -INF
                else:
                    beta = INF
            if stopped:
                break
            best_move, best_score, completed = moves[index], score, depth
            # The best move so far is searched first next time; the rest keep their order.
            moves[1 : index + 1] = moves[:index].copy()
            moves[0] = best_move
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if abs(score) >= MATE_BOUND or elapsed_ms > optimum_ms * 0.6:
                break
            depth += 1

        self.nodes = int(search.node_count(self.state))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        # Unflushed output dies with the process at game end, taking the log with it.
        print(
            f"depth {completed} score {best_score:+d} nodes {self.nodes} time {elapsed_ms:.0f}ms",
            flush=True,
        )
        return self.to_move(best_move)

    def ordered_move(self, board: chess.Board) -> chess.Move:
        """The move the search would try first, without searching."""
        position.load(self.position, board)
        count = search.order_root(self.position, self.state, 0, self.root_moves)
        if count == 0:
            raise ValueError(f"no legal moves in {board.fen()}")
        return self.to_move(self.root_moves[0])

    @staticmethod
    def to_move(move: int) -> chess.Move:
        return chess.Move.from_uci(position.to_uci(move))


# --- time ----------------------------------------------------------------------------------


class Clock:
    """Per-move budgets. The increment is not in the interface, so it is measured."""

    def __init__(self) -> None:
        self.increment_ms = CONTRACT_INCREMENT_MS
        self.previous_left: int | None = None
        self.previous_spent_ms = 0.0

    def observe(self, time_left_ms: int) -> None:
        if self.previous_left is not None:
            estimate = time_left_ms - (self.previous_left - self.previous_spent_ms)
            if -50.0 <= estimate <= 5000.0:
                self.increment_ms = max(0, int(estimate) - 20)
        self.previous_left = time_left_ms

    def budget(self, time_left_ms: int) -> tuple[float, float] | None:
        remaining = time_left_ms - RESERVE_MS
        if remaining < PANIC_MS:
            return None
        divisor = MOVE_DIVISOR if remaining > LOW_CLOCK_MS else LOW_DIVISOR
        earned = min(self.increment_ms * INCREMENT_SHARE, remaining / 20.0)
        optimum = remaining / divisor + earned
        maximum = min(optimum * 2.6, remaining / 5.0)
        return optimum, maximum

    def finish(self, started: float) -> None:
        self.previous_spent_ms = (time.perf_counter() - started) * 1000.0


# --- start-up ------------------------------------------------------------------------------

# Positions that reach every part of the search: castling, en passant, promotions, checks.
WARM_UP_FENS = (
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "8/1P4k1/8/8/8/8/6K1/8 w - - 0 1",
)


def _warm_up() -> None:
    """Finish compiling during the 90 s import budget, not on the clock.

    The search functions have fixed signatures and compiled when their modules were imported;
    the numba classes' constructors compile the first time one is built. A throwaway Searcher
    plays each warm-up position through, so the game's own table, killers and history start
    empty.
    """
    started = time.perf_counter()
    searcher = Searcher()
    for fen in WARM_UP_FENS:
        board = chess.Board(fen)
        searcher.choose(board, 150.0, 150.0)
        searcher.ordered_move(board)
    print(f"compiled in {time.perf_counter() - started:.1f} s", flush=True)


_warm_up()
SEARCHER = Searcher()
CLOCK = Clock()

# --- opening book --------------------------------------------------------------------------

BOOK_PATH = Path(__file__).resolve().parent / "book" / "hiarcs_ref.bin"
# "The opening is a position whose move number is 20 or lower." A table that answers anything
# later is a middlegame table, which the rules count as an engine.
LAST_BOOK_MOVE = 20


def _open_book() -> chess.polyglot.MemoryMappedReader | None:
    try:
        return chess.polyglot.open_reader(BOOK_PATH)
    except OSError:
        return None


BOOK = _open_book()
print(f"opening book: {'hiarcs_ref' if BOOK else 'none (no book file)'}", flush=True)


def _book_move(board: chess.Board) -> chess.Move | None:
    if BOOK is None or board.fullmove_number > LAST_BOOK_MOVE:
        return None
    try:
        entry = max(BOOK.find_all(board), key=attrgetter("weight"), default=None)
    except Exception as error:  # a damaged book must never cost a move, search still has it
        print(f"book failed, searching: {error!r}", flush=True)
        return None
    return None if entry is None else entry.move


# --- the move ------------------------------------------------------------------------------


def _fallback(board: chess.Board) -> chess.Move:
    """The first move in search order, or python-chess's first legal move if even that fails."""
    try:
        return SEARCHER.ordered_move(board)
    except Exception as error:
        print(f"ordering failed, playing the first legal move: {error!r}", flush=True)
        return next(iter(board.legal_moves))


def _pick(board: chess.Board, time_left_ms: int) -> chess.Move:
    book = _book_move(board)
    if book is not None:
        return book
    budget = CLOCK.budget(time_left_ms)
    if budget is None:
        return _fallback(board)
    optimum_ms, maximum_ms = budget
    return SEARCHER.choose(board, optimum_ms, maximum_ms)


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation for the side to move in fen."""
    started = time.perf_counter()
    CLOCK.observe(time_left_ms)
    move: chess.Move | None = None
    try:
        move = _pick(chess.Board(fen), time_left_ms)
    except Exception as error:  # any failure here must still produce a legal move
        print(f"search failed, falling back: {error!r}", flush=True)
    # Judged against the position we were handed, never against the board the search walked.
    position_now = chess.Board(fen)
    if move is None or move not in position_now.legal_moves:
        move = _fallback(position_now)
    CLOCK.finish(started)
    return move.uci()
