"""AI Chessathon submission: alpha-beta search over a neural evaluation.

Search: iterative deepening, principal variation search with aspiration windows, a
transposition table kept across moves, move ordering by TT move, MVV-LVA with static exchange
evaluation, killers and history, null-move, reverse futility, futility and late move pruning,
late move reductions, internal iterative deepening, mate distance pruning, check extensions and
a captures-only quiescence search.

Evaluation: a HalfKAv2_hm network trained from scratch (see nnue_eval.py). The hand-written
evaluation below stays as the fallback if the weights file is missing.
"""

import math
import time
from collections.abc import Hashable
from operator import itemgetter

import chess
import nnue_eval

CONTRACT_INCREMENT_MS = 500
RESERVE_MS = 250  # the referee's watchdog grace and pipe latency
PANIC_MS = 400
NODE_CHECK_MASK = 1023
# A rated game ran 69 moves and ended with 10 s left, so the clock is spread wider than the
# ~30 moves a middlegame position suggests, and the increment is only mostly spent.
MOVE_DIVISOR = 34.0
INCREMENT_SHARE = 0.85
LOW_CLOCK_MS = 25_000
LOW_DIVISOR = 45.0

MAX_DEPTH = 64
MAX_PLY = 96
INF = 40_000
MATE = 32_000
MATE_BOUND = 31_000
DRAW = 0
TT_MAX_ENTRIES = 1_000_000
EXACT, LOWER, UPPER = 0, 1, 2
ASPIRATION = 40
DELTA_MARGIN = 150
LMR_FROM_MOVE = 3
RFP_DEPTH = 6
RFP_MARGIN = 85
FUTILITY_DEPTH = 4
FUTILITY_MARGIN = 110
IID_DEPTH = 5
LMP_DEPTH = 5
LMP_COUNT = (0, 6, 10, 16, 24, 34)

# Reduce late quiet moves harder the deeper the node and the further down the move list, rather
# than by the flat one-or-two plies a hand-written rule gives.
LMR_TABLE = [
    [0 if depth == 0 or index == 0 else int(0.75 + math.log(depth) * math.log(index) / 2.25)
     for index in range(64)]
    for depth in range(64)
]

TTEntry = tuple[int, int, int, chess.Move | None]


class SearchTimeout(Exception):
    pass


# --- evaluation ---------------------------------------------------------------------------

PIECE_VALUES = (0, 100, 320, 330, 500, 900, 0)
ATTACKER_VALUES = (0, 100, 320, 330, 500, 900, 1000)
PHASE_MAX = 24

BISHOP_PAIR = 30
ROOK_OPEN_FILE = 20
ROOK_SEMI_OPEN_FILE = 10
DOUBLED_PAWN = 12
ISOLATED_PAWN = 12
PASSED_PAWN = (0, 10, 15, 25, 45, 70, 100, 0)
SHIELD_NEAR = 10
SHIELD_FAR = 5
TEMPO = 10

# Tables are written from White's point of view with rank 8 on the top row.
_PAWN = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]  # fmt: skip
_KNIGHT = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]  # fmt: skip
_BISHOP = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]  # fmt: skip
_ROOK = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0,
]  # fmt: skip
_QUEEN = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]  # fmt: skip
_KING_MIDDLEGAME = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
]  # fmt: skip
_KING_ENDGAME = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
]  # fmt: skip


def _tables(table: list[int], value: int) -> tuple[list[int], list[int]]:
    """Square-indexed (a1 = 0) tables for White and Black, with the piece value folded in."""
    white = [value + table[square ^ 56] for square in range(64)]
    black = [value + table[square] for square in range(64)]
    return white, black


PAWN_W, PAWN_B = _tables(_PAWN, PIECE_VALUES[chess.PAWN])
KNIGHT_W, KNIGHT_B = _tables(_KNIGHT, PIECE_VALUES[chess.KNIGHT])
BISHOP_W, BISHOP_B = _tables(_BISHOP, PIECE_VALUES[chess.BISHOP])
ROOK_W, ROOK_B = _tables(_ROOK, PIECE_VALUES[chess.ROOK])
QUEEN_W, QUEEN_B = _tables(_QUEEN, PIECE_VALUES[chess.QUEEN])
KING_MG_W, KING_MG_B = _tables(_KING_MIDDLEGAME, 0)
KING_EG_W, KING_EG_B = _tables(_KING_ENDGAME, 0)


def _masks() -> tuple[list[int], ...]:
    adjacent = [
        (chess.BB_FILES[f - 1] if f > 0 else 0) | (chess.BB_FILES[f + 1] if f < 7 else 0)
        for f in range(8)
    ]
    passed_w, passed_b = [0] * 64, [0] * 64
    near_w, far_w, near_b, far_b = [0] * 64, [0] * 64, [0] * 64, [0] * 64
    for square in range(64):
        file, rank = square & 7, square >> 3
        span = chess.BB_FILES[file] | adjacent[file]
        ahead_w = sum(chess.BB_RANKS[rank + 1 :])
        ahead_b = sum(chess.BB_RANKS[:rank])
        passed_w[square] = span & ahead_w
        passed_b[square] = span & ahead_b
        near_w[square] = span & (chess.BB_RANKS[rank + 1] if rank < 7 else 0)
        far_w[square] = span & (chess.BB_RANKS[rank + 2] if rank < 6 else 0)
        near_b[square] = span & (chess.BB_RANKS[rank - 1] if rank > 0 else 0)
        far_b[square] = span & (chess.BB_RANKS[rank - 2] if rank > 1 else 0)
    return adjacent, passed_w, passed_b, near_w, far_w, near_b, far_b


ADJACENT_FILES, PASSED_W, PASSED_B, SHIELD_NEAR_W, SHIELD_FAR_W, SHIELD_NEAR_B, SHIELD_FAR_B = (
    _masks()
)

def classical_evaluate(board: chess.Board) -> int:
    """Hand-written evaluation in centipawns for the side to move, used without weights."""
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    pawns, knights, bishops = board.pawns, board.knights, board.bishops
    rooks, queens, kings = board.rooks, board.queens, board.kings
    white_pawns = pawns & white
    black_pawns = pawns & black

    phase = min(
        PHASE_MAX,
        (knights | bishops).bit_count() + 2 * rooks.bit_count() + 4 * queens.bit_count(),
    )
    score = 0
    middlegame = 0
    endgame = 0

    for square in chess.scan_forward(white_pawns):
        score += PAWN_W[square]
        if not black_pawns & PASSED_W[square]:
            bonus = PASSED_PAWN[square >> 3]
            middlegame += bonus // 2
            endgame += bonus
    for square in chess.scan_forward(black_pawns):
        score -= PAWN_B[square]
        if not white_pawns & PASSED_B[square]:
            bonus = PASSED_PAWN[7 - (square >> 3)]
            middlegame -= bonus // 2
            endgame -= bonus

    for square in chess.scan_forward(knights & white):
        score += KNIGHT_W[square]
    for square in chess.scan_forward(knights & black):
        score -= KNIGHT_B[square]

    white_bishops = bishops & white
    black_bishops = bishops & black
    for square in chess.scan_forward(white_bishops):
        score += BISHOP_W[square]
    for square in chess.scan_forward(black_bishops):
        score -= BISHOP_B[square]
    if white_bishops.bit_count() >= 2:
        score += BISHOP_PAIR
    if black_bishops.bit_count() >= 2:
        score -= BISHOP_PAIR

    for square in chess.scan_forward(rooks & white):
        score += ROOK_W[square]
        file_mask = chess.BB_FILES[square & 7]
        if not pawns & file_mask:
            score += ROOK_OPEN_FILE
        elif not white_pawns & file_mask:
            score += ROOK_SEMI_OPEN_FILE
    for square in chess.scan_forward(rooks & black):
        score -= ROOK_B[square]
        file_mask = chess.BB_FILES[square & 7]
        if not pawns & file_mask:
            score -= ROOK_OPEN_FILE
        elif not black_pawns & file_mask:
            score -= ROOK_SEMI_OPEN_FILE

    for square in chess.scan_forward(queens & white):
        score += QUEEN_W[square]
    for square in chess.scan_forward(queens & black):
        score -= QUEEN_B[square]

    for file_mask, adjacent in zip(chess.BB_FILES, ADJACENT_FILES, strict=True):
        on_file = (white_pawns & file_mask).bit_count()
        if on_file:
            if on_file > 1:
                score -= DOUBLED_PAWN * (on_file - 1)
            if not white_pawns & adjacent:
                score -= ISOLATED_PAWN * on_file
        on_file = (black_pawns & file_mask).bit_count()
        if on_file:
            if on_file > 1:
                score += DOUBLED_PAWN * (on_file - 1)
            if not black_pawns & adjacent:
                score += ISOLATED_PAWN * on_file

    white_king = (kings & white).bit_length() - 1
    black_king = (kings & black).bit_length() - 1
    middlegame += KING_MG_W[white_king] - KING_MG_B[black_king]
    endgame += KING_EG_W[white_king] - KING_EG_B[black_king]
    middlegame += SHIELD_NEAR * (white_pawns & SHIELD_NEAR_W[white_king]).bit_count()
    middlegame += SHIELD_FAR * (white_pawns & SHIELD_FAR_W[white_king]).bit_count()
    middlegame -= SHIELD_NEAR * (black_pawns & SHIELD_NEAR_B[black_king]).bit_count()
    middlegame -= SHIELD_FAR * (black_pawns & SHIELD_FAR_B[black_king]).bit_count()

    total = score + (middlegame * phase + endgame * (PHASE_MAX - phase)) // PHASE_MAX
    return total + TEMPO if board.turn == chess.WHITE else TEMPO - total


# The network replaces the hand-written evaluation whenever its weights ship beside this file.
NETWORK_LOADED = nnue_eval.load()
evaluate = nnue_eval.evaluate if NETWORK_LOADED else classical_evaluate
print(f"evaluation: {'network' if NETWORK_LOADED else 'classical (no weights file)'}", flush=True)


# --- search --------------------------------------------------------------------------------


def _to_table(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


def _from_table(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score - ply
    if score <= -MATE_BOUND:
        return score + ply
    return score


def _has_pieces(board: chess.Board) -> bool:
    return bool(board.occupied_co[board.turn] & ~(board.pawns | board.kings))


SEE_VALUES = (0, 100, 320, 330, 500, 900, 20_000)


def see(board: chess.Board, move: chess.Move) -> int:
    """Net centipawns from playing out the whole exchange on the target square.

    MVV-LVA alone ranks a queen taking a defended pawn as a fine capture. This plays the
    swap-off out, recomputing attackers against a shrinking occupancy so pieces behind the
    ones that just traded off join the exchange.
    """
    target = move.to_square
    if board.is_en_passant(move):
        captured_value = SEE_VALUES[chess.PAWN]
    else:
        victim = board.piece_type_at(target)
        captured_value = SEE_VALUES[victim] if victim else 0
    attacker = board.piece_type_at(move.from_square)
    if attacker is None:
        return 0

    occupied = board.occupied & ~chess.BB_SQUARES[move.from_square]
    if board.is_en_passant(move) and board.ep_square is not None:
        behind = board.ep_square + (-8 if board.turn == chess.WHITE else 8)
        occupied &= ~chess.BB_SQUARES[behind]

    gains = [captured_value]
    on_square = SEE_VALUES[attacker]
    side = not board.turn
    while True:
        attackers = board.attackers_mask(side, target, occupied) & occupied
        if not attackers:
            break
        square = -1
        value = 0
        for piece_type in range(chess.PAWN, chess.KING + 1):
            subset = attackers & board.pieces_mask(piece_type, side)
            if subset:
                square = chess.lsb(subset)
                value = SEE_VALUES[piece_type]
                break
        if square < 0:
            break
        gains.append(on_square - gains[-1])
        if max(-gains[-2], gains[-1]) < 0:
            break
        on_square = value
        occupied &= ~chess.BB_SQUARES[square]
        side = not side

    for index in range(len(gains) - 1, 0, -1):
        gains[index - 1] = -max(-gains[index - 1], gains[index])
    return gains[0]


class Searcher:
    def __init__(self) -> None:
        self.table: dict[Hashable, TTEntry] = {}
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY + 1)]
        self.history = [0] * (2 * 64 * 64)
        self.path: dict[Hashable, int] = {}
        self.seen: dict[Hashable, int] = {}
        self.nodes = 0
        self.deadline = 0.0

    def choose(self, board: chess.Board, optimum_ms: float, maximum_ms: float) -> chess.Move:
        started = time.perf_counter()
        self.deadline = started + maximum_ms / 1000.0
        self.nodes = 0
        self.path = {}
        self.history = [value >> 1 for value in self.history]

        root_key = board._transposition_key()
        self.seen[root_key] = self.seen.get(root_key, 0) + 1
        entry = self.table.get(root_key)
        moves = self.order(board, entry[3] if entry is not None else None, 0)
        if len(moves) == 1:
            return moves[0]

        best_move = moves[0]
        best_score = 0
        completed = 0
        depth = 1
        while depth <= MAX_DEPTH:
            alpha, beta = -INF, INF
            if depth >= 4:
                alpha, beta = best_score - ASPIRATION, best_score + ASPIRATION
            try:
                while True:
                    score, move = self.search_root(board, moves, depth, alpha, beta)
                    if score <= alpha:
                        alpha = -INF
                    elif score >= beta:
                        beta = INF
                    else:
                        break
            except SearchTimeout:
                break
            best_move, best_score, completed = move, score, depth
            moves.remove(move)
            moves.insert(0, move)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if abs(score) >= MATE_BOUND or elapsed_ms > optimum_ms * 0.6:
                break
            depth += 1

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        # Unflushed output dies with the process at game end, taking the log with it.
        print(
            f"depth {completed} score {best_score:+d} nodes {self.nodes} time {elapsed_ms:.0f}ms",
            flush=True,
        )
        return best_move

    def search_root(
        self, board: chess.Board, moves: list[chess.Move], depth: int, alpha: int, beta: int
    ) -> tuple[int, chess.Move]:
        best = -INF
        best_move = moves[0]
        for index, move in enumerate(moves):
            board.push(move)
            try:
                if index == 0:
                    score = -self.search(board, depth - 1, -beta, -alpha, 1, True)
                else:
                    score = -self.search(board, depth - 1, -alpha - 1, -alpha, 1, True)
                    if alpha < score < beta:
                        score = -self.search(board, depth - 1, -beta, -alpha, 1, True)
            finally:
                board.pop()  # a timeout must never leave the board off its root
            if score > best:
                best, best_move = score, move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return best, best_move

    def search(
        self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int, allow_null: bool
    ) -> int:
        self.nodes += 1
        if not self.nodes & NODE_CHECK_MASK and time.perf_counter() > self.deadline:
            raise SearchTimeout

        if board.halfmove_clock >= 100 or (
            board.occupied.bit_count() <= 4 and board.is_insufficient_material()
        ):
            return DRAW
        key = board._transposition_key()
        if self.path.get(key) or self.seen.get(key):
            return DRAW

        # Mate distance: a faster mate is already available above, so nothing here can matter.
        alpha = max(alpha, -MATE + ply)
        beta = min(beta, MATE - ply - 1)
        if alpha >= beta:
            return alpha

        in_check = board.is_check()
        if in_check:
            depth += 1
        if depth <= 0:
            return self.quiesce(board, alpha, beta, ply)
        if ply >= MAX_PLY:
            return evaluate(board)

        table_move: chess.Move | None = None
        entry = self.table.get(key)
        if entry is not None:
            entry_depth, flag, entry_score, table_move = entry
            if entry_depth >= depth:
                score = _from_table(entry_score, ply)
                if (
                    flag == EXACT
                    or (flag == LOWER and score >= beta)
                    or (flag == UPPER and score <= alpha)
                ):
                    return score

        pv_node = beta - alpha > 1
        # Internal iterative deepening: with no hash move, a shallow search is cheaper than
        # searching this node in a bad order.
        if table_move is None and pv_node and depth >= IID_DEPTH:
            self.search(board, depth - 2, alpha, beta, ply, False)
            probe = self.table.get(key)
            if probe is not None:
                table_move = probe[3]
        static = 0 if in_check else evaluate(board)

        # Reverse futility: so far ahead that handing back a piece a ply would still hold beta.
        if (
            not pv_node
            and not in_check
            and depth <= RFP_DEPTH
            and abs(beta) < MATE_BOUND
            and static - RFP_MARGIN * depth >= beta
        ):
            return static

        if (
            allow_null
            and not pv_node
            and not in_check
            and depth >= 3
            and _has_pieces(board)
            and static >= beta
        ):
            board.push(chess.Move.null())
            reduced = depth - 2 - depth // 6
            try:
                score = -self.search(board, reduced, -beta, -beta + 1, ply + 1, False)
            finally:
                board.pop()
            if score >= beta:
                return beta if score >= MATE_BOUND else score

        moves = self.order(board, table_move, ply)
        if not moves:
            return -MATE + ply if in_check else DRAW

        self.path[key] = 1
        original_alpha = alpha
        best = -INF
        best_move: chess.Move | None = None
        shallow = not pv_node and not in_check
        # Quiet moves at a shallow node this far below alpha are not going to lift it.
        futile = shallow and depth <= FUTILITY_DEPTH and static + FUTILITY_MARGIN * depth <= alpha
        late_from = LMP_COUNT[depth] if shallow and depth <= LMP_DEPTH else 1 << 30
        try:
            for index, move in enumerate(moves):
                quiet = move.promotion is None and not board.is_capture(move)
                board.push(move)
                try:
                    gives_check = board.is_check()
                    if (
                        index > 0
                        and quiet
                        and not gives_check
                        and best > -MATE_BOUND
                        and (futile or index >= late_from)
                    ):
                        continue
                    next_depth = depth - 1
                    if index == 0:
                        score = -self.search(board, next_depth, -beta, -alpha, ply + 1, True)
                    else:
                        reduction = 0
                        if quiet and depth >= 3 and index >= LMR_FROM_MOVE and not in_check:
                            if gives_check:
                                reduction = 0
                            else:
                                reduction = LMR_TABLE[min(depth, 63)][min(index, 63)]
                                if pv_node:
                                    reduction -= 1
                                reduction = max(0, min(reduction, depth - 2))
                        score = -self.search(
                            board, next_depth - reduction, -alpha - 1, -alpha, ply + 1, True
                        )
                        if reduction and score > alpha:
                            score = -self.search(
                                board, next_depth, -alpha - 1, -alpha, ply + 1, True
                            )
                        if alpha < score < beta:
                            score = -self.search(board, next_depth, -beta, -alpha, ply + 1, True)
                finally:
                    board.pop()

                if score > best:
                    best, best_move = score, move
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        if quiet:
                            self.remember_cutoff(board, move, depth, ply)
                        break
        finally:
            self.path[key] = 0

        if best >= beta:
            flag = LOWER
        elif best <= original_alpha:
            flag = UPPER
        else:
            flag = EXACT
        if len(self.table) >= TT_MAX_ENTRIES:
            self.table.clear()
        # Depth-preferred: a shallow result must not evict the deep one it was cheaper to get.
        existing = self.table.get(key)
        if existing is None or depth >= existing[0] or flag == EXACT:
            self.table[key] = (depth, flag, _to_table(best, ply), best_move)
        return best

    def quiesce(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        if not self.nodes & NODE_CHECK_MASK and time.perf_counter() > self.deadline:
            raise SearchTimeout
        if ply >= MAX_PLY:
            return evaluate(board)

        if board.is_check():
            moves = self.order(board, None, ply)
            if not moves:
                return -MATE + ply
            best = -INF
            for move in moves:
                board.push(move)
                try:
                    score = -self.quiesce(board, -beta, -alpha, ply + 1)
                finally:
                    board.pop()
                if score > best:
                    best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
            return best

        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)
        best = stand_pat
        for _, gain, move in self.captures(board):
            if move.promotion is None and stand_pat + gain + DELTA_MARGIN <= alpha:
                continue
            board.push(move)
            try:
                score = -self.quiesce(board, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if score > best:
                best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
        return best

    def order(
        self, board: chess.Board, table_move: chess.Move | None, ply: int
    ) -> list[chess.Move]:
        killers = self.killers[ply]
        history = self.history
        base = 0 if board.turn else 4096

        def priority(move: chess.Move) -> int:
            if move == table_move:
                return 1_000_000
            victim = board.piece_type_at(move.to_square)
            if victim is None and board.is_en_passant(move):
                victim = chess.PAWN
            promotion = PIECE_VALUES[move.promotion] if move.promotion is not None else 0
            if victim is not None:
                attacker = board.piece_type_at(move.from_square) or 0
                rank = 10 * PIECE_VALUES[victim] - ATTACKER_VALUES[attacker] + promotion
                # Trading up needs no proof; only a capture that looks to lose material is
                # worth paying for a swap-off to check.
                if PIECE_VALUES[victim] >= PIECE_VALUES[attacker] or see(board, move) >= 0:
                    return 300_000 + rank
                return 100_000 + rank
            if promotion:
                return 290_000 + promotion
            if move in killers:
                return 200_000
            return min(history[base + move.from_square * 64 + move.to_square], 99_000)

        moves = list(board.legal_moves)
        moves.sort(key=priority, reverse=True)
        return moves

    def captures(self, board: chess.Board) -> list[tuple[int, int, chess.Move]]:
        """Captures and queen promotions as (ordering key, material gain, move), best first."""
        scored: list[tuple[int, int, chess.Move]] = []
        for move in board.generate_legal_captures():
            victim = board.piece_type_at(move.to_square) or chess.PAWN
            attacker = board.piece_type_at(move.from_square) or 0
            gain = PIECE_VALUES[victim]
            # A capture that loses material has nothing to settle, so it buys no accuracy here.
            if PIECE_VALUES[victim] < PIECE_VALUES[attacker] and see(board, move) < 0:
                continue
            scored.append((10 * gain - ATTACKER_VALUES[attacker], gain, move))
        own_pawns = board.pawns & board.occupied_co[board.turn]
        seventh = chess.BB_RANK_7 if board.turn else chess.BB_RANK_2
        if own_pawns & seventh:
            last = chess.BB_RANK_8 if board.turn else chess.BB_RANK_1
            queen = PIECE_VALUES[chess.QUEEN]
            for move in board.generate_legal_moves(own_pawns & seventh, last):
                if move.promotion == chess.QUEEN and not board.is_capture(move):
                    scored.append((10 * queen, queen, move))
        scored.sort(key=itemgetter(0), reverse=True)
        return scored

    def remember_cutoff(self, board: chess.Board, move: chess.Move, depth: int, ply: int) -> None:
        killers = self.killers[ply]
        if move != killers[0]:
            killers[1] = killers[0]
            killers[0] = move
        base = 0 if board.turn else 4096
        self.history[base + move.from_square * 64 + move.to_square] += depth * depth


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


SEARCHER = Searcher()
CLOCK = Clock()


def _fallback(board: chess.Board) -> chess.Move:
    return SEARCHER.order(board, None, 0)[0]


def _pick(board: chess.Board, time_left_ms: int) -> chess.Move:
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
    position = chess.Board(fen)
    if move is None or move not in position.legal_moves:
        move = _fallback(position)
    CLOCK.finish(started)
    return move.uci()
