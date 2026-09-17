"""The compiled search: principal variation search over the network, entirely inside numba.

This is the search versions/v9b ran in Python, rule for rule: the same pruning, margins, move
ordering and draw detection. What changes is where it runs. The Python search spent about 87% of
its time in python-chess and its own bookkeeping; here a node is a few hundred machine
instructions, and only iterative deepening and the clock stay in agent.py.

Two things work differently because compiled code needs them to:

- Running out of time sets `stopped` and every node returns straight away, where the Python
  search raised an exception. A search that stopped leaves no trace: nothing after the stop is
  stored, and the root throws the unfinished iteration away.
- The transposition table is a fixed array indexed by the key rather than a dictionary, so two
  positions can compete for a slot. Depth-preferred replacement is kept for a key already there.
"""

import math
import time

import numba
import numpy as np
from numba import boolean, float64, int8, int16, int32, int64, njit, objmode, types, uint64
from numba.experimental import structref

from evaluation import NETWORK, evaluate
from position import (
    CACHE,
    EMPTY,
    EN_PASSANT,
    MAX_MOVES,
    MAX_PLY,
    MOVES,
    NULL_MOVE,
    POSITION,
    U0,
    WHITE,
    generate,
    has_pieces,
    in_check,
    insufficient_material,
    make_move,
    make_null,
    popcount,
    see,
    unmake_move,
    unmake_null,
)

NODE_CHECK_MASK = 1023
SEARCH_MAX_PLY = 96
INF = 40_000
MATE = 32_000
MATE_BOUND = 31_000
DRAW = 0
EXACT, LOWER, UPPER = 0, 1, 2
DELTA_MARGIN = 150
LMR_FROM_MOVE = 3
RFP_DEPTH = 6
RFP_MARGIN = 85
FUTILITY_DEPTH = 4
FUTILITY_MARGIN = 110
IID_DEPTH = 5
LMP_DEPTH = 5
LMP_COUNT = np.array([0, 6, 10, 16, 24, 34], dtype=np.int64)
NO_LIMIT = 1 << 30

# Reduce late quiet moves harder the deeper the node and the further down the move list, rather
# than by the flat one-or-two plies a hand-written rule gives.
LMR_TABLE = np.array(
    [
        [
            0 if depth == 0 or index == 0 else int(0.75 + math.log(depth) * math.log(index) / 2.25)
            for index in range(64)
        ]
        for depth in range(64)
    ],
    dtype=np.int64,
)

PIECE_VALUES = np.array([100, 320, 330, 500, 900, 0], dtype=np.int64)
ATTACKER_VALUES = np.array([100, 320, 330, 500, 900, 1000], dtype=np.int64)

# About 1M entries, the size the Python table was cleared at: 19 bytes each, 20 MB in all.
TABLE_SIZE = 1 << 20
TABLE_MASK = np.uint64(TABLE_SIZE - 1)
GAME_KEYS = 1024  # our positions over a whole game; the game ends at 600 plies


class SearchStateType(types.StructRef):
    def preprocess_fields(self, fields):  # type: ignore[no-untyped-def]
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


class SearchState(structref.StructRefProxy):
    """Everything the search keeps: the table, killers and history last for the whole game."""


STATE_FIELDS = (
    ("table_keys", uint64[::1]),
    ("table_moves", int32[::1]),
    ("table_scores", int32[::1]),
    ("table_depths", int16[::1]),  # -1 marks an empty slot
    ("table_flags", int8[::1]),
    ("killers", int32[:, ::1]),
    ("history", int64[::1]),
    ("moves", int32[:, ::1]),  # a move list per ply
    ("scores", int64[:, ::1]),  # and the ordering score of each move in it
    ("game_keys", uint64[::1]),  # the positions we searched from earlier in this game
    ("game_length", int64),
    ("nodes", int64),
    ("deadline", float64),  # when to stop, on time.perf_counter()
    ("stopped", boolean),
)
structref.register(SearchStateType)
structref.define_proxy(SearchState, SearchStateType, [name for name, _ in STATE_FIELDS])
STATE = SearchStateType(STATE_FIELDS)


def new_state() -> SearchState:
    values = {
        "table_keys": np.zeros(TABLE_SIZE, dtype=np.uint64),
        "table_moves": np.zeros(TABLE_SIZE, dtype=np.int32),
        "table_scores": np.zeros(TABLE_SIZE, dtype=np.int32),
        "table_depths": np.full(TABLE_SIZE, -1, dtype=np.int16),
        "table_flags": np.zeros(TABLE_SIZE, dtype=np.int8),
        "killers": np.zeros((MAX_PLY + 1, 2), dtype=np.int32),
        "history": np.zeros(2 * 64 * 64, dtype=np.int64),
        "moves": np.zeros((MAX_PLY + 1, MAX_MOVES), dtype=np.int32),
        "scores": np.zeros((MAX_PLY + 1, MAX_MOVES), dtype=np.int64),
        "game_keys": np.zeros(GAME_KEYS, dtype=np.uint64),
        "game_length": np.int64(0),
        "nodes": np.int64(0),
        "deadline": 0.0,
        "stopped": False,
    }
    return SearchState(*(values[name] for name, _ in STATE_FIELDS))


assert numba.typeof(new_state()) == STATE
# What probe_cutoff and search_root return: (score, move, settled) and (score, index, stopped).
CUTOFF = types.Tuple((int64, int32, boolean))
ROOT_RESULT = types.Tuple((int64, int64, boolean))


# --- time ----------------------------------------------------------------------------------


@njit(float64(), cache=CACHE)
def now():  # type: ignore[no-untyped-def]
    """Seconds on Python's clock. objmode steps out to Python for one call, about 1.6 us."""
    with objmode(seconds="float64"):
        seconds = time.perf_counter()
    return seconds


@njit(boolean(STATE), cache=CACHE)
def out_of_time(s):  # type: ignore[no-untyped-def]
    """Count a node, and read the clock every 1024 of them."""
    s.nodes += 1
    if (s.nodes & NODE_CHECK_MASK) == 0 and now() > s.deadline:
        s.stopped = True
    return s.stopped


# --- transposition table -------------------------------------------------------------------


@njit(int64(int64, int64), cache=CACHE)
def to_table(score, ply):  # type: ignore[no-untyped-def]
    """Mate scores are stored as distance from this node, so they read right from any ply."""
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


@njit(int64(int64, int64), cache=CACHE)
def from_table(score, ply):  # type: ignore[no-untyped-def]
    if score >= MATE_BOUND:
        return score - ply
    if score <= -MATE_BOUND:
        return score + ply
    return score


@njit(int64(STATE, uint64), cache=CACHE)
def probe(s, key):  # type: ignore[no-untyped-def]
    """The slot holding key, or -1."""
    slot = np.int64(key & TABLE_MASK)
    if s.table_depths[slot] >= 0 and s.table_keys[slot] == key:
        return slot
    return -1


@njit(types.void(STATE, uint64, int64, int64, int64, int64), cache=CACHE)
def store(s, key, depth, flag, score, move):  # type: ignore[no-untyped-def]
    slot = np.int64(key & TABLE_MASK)
    same = s.table_depths[slot] >= 0 and s.table_keys[slot] == key
    # Depth-preferred: a shallow result must not evict the deep one it was cheaper to get.
    if not same or depth >= s.table_depths[slot] or flag == EXACT:
        s.table_keys[slot] = key
        s.table_depths[slot] = depth
        s.table_flags[slot] = flag
        s.table_scores[slot] = score
        s.table_moves[slot] = move


@njit(int32(STATE, uint64), cache=CACHE)
def table_move(s, key):  # type: ignore[no-untyped-def]
    slot = probe(s, key)
    return s.table_moves[slot] if slot >= 0 else np.int32(NULL_MOVE)


# --- draws ---------------------------------------------------------------------------------


@njit(boolean(POSITION, STATE), cache=CACHE)
def repeated(pos, s):  # type: ignore[no-untyped-def]
    """Whether this position already stands earlier on the search path or in the game.

    Only positions with the same side to move and no pawn move or capture since can match, so
    both scans step back two plies at a time and stop at the halfmove clock. A null move is not
    a real move, so the path scan stops there too.
    """
    key = pos.key
    earlier = pos.ply - 2
    while earlier >= 0 and pos.ply - earlier <= pos.halfmove:
        if pos.undo_moves[earlier] == NULL_MOVE or pos.undo_moves[earlier + 1] == NULL_MOVE:
            break
        if pos.undo_keys[earlier] == key:
            return True
        earlier -= 2
    back = 0
    while back < s.game_length and 2 * back + pos.ply <= pos.halfmove:
        if s.game_keys[s.game_length - 1 - back] == key:
            return True
        back += 1
    return False


# --- move ordering -------------------------------------------------------------------------


@njit(types.void(POSITION, STATE, int64, int64, int64), cache=CACHE)
def score_moves(pos, s, ply, count, hash_move):  # type: ignore[no-untyped-def]
    """Hash move, then captures that do not lose material (MVV-LVA), promotions, killers,
    history, and captures that lose material last among the noisy moves."""
    moves = s.moves[ply]
    scores = s.scores[ply]
    killer_0 = s.killers[ply, 0]
    killer_1 = s.killers[ply, 1]
    base = 0 if pos.side == WHITE else 4096
    for index in range(count):
        move = moves[index]
        if move == hash_move:
            scores[index] = 1_000_000
            continue
        from_square = move & 63
        to_square = (move >> 6) & 63
        promotion_type = (move >> 12) & 7
        promotion = PIECE_VALUES[promotion_type] if promotion_type else 0
        victim = pos.mailbox[to_square]
        victim_type = -1
        if victim != EMPTY:
            victim_type = victim % 6
        elif ((move >> 15) & 3) == EN_PASSANT:
            victim_type = 0
        if victim_type >= 0:
            attacker_type = pos.mailbox[from_square] % 6
            rank = 10 * PIECE_VALUES[victim_type] - ATTACKER_VALUES[attacker_type] + promotion
            # Trading up needs no proof; only a capture that looks to lose material is worth
            # paying for a swap-off to check.
            if PIECE_VALUES[victim_type] >= PIECE_VALUES[attacker_type] or see(pos, move) >= 0:
                scores[index] = 300_000 + rank
            else:
                scores[index] = 100_000 + rank
        elif promotion:
            scores[index] = 290_000 + promotion
        elif move in (killer_0, killer_1):
            scores[index] = 200_000
        else:
            scores[index] = min(s.history[base + from_square * 64 + to_square], 99_000)


@njit(int64(POSITION, STATE, int64, int64), cache=CACHE)
def score_captures(pos, s, ply, count):  # type: ignore[no-untyped-def]
    """Order quiescence moves by gain, and drop captures that lose material. Returns the count."""
    moves = s.moves[ply]
    scores = s.scores[ply]
    kept = 0
    for index in range(count):
        move = moves[index]
        to_square = (move >> 6) & 63
        victim = pos.mailbox[to_square]
        if victim == EMPTY and ((move >> 15) & 3) != EN_PASSANT:
            scores[kept] = 10 * PIECE_VALUES[4]  # a push that promotes to a queen
        else:
            victim_type = victim % 6 if victim != EMPTY else 0
            attacker_type = pos.mailbox[move & 63] % 6
            # A capture that loses material has nothing to settle, so it buys no accuracy here.
            if PIECE_VALUES[victim_type] < PIECE_VALUES[attacker_type] and see(pos, move) < 0:
                continue
            scores[kept] = 10 * PIECE_VALUES[victim_type] - ATTACKER_VALUES[attacker_type]
        moves[kept] = move
        kept += 1
    return kept


@njit(int32(STATE, int64, int64, int64), cache=CACHE)
def pick(s, ply, start, count):  # type: ignore[no-untyped-def]
    """Bring the best remaining move to start and return it.

    Only as much of the list gets sorted as the node searches, which after a cutoff is one or
    two moves. Ties keep their order, as a stable sort would.
    """
    moves = s.moves[ply]
    scores = s.scores[ply]
    best = start
    for index in range(start + 1, count):
        if scores[index] > scores[best]:
            best = index
    move = moves[best]
    score = scores[best]
    for index in range(best, start, -1):
        moves[index] = moves[index - 1]
        scores[index] = scores[index - 1]
    moves[start] = move
    scores[start] = score
    return move


@njit(types.void(POSITION, STATE, int64, int64, int64), cache=CACHE)
def remember_cutoff(pos, s, move, depth, ply):  # type: ignore[no-untyped-def]
    if move != s.killers[ply, 0]:
        s.killers[ply, 1] = s.killers[ply, 0]
        s.killers[ply, 0] = move
    base = 0 if pos.side == WHITE else 4096
    s.history[base + (move & 63) * 64 + ((move >> 6) & 63)] += depth * depth


# --- search --------------------------------------------------------------------------------


@njit(int64(POSITION, NETWORK, STATE, int64, int64, int64), cache=CACHE)
def quiesce(pos, net, s, alpha, beta, ply):  # type: ignore[no-untyped-def]
    if out_of_time(s):
        return 0
    if ply >= SEARCH_MAX_PLY:
        return evaluate(pos, net)

    if in_check(pos):
        # No standing pat in check: every evasion is searched, and none means mate.
        count = generate(pos, s.moves[ply], False)
        score_moves(pos, s, ply, count, NULL_MOVE)
        best = -INF
        legal = 0
        for next_index in range(count):
            move = pick(s, ply, next_index, count)
            if not make_move(pos, move):
                continue
            legal += 1
            score = -quiesce(pos, net, s, -beta, -alpha, ply + 1)
            unmake_move(pos)
            if s.stopped:
                return 0
            if score > best:
                best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
        return best if legal else -MATE + ply

    stand_pat = evaluate(pos, net)
    if stand_pat >= beta:
        return stand_pat
    alpha = max(alpha, stand_pat)
    best = stand_pat
    count = score_captures(pos, s, ply, generate(pos, s.moves[ply], True))
    for next_index in range(count):
        move = pick(s, ply, next_index, count)
        victim = pos.mailbox[(move >> 6) & 63]
        if ((move >> 12) & 7) == 0:
            gain = PIECE_VALUES[victim % 6] if victim != EMPTY else PIECE_VALUES[0]
            if stand_pat + gain + DELTA_MARGIN <= alpha:
                continue
        if not make_move(pos, move):
            continue
        score = -quiesce(pos, net, s, -beta, -alpha, ply + 1)
        unmake_move(pos)
        if s.stopped:
            return 0
        if score > best:
            best = score
        if score > alpha:
            alpha = score
            if alpha >= beta:
                break
    return best


@njit(boolean(POSITION, STATE), cache=CACHE)
def is_draw(pos, s):  # type: ignore[no-untyped-def]
    """Fifty moves, bare material, or a repetition."""
    if pos.halfmove >= 100:
        return True
    if popcount(pos.colours[0] | pos.colours[1]) <= 4 and insufficient_material(pos):
        return True
    return repeated(pos, s)


@njit(CUTOFF(STATE, uint64, int64, int64, int64, int64), cache=CACHE)
def probe_cutoff(s, key, depth, alpha, beta, ply):  # type: ignore[no-untyped-def]
    """Look the position up: the stored score when it settles this node, the stored move, and
    whether the score settles it."""
    slot = probe(s, key)
    if slot < 0:
        return 0, np.int32(NULL_MOVE), False
    move = s.table_moves[slot]
    if s.table_depths[slot] < depth:
        return 0, move, False
    score = from_table(np.int64(s.table_scores[slot]), ply)
    flag = s.table_flags[slot]
    settled = (
        flag == EXACT or (flag == LOWER and score >= beta) or (flag == UPPER and score <= alpha)
    )
    return score, move, settled


@njit(int64(int64, int64, boolean), cache=CACHE)
def reduction(depth, index, pv_node):  # type: ignore[no-untyped-def]
    reduce = LMR_TABLE[min(depth, 63), min(index, 63)]
    if pv_node:
        reduce -= 1
    return max(0, min(reduce, depth - 2))


@njit(boolean(POSITION, int64), cache=CACHE)
def is_quiet(pos, move):  # type: ignore[no-untyped-def]
    return (
        ((move >> 12) & 7) == 0
        and pos.mailbox[(move >> 6) & 63] == EMPTY
        and ((move >> 15) & 3) != EN_PASSANT
    )


@njit(int64(POSITION, NETWORK, STATE, int64, int64, int64, int64, int64), cache=CACHE)
def search(pos, net, s, depth, alpha, beta, ply, allow_null):  # type: ignore[no-untyped-def]
    """Principal variation search.

    search calls itself from exactly three places: internal iterative deepening, the null
    move, and one call inside the move loop that serves every re-search. numba types a recursive
    function again at each place it recurses, and seven call sites took 25 s to compile.

    Arguments to those calls are never written as constants. numba compiles a separate copy of
    the function for a constant argument, even with a pinned signature, and `True` and `False`
    for allow_null cost two extra copies of the whole search. null_off and null_on are runtime
    values, so they do not count as constants.
    """
    null_off = np.int64(0)
    null_on = np.int64(1)
    if out_of_time(s):
        return 0
    if is_draw(pos, s):
        return DRAW

    # Mate distance: a faster mate is already available above, so nothing here can matter.
    alpha = max(alpha, -MATE + ply)
    beta = min(beta, MATE - ply - 1)
    if alpha >= beta:
        return alpha

    checked = in_check(pos)
    if checked:
        depth += 1
    if depth <= 0:
        return quiesce(pos, net, s, alpha, beta, ply)
    if ply >= SEARCH_MAX_PLY:
        return evaluate(pos, net)

    key = pos.key
    stored, hash_move, settled = probe_cutoff(s, key, depth, alpha, beta, ply)
    if settled:
        return stored

    pv_node = beta - alpha > 1
    # Internal iterative deepening: with no hash move, a shallow search is cheaper than
    # searching this node in a bad order.
    if hash_move == NULL_MOVE and pv_node and depth >= IID_DEPTH:
        search(pos, net, s, depth - 2, alpha, beta, ply, null_off)
        if s.stopped:
            return 0
        hash_move = table_move(s, key)
    static = 0 if checked else evaluate(pos, net)

    # Reverse futility: so far ahead that handing back a piece a ply would still hold beta.
    shallow = not pv_node and not checked
    if (
        shallow
        and depth <= RFP_DEPTH
        and abs(beta) < MATE_BOUND
        and static - RFP_MARGIN * depth >= beta
    ):
        return static

    if allow_null != 0 and shallow and depth >= 3 and static >= beta and has_pieces(pos):
        make_null(pos)
        score = -search(pos, net, s, depth - 2 - depth // 6, -beta, -beta + 1, ply + 1, null_off)
        unmake_null(pos)
        if s.stopped:
            return 0
        if score >= beta:
            return beta if score >= MATE_BOUND else score

    count = generate(pos, s.moves[ply], False)
    score_moves(pos, s, ply, count, hash_move)

    original_alpha = alpha
    best = -INF
    best_move = np.int32(NULL_MOVE)
    # Quiet moves at a shallow node this far below alpha are not going to lift it.
    futile = shallow and depth <= FUTILITY_DEPTH and static + FUTILITY_MARGIN * depth <= alpha
    late_from = LMP_COUNT[depth] if shallow and depth <= LMP_DEPTH else NO_LIMIT
    legal = 0
    for next_index in range(count):
        move = pick(s, ply, next_index, count)
        quiet = is_quiet(pos, move)
        if not make_move(pos, move):
            continue
        index = legal
        legal += 1
        gives_check = in_check(pos)
        prunable = index > 0 and quiet and not gives_check and best > -MATE_BOUND
        if prunable and (futile or index >= late_from):
            unmake_move(pos)
            continue

        # The first move gets the full window. Later ones get a null window, reduced if quiet
        # and late; a reduced search that beats alpha is repeated at full depth, and a null
        # window that lands inside (alpha, beta) is repeated with the full window.
        full_depth = depth - 1
        child_depth = full_depth
        late_quiet = index >= LMR_FROM_MOVE and quiet and not checked and not gives_check
        if late_quiet and depth >= 3:
            child_depth -= reduction(depth, index, pv_node)
        full_window = index == 0
        while True:
            child_alpha = -beta if full_window else -alpha - 1
            score = -search(pos, net, s, child_depth, child_alpha, -alpha, ply + 1, null_on)
            if s.stopped or full_window:
                break
            if child_depth < full_depth:
                if score <= alpha:
                    break
                child_depth = full_depth
            elif alpha < score < beta:
                full_window = True
            else:
                break
        unmake_move(pos)
        if s.stopped:
            return 0

        if score > best:
            best = score
            best_move = move
        if score > alpha:
            alpha = score
            if alpha >= beta:
                if quiet:
                    remember_cutoff(pos, s, move, depth, ply)
                break

    if legal == 0:
        return -MATE + ply if checked else DRAW

    if best >= beta:
        flag = LOWER
    elif best <= original_alpha:
        flag = UPPER
    else:
        flag = EXACT
    store(s, key, depth, flag, to_table(best, ply), best_move)
    return best


# --- the root ------------------------------------------------------------------------------


@njit(int64(POSITION, STATE, int64, MOVES), cache=CACHE)
def order_root(pos, s, hash_move, root_moves):  # type: ignore[no-untyped-def]
    """Write the legal moves into root_moves, best first, and return how many."""
    count = generate(pos, s.moves[0], False)
    score_moves(pos, s, 0, count, hash_move)
    legal = 0
    for next_index in range(count):
        move = pick(s, 0, next_index, count)
        if make_move(pos, move):
            unmake_move(pos)
            root_moves[legal] = move
            legal += 1
    return legal


@njit(int64(POSITION, STATE, float64, MOVES), cache=CACHE)
def begin(pos, s, maximum_seconds, root_moves):  # type: ignore[no-untyped-def]
    """Start a search from pos: set the deadline, note the position, order the root moves."""
    s.deadline = now() + maximum_seconds
    s.nodes = 0
    s.stopped = False
    for index in range(s.history.shape[0]):
        s.history[index] >>= 1
    if s.game_length < GAME_KEYS:
        s.game_keys[s.game_length] = pos.key
        s.game_length += 1
    return order_root(pos, s, table_move(s, pos.key), root_moves)


@njit(
    ROOT_RESULT(POSITION, NETWORK, STATE, MOVES, int64, int64, int64, int64),
    cache=CACHE,
)
def search_root(pos, net, s, root_moves, count, depth, alpha, beta):  # type: ignore[no-untyped-def]
    """One iteration over the root moves. Returns the score, the best move's index, and whether
    time ran out, in which case the other two mean nothing."""
    child_ply = np.int64(1)  # not the constant 1: see search()
    allow_null = np.int64(1)
    best = -INF
    best_index = 0
    for index in range(count):
        make_move(pos, root_moves[index])
        if index == 0:
            score = -search(pos, net, s, depth - 1, -beta, -alpha, child_ply, allow_null)
        else:
            score = -search(pos, net, s, depth - 1, -alpha - 1, -alpha, child_ply, allow_null)
            if not s.stopped and alpha < score < beta:
                score = -search(pos, net, s, depth - 1, -beta, -alpha, child_ply, allow_null)
        unmake_move(pos)
        if s.stopped:
            return 0, 0, True
        if score > best:
            best = score
            best_index = index
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best, best_index, False


@njit(int64(STATE), cache=CACHE)
def node_count(s):  # type: ignore[no-untyped-def]
    return s.nodes


@njit(types.void(STATE), cache=CACHE)
def clear(s):  # type: ignore[no-untyped-def]
    """Forget everything, as a new game would."""
    s.table_depths[:] = -1
    s.killers[:, :] = 0
    s.history[:] = 0
    s.game_length = 0
    s.game_keys[:] = U0
