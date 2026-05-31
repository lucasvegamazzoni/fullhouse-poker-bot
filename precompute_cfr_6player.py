# =============================================================================
# precompute_cfr.py  —  Full-Hand CFR Trainer  (v4, 6-player)
# =============================================================================

import eval7
import numpy as np
import random
import os
import signal
import sys
from collections import defaultdict

# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

OUTPUT_DIR       = "bots/mybot/data"
OUTPUT_FILE      = os.path.join(OUTPUT_DIR, "blueprint.npz")
NUM_PLAYERS      = 6
NUM_ITERATIONS   = 10_000        # quick test; increase to 1_000_000 for overnight
CHECKPOINT_EVERY = 1_000         # save + print progress every N iterations
STARTING_STACK   = 10_000
SMALL_BLIND      = 50
BIG_BLIND        = 100

RAISE_SIZES = {
    "raise_small": 0.5,    # 50% pot  — probing bet
    "raise_pot":   1.0,    # 100% pot — standard strong bet
    "raise_large": 2.0,    # 200% pot — polarising overbet
}
ACTIONS = ["fold", "call"] + list(RAISE_SIZES.keys())

# Maximum raises per street before we treat remaining action as call-only.
# Prevents infinite raise loops.
MAX_RAISES_PER_STREET = 3

# =============================================================================
# SECTION 2 — HAND ABSTRACTION
#
# Abstracting hands into groups is ESSENTIAL for tractability.
# Full NLHE has 10^160 game states — we can't visit all of them.
# By grouping similar hands, we reduce the tree to a manageable size
# while keeping the strategy approximately correct.
#
# PREFLOP: 169 canonical hands → 10 groups
# POSTFLOP: raw eval7 score → 6 strength buckets (0=weakest, 5=strongest)
# =============================================================================

# ── Preflop groups ────────────────────────────────────────────────────────────

PREFLOP_GROUPS = {
    # Group name          : set of canonical hand strings that belong to it
    "PREMIUM_PAIR"    : {"AA","KK","QQ"},
    "STRONG_PAIR"     : {"JJ","TT","99","88"},
    "MEDIUM_PAIR"     : {"77","66","55"},
    "SMALL_PAIR"      : {"44","33","22"},
    "BROADWAY_SUITED" : {"AKs","AQs","AJs","ATs","KQs","KJs","QJs"},
    "BROADWAY_OFFSUIT": {"AKo","AQo","AJo","ATo","KQo"},
    "SUITED_ACE"      : {"A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s"},
    "SUITED_CONNECTOR": {"JTs","T9s","98s","87s","76s","65s","54s"},
    "MARGINAL"        : {"KTo","KJo","QJo","QTo","JTo","K9s","Q9s","J9s","T8s"},
    # Everything else = "TRASH"
}

# Build reverse lookup: canonical string → group name
HAND_TO_GROUP = {}
for group_name, hands in PREFLOP_GROUPS.items():
    for h in hands:
        HAND_TO_GROUP[h] = group_name

# Numeric strength of each preflop group (for logging/debugging)
GROUP_RANK = {
    "TRASH":            0,
    "SMALL_PAIR":       1,
    "MARGINAL":         2,
    "SUITED_CONNECTOR": 3,
    "SUITED_ACE":       4,
    "MEDIUM_PAIR":      5,
    "BROADWAY_OFFSUIT": 6,
    "BROADWAY_SUITED":  7,
    "STRONG_PAIR":      8,
    "PREMIUM_PAIR":     9,
}


def cards_to_preflop_group(card1, card2):
    """
    Map two eval7 Card objects to one of 10 preflop group strings.

    Steps:
      1. Sort so higher rank is always first
      2. Build canonical label e.g. "AKs", "QQ", "72o"
      3. Look up in HAND_TO_GROUP — return "TRASH" if not found

    This is called once per training iteration and once per bot decision.
    It must be fast and deterministic.
    """
    RANKS = "23456789TJQKA"    # index = eval7 rank value (0=2, 12=Ace)

    r1, s1 = card1.rank, card1.suit
    r2, s2 = card2.rank, card2.suit

    # Higher rank first
    if r1 < r2:
        r1, r2, s1, s2 = r2, r1, s2, s1

    rc1 = RANKS[r1]
    rc2 = RANKS[r2]

    if r1 == r2:
        canon = f"{rc1}{rc2}"        # pair — suits irrelevant
    elif s1 == s2:
        canon = f"{rc1}{rc2}s"      # suited
    else:
        canon = f"{rc1}{rc2}o"      # offsuit

    return HAND_TO_GROUP.get(canon, "TRASH")


def cards_to_postflop_bucket(hole_cards, board_cards):
    """
    Map our 5-7 card holding to a strength bucket 0-5.

    Uses eval7 to score the hand, then divides into 6 equal buckets.
    Bucket 5 = top 16.7% of hands (sets, flushes, full houses etc.)
    Bucket 0 = bottom 16.7% (high card, weak pair)

    6 buckets is a good tradeoff for 6-player:
      - Coarser than 8 (reduces info set explosion)
      - Finer than 4 (preserves meaningful distinctions)

    Returns an integer 0-5.
    """
    try:
        h = [eval7.Card(c) if isinstance(c, str) else c for c in hole_cards]
        b = [eval7.Card(c) if isinstance(c, str) else c for c in board_cards]
        cards = (h + b)[:7]

        if len(cards) < 5:
            return 2    # not enough cards — return middle bucket

        score = eval7.evaluate(cards)

        # eval7 scores range 1 to 7462 (higher = stronger hand)
        # Divide into 6 equal-width buckets
        MAX_SCORE = 7462
        bucket    = min(5, int(score / MAX_SCORE * 6))
        return bucket

    except Exception:
        return 2    # middle bucket on any error


def classify_board(board_cards):
    """
    Classify the board into one of 4 texture categories.

    PAIRED   — board has a duplicate rank (trips/boat possible)
    WET      — 2+ suited cards AND connected ranks (many draws)
    SEMI_WET — either suited OR connected but not both
    DRY      — rainbow and disconnected (few draws)

    Board texture matters because:
      - On WET boards, bluffs get called more (opponents have draws)
      - On DRY boards, bluffs work better (fewer draws, more folds)
      - On PAIRED boards, range advantage shifts differently

    Returns one of: "PAIRED", "WET", "SEMI_WET", "DRY"
    """
    if not board_cards:
        return "DRY"

    try:
        b = [eval7.Card(c) if isinstance(c, str) else c for c in board_cards]
        ranks = [c.rank for c in b]
        suits = [c.suit for c in b]

        # Paired board check
        if len(set(ranks)) < len(ranks):
            return "PAIRED"

        # Count suits
        suit_counts     = defaultdict(int)
        for s in suits:
            suit_counts[s] += 1
        flush_possible  = any(v >= 2 for v in suit_counts.values())

        # Check connectivity (all ranks within 4 of each other)
        sorted_ranks    = sorted(set(ranks))
        connected       = (sorted_ranks[-1] - sorted_ranks[0] <= 4)

        if flush_possible and connected:
            return "WET"
        elif flush_possible or connected:
            return "SEMI_WET"
        return "DRY"

    except Exception:
        return "DRY"

# =============================================================================
# SECTION 3 — INFORMATION SET KEY
#
# The info set key uniquely identifies a decision situation.
# Two situations with the same key get the SAME strategy.
#
# Key components:
#   preflop_group  — which of 10 hand groups we're in
#   postflop_bucket— our hand strength bucket (0-5), or "pre" preflop
#   board_texture  — PAIRED/WET/SEMI_WET/DRY, or "pre" preflop
#   street         — preflop/flop/turn/river
#   bet_sequence   — the exact sequence of actions this street
#   position       — BTN/SB/BB/UTG/UTG+1/MP
# =============================================================================

# Position names for 6-player poker (relative to button)
POSITION_NAMES = {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "UTG+1", 5: "MP"}

def make_key(preflop_group, postflop_bucket, board_texture,
             street, bet_sequence, position):
    """
    Build the information set key string for 6-player poker.

    bet_sequence is a tuple of action strings taken THIS street so far.
    For 6-player, encode action sequence with full context.
    Truncate at max length to cap the key space.
    """
    post  = str(postflop_bucket) if postflop_bucket is not None else "pre"
    board = board_texture if board_texture else "pre"
    bets  = "-".join(bet_sequence[:15]) if bet_sequence else "none"  # store full sequence, up to 15 actions
    return f"{preflop_group}|{post}|{board}|{street}|{bets}|{position}"

# =============================================================================
# SECTION 4 — CFR DATA TABLES
# =============================================================================

# regret_sum[key][action]   → cumulative regret for this action at this info set
# strategy_sum[key][action] → cumulative strategy probability
#
# IMPORTANT: the AVERAGE strategy (strategy_sum / total visits) converges
# to Nash Equilibrium. The CURRENT strategy (regret-matching) oscillates.
# Always use the average strategy in the final blueprint.

regret_sum   = defaultdict(lambda: defaultdict(float))
strategy_sum = defaultdict(lambda: defaultdict(float))


def current_strategy(key):
    """
    Regret-matching: turn accumulated regrets into action probabilities.

    Rule: play each action proportionally to how much you regret NOT playing it.
    Negative regrets are floored at 0 — they represent "glad I didn't do that".

    If no regrets accumulated yet (first visit), return uniform distribution.
    This is mathematically the same as initialising with zero regrets.
    """
    regs  = regret_sum[key]
    pos   = {a: max(0.0, regs[a]) for a in ACTIONS}
    total = sum(pos.values())

    if total > 0:
        return {a: pos[a] / total for a in ACTIONS}
    return {a: 1.0 / len(ACTIONS) for a in ACTIONS}


def average_strategy(key):
    """
    The average strategy across all training iterations.
    This is what we save to the blueprint.

    The average converges to Nash because:
    - Early iterations have high regrets and swing wildly
    - Later iterations have lower regrets and stabilise
    - The average smooths out the early oscillations
    """
    totals = strategy_sum[key]
    total  = sum(totals.values())

    if total > 0:
        return {a: totals[a] / total for a in ACTIONS}
    return {a: 1.0 / len(ACTIONS) for a in ACTIONS}

# =============================================================================
# SECTION 5 — GAME MECHANICS
# Helper functions that implement poker rules during CFR simulation.
# =============================================================================

def deal_board_cards(existing_board, hole_p0, hole_p1, n_cards):
    """
    Deal n_cards new community cards, avoiding cards already in play.

    Parameters:
      existing_board — cards already on the board (list of eval7.Card)
      hole_p0, hole_p1 — both players' hole cards (excluded from deck)
      n_cards — how many new cards to deal (3 for flop, 1 for turn/river)

    Returns a list of n new eval7.Card objects.
    """
    # Build the set of cards already in play
    used = set()
    for c in existing_board + list(hole_p0) + list(hole_p1):
        used.add(str(c))

    # Build available deck
    deck = eval7.Deck()
    available = [c for c in deck.cards if str(c) not in used]
    random.shuffle(available)

    return available[:n_cards]


def eval_winner_multiway(hole_cards, board):
    """
    Evaluate all 6 hands at showdown and return payoff array.

    Returns a list of 6 payoff values:
      +1 if player wins (or ties split-pot)
      -1 if player loses

    For split pots: winners split the pot equally
    """
    try:
        scores = []
        for i in range(NUM_PLAYERS):
            h = [c if not isinstance(c, str) else eval7.Card(c)
                 for c in list(hole_cards[i]) + list(board)]
            s = eval7.evaluate(h[:7])
            scores.append(s)

        max_score = max(scores)
        winners = [i for i in range(NUM_PLAYERS) if scores[i] == max_score]
        num_winners = len(winners)

        payoffs = []
        for i in range(NUM_PLAYERS):
            if i in winners:
                payoffs.append(1.0 / num_winners)  # winner's share
            else:
                payoffs.append(-1.0 / num_winners)  # loser's share (negative)

        return payoffs

    except Exception:
        return [0.0] * NUM_PLAYERS  # treat errors as all-tie


def advance_to_next_street(current_street, board, hole_p0, hole_p1):
    """
    Deal the next community cards and return the new street name + board.

    Street progression:
      preflop → flop  (deal 3 cards)
      flop    → turn  (deal 1 card)
      turn    → river (deal 1 card)
      river   → showdown (no new cards)
    """
    if current_street == "preflop":
        new_cards = deal_board_cards(board, hole_p0, hole_p1, 3)
        return "flop", board + new_cards

    elif current_street == "flop":
        new_cards = deal_board_cards(board, hole_p0, hole_p1, 1)
        return "turn", board + new_cards

    elif current_street == "turn":
        new_cards = deal_board_cards(board, hole_p0, hole_p1, 1)
        return "river", board + new_cards

    else:
        return "showdown", board   # river → showdown, no new cards


def street_is_complete(bet_sequence, street_first_to_act_has_option):
    """
    Determine whether the current betting street is over.

    A street ends when:
      (a) The last action was a call after a raise
          e.g. ("raise_pot", "call") — someone raised and was called
      (b) Both players checked (action count == 2, all calls with no raise)
          e.g. ("call", "call") preflop = both limped
      (c) The raise cap was hit and the last action was a call

    Returns True if the street is over.
    """
    if not bet_sequence:
        return False

    n      = len(bet_sequence)
    last   = bet_sequence[-1]
    raises = [a for a in bet_sequence if "raise" in a]

    # Someone raised and was called → street over
    if last == "call" and len(raises) >= 1:
        return True

    # Both players checked (preflop: both limped — call, call with no raises)
    if n >= 2 and last == "call" and not raises:
        return True

    # Raise cap hit — treat as all-in, street over
    if len(raises) >= MAX_RAISES_PER_STREET:
        return True

    return False

# =============================================================================
# SECTION 6 — THE CFR FUNCTION
#
# This is the core algorithm. It recursively simulates a complete hand
# from any point (street, bet sequence, board) to showdown.
#
# Now supports 6 players with simultaneous CFR training.
#
# Parameters:
#   hole_cards    — list of 6 eval7.Card pairs
#   board         — current community cards
#   street        — "preflop"/"flop"/"turn"/"river"
#   pot           — current pot size in chips
#   stacks        — list of 6 remaining stacks
#   bet_sequence  — actions taken so far THIS street (tuple of strings)
#   acting_player — 0-5, whose turn it is
#   reach_probs   — array of 6 counterfactual reach probabilities
#   depth         — recursion depth (safety limit)
#
# Returns: expected value from acting_player's perspective
# =============================================================================

def cfr(hole_cards, board, street,
        pot, stacks,
        bet_sequence, acting_player,
        reach_probs,
        depth=0):

    # ── Safety: prevent runaway recursion ────────────────────────────────────
    if depth > 30:
        payoffs = eval_winner_multiway(hole_cards, board)
        return payoffs[acting_player] * pot / NUM_PLAYERS

    # ── Terminal: showdown ────────────────────────────────────────────────────
    if street == "showdown":
        payoffs = eval_winner_multiway(hole_cards, board)
        return payoffs[acting_player] * pot / NUM_PLAYERS

    # ── Terminal: someone folded ──────────────────────────────────────────────
    if bet_sequence and bet_sequence[-1] == "fold":
        return pot / NUM_PLAYERS

    # ── Street complete: advance to next street ───────────────────────────────
    if street_is_complete(bet_sequence, True):
        if street == "river":
            # River betting done → showdown
            return cfr(hole_cards, board, "showdown",
                       pot, stacks, (), 0,
                       reach_probs, depth + 1)
        else:
            # Deal next street and recurse
            next_street, new_board = advance_to_next_street(
                street, board, hole_cards[0], hole_cards[1]
            )
            # Reset bet sequence, player 0 acts first on new street (UTG)
            return cfr(hole_cards, new_board, next_street,
                       pot, stacks, (),
                       0,  # UTG acts first post-flop
                       reach_probs, depth + 1)

    # ── Active decision node: build info set key ──────────────────────────────
    hole_cards_me = hole_cards[acting_player]
    my_stack = stacks[acting_player]
    preflop_group = cards_to_preflop_group(hole_cards_me[0], hole_cards_me[1])

    # Postflop bucket and board texture only available once board exists
    if board and street != "preflop":
        hole_strs = [str(c) for c in hole_cards_me]
        board_strs = [str(c) for c in board]
        postflop_buck = cards_to_postflop_bucket(hole_strs, board_strs)
        board_tex = classify_board(board_strs)
    else:
        postflop_buck = None
        board_tex = None

    # Position: map acting_player to position name
    position = POSITION_NAMES[acting_player]

    key = make_key(preflop_group, postflop_buck, board_tex,
                   street, bet_sequence, position)

    # ── Get current strategy (regret matching) ────────────────────────────────
    strategy = current_strategy(key)

    # ── Compute counterfactual value for each action ──────────────────────────
    action_vals = {}

    for action in ACTIONS:

        raise_count = sum(1 for a in bet_sequence if "raise" in a)

        if action == "fold":
            # Folding: we lose, opponents gain
            action_vals["fold"] = -pot / NUM_PLAYERS

        elif action == "call":
            # Determine call amount
            if raise_count == 0 and street == "preflop":
                call_amount = BIG_BLIND
            elif raise_count == 0:
                call_amount = 0  # check
            else:
                call_amount = min(int(pot * 0.3), my_stack)

            call_amount = min(call_amount, my_stack)
            new_pot = pot + call_amount
            new_stacks = list(stacks)
            new_stacks[acting_player] -= call_amount
            new_seq = bet_sequence + ("call",)
            next_player = (acting_player + 1) % NUM_PLAYERS

            new_reach = list(reach_probs)
            new_reach[acting_player] *= strategy[action]

            val = cfr(hole_cards, board, street,
                      new_pot, new_stacks,
                      new_seq, next_player,
                      new_reach, depth + 1)

            action_vals[action] = -val  # negate for opponent perspective

        else:
            # Raise
            if raise_count >= MAX_RAISES_PER_STREET:
                action_vals[action] = action_vals.get("call", -pot / NUM_PLAYERS)
                continue

            ratio = RAISE_SIZES[action]
            raise_amt = max(int(pot * ratio), BIG_BLIND * 2)
            raise_amt = min(raise_amt, my_stack)

            new_pot = pot + raise_amt
            new_stacks = list(stacks)
            new_stacks[acting_player] -= raise_amt
            new_seq = bet_sequence + (action,)
            next_player = (acting_player + 1) % NUM_PLAYERS

            new_reach = list(reach_probs)
            new_reach[acting_player] *= strategy[action]

            val = cfr(hole_cards, board, street,
                      new_pot, new_stacks,
                      new_seq, next_player,
                      new_reach, depth + 1)

            action_vals[action] = -val

    # ── Node expected value ───────────────────────────────────────────────────
    node_ev = sum(strategy[a] * action_vals[a] for a in ACTIONS)

    # ── Update regret sums ────────────────────────────────────────────────────
    # For multi-player CFR: opponent reach = product of all other players' reaches
    opp_reach = 1.0
    for i in range(NUM_PLAYERS):
        if i != acting_player:
            opp_reach *= reach_probs[i]

    my_reach = reach_probs[acting_player]

    for action in ACTIONS:
        regret = action_vals[action] - node_ev
        regret_sum[key][action] += opp_reach * regret
        strategy_sum[key][action] += my_reach * strategy[action]

    return node_ev

# =============================================================================
# SECTION 7 — SAVE AND LOAD THE BLUEPRINT
# =============================================================================

def save_blueprint(label=""):
    """
    Compute the average strategy for every info set and save to disk.

    The average strategy (not the current strategy) is what converges
    to Nash Equilibrium. We compute it here by dividing strategy_sum
    by the total visits for each info set.

    File format: numpy compressed archive (.npz)
    Contains one array: "avg_strategy" — a dict wrapped in an object array
    (numpy requires this to save Python dicts).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    avg = {key: average_strategy(key) for key in strategy_sum}

    np.savez_compressed(
        OUTPUT_FILE,
        avg_strategy=np.array([avg], dtype=object),
    )

    tag = f" [{label}]" if label else ""
    n   = len(avg)
    print(f"    ✓ Saved{tag} — {n:,} info sets → {OUTPUT_FILE}")
    return n


def load_blueprint(path):
    """
    Load the blueprint in bot.py at module import time.

    Usage in bot.py:
      import os, numpy as np
      DATA_DIR  = os.environ.get("BOT_DATA_DIR", "bots/mybot/data")
      BLUEPRINT = load_blueprint(os.path.join(DATA_DIR, "blueprint.npz"))

      Then in decide():
        key   = make_key(preflop_group, postflop_bucket, board_texture,
                         street, tuple(bet_sequence), position)
        strat = BLUEPRINT.get(key)
        if strat:
            action_name = max(strat, key=strat.get)   # greedy: highest prob
            # OR: sample_from_strategy(strat) for mixed strategy
        else:
            action = baseline_action(state, equity)   # fallback
    """
    data = np.load(path, allow_pickle=True)
    return data["avg_strategy"][0]


def sample_from_strategy(strat):
    """
    Randomly pick an action according to its probability weight.

    WHY RANDOM AND NOT GREEDY?
    A greedy bot (always picks highest probability action) is exploitable —
    a smart opponent figures out your pattern and counters it.
    A mixed strategy (sampling by probability) is unexploitable at Nash.

    In the finals use sample_from_strategy.
    In testing you can use max(strat, key=strat.get) to see what the
    'main' strategy is for each situation.
    """
    actions = list(strat.keys())
    probs   = np.array([strat[a] for a in actions], dtype=float)
    probs  /= probs.sum()   # re-normalise to handle any floating point drift
    return str(np.random.choice(actions, p=probs))

# =============================================================================
# SECTION 8 — TRAINING LOOP
# =============================================================================

def train():
    """
    Main training loop for 6-player poker.

    Each iteration:
      1. Deal 6 random hands
      2. Run CFR for all 6 players simultaneously
      3. Every CHECKPOINT_EVERY iters: print progress + save

    CONVERGENCE SIGNAL
    Watch the "info sets" count as training progresses:
      - It grows quickly at first (new situations being discovered)
      - It slows down as the tree is more fully explored
      - When it stops growing much, the strategy is close to converged
    """
    print("=" * 65)
    print("  FULLHOUSE 6-PLAYER CFR TRAINER  (v4 — multi-player)")
    print("=" * 65)
    print(f"  Players    : {NUM_PLAYERS}")
    print(f"  Iterations : {NUM_ITERATIONS:,}")
    print(f"  Checkpoint : every {CHECKPOINT_EVERY:,}")
    print(f"  Output     : {OUTPUT_FILE}")
    print(f"  Stop       : Ctrl+C saves and exits cleanly")
    print("=" * 65 + "\n")

    def on_interrupt(sig, frame):
        print("\n\nCtrl+C caught — saving...")
        n = save_blueprint("interrupted")
        print(f"Saved {n:,} info sets. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, on_interrupt)

    for iteration in range(1, NUM_ITERATIONS + 1):

        # Deal a fresh hand for this iteration
        deck = eval7.Deck()
        deck.shuffle()

        hole_cards = [deck.cards[i*2:(i+1)*2] for i in range(NUM_PLAYERS)]
        stacks = [STARTING_STACK - (SMALL_BLIND if i==0 else BIG_BLIND if i==1 else 0)
                  for i in range(NUM_PLAYERS)]
        pot = SMALL_BLIND + BIG_BLIND

        # Train all 6 players simultaneously in one CFR tree
        cfr(hole_cards, [], "preflop",
            pot, stacks,
            (), 0, [1.0] * NUM_PLAYERS)

        # ── Progress checkpoint ───────────────────────────────────────────────
        if iteration % CHECKPOINT_EVERY == 0:
            pct  = 100 * iteration / NUM_ITERATIONS
            n_is = len(regret_sum)
            print(f"  [{pct:5.1f}%]  iter {iteration:>8,}  |  info sets: {n_is:>8,}")
            save_blueprint(f"{pct:.0f}%")

    print("\n  Training complete!")
    save_blueprint("FINAL")


if __name__ == "__main__":
    train()
