# =============================================================================
# precompute_cfr.py  —  6-Player Poker CFR Trainer  (FINAL, all bugs fixed)
# =============================================================================
#
# BUGS FIXED FROM PREVIOUS VERSION
# ──────────────────────────────────
# FIX 1: NUM_ITERATIONS was 100 (test value) → 500,000 for overnight run
#         CHECKPOINT_EVERY was 20 → 10,000
#
# FIX 2: Blinds never recorded in GameState at start of each hand.
#         max_bet was 0 so UTG could "call" for free and street ended instantly.
#         Now: bets_this_street and max_bet are initialised with blind amounts.
#
# FIX 3: eval_winner evaluated ALL 6 players including those who folded.
#         A folded player could "win" the pot with a strong hand they never used.
#         Now: eval_winner takes active_players and only evaluates those players.
#
# FIX 4: Single remaining player terminal returned +1/-1 (wrong scale).
#         eval_winner returns pot-proportional values — terminal must match.
#         Now: returns pot for winner, 0 for everyone else.
#
# FIX 5: Minimum raise could be below the big blind (e.g. 75 chips on first hand).
#         An illegal raise size confuses the game tree.
#         Now: raise = max(50% pot, 2 × BIG_BLIND), capped at stack.
#
# HOW TO RUN (overnight)
# ──────────────────────
#   cd ~/fullhouse-engine
#   source venv/bin/activate
#   mkdir -p bots/mybot/data
#   caffeinate -i python3 precompute_cfr.py
#
#   caffeinate prevents your Mac sleeping mid-training.
#   Ctrl+C at any time — saves progress automatically.
#   Plug in your charger before you sleep.
#
# WHAT TO CHECK IN THE MORNING
# ──────────────────────────────
#   ls -lh bots/mybot/data/blueprint.npz          ← should be a few MB
#   python3 -c "
#     import numpy as np
#     d = np.load('bots/mybot/data/blueprint.npz', allow_pickle=True)
#     s = d['avg_strategy'][0]
#     print(f'Info sets trained: {len(s):,}')
#     k = list(s.keys())[0]
#     print(f'Example key:  {k}')
#     print(f'Example strat:{s[k]}')
#   "
#
# OUTPUT
#   bots/mybot/data/blueprint.npz
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

OUTPUT_DIR       = "bots/myboy/data"       # folder for the blueprint file
OUTPUT_FILE      = os.path.join(OUTPUT_DIR, "blueprint.npz")

NUM_PLAYERS      = 6
NUM_ITERATIONS   = 500_000   # FIX 1: was 100 (test value)
CHECKPOINT_EVERY = 10_000    # FIX 1: was 20

STARTING_STACK   = 10_000
SMALL_BLIND      = 50
BIG_BLIND        = 100

# One raise size kept simple for tractability.
# A 6-player tree with 3 raise sizes would be too large to explore overnight.
# 50% pot is credible and covers most real-game spots.
ACTIONS              = ["fold", "call", "raise_50"]
MAX_RAISES_PER_STREET = 2   # cap raise loops to prevent runaway recursion

# =============================================================================
# SECTION 2 — HAND ABSTRACTION
#
# We group the 169 canonical starting hands into 10 coarser categories.
# Fewer groups = more visits per group = faster CFR convergence.
# Hands within the same group make approximately the same decisions.
# =============================================================================

PREFLOP_GROUPS = {
    "PREMIUM":    {"AA", "KK", "QQ"},
    "STRONG":     {"JJ", "TT", "99", "88"},
    "MEDIUM":     {"77", "66", "55"},
    "SMALL":      {"44", "33", "22"},
    "BROADWAY":   {"AKs", "AQs", "AJs", "KQs", "KJs"},
    "BROADWAY_O": {"AKo", "AQo", "KQo"},
    "ACE":        {"A9s", "A8s", "A7s", "A6s"},
    "CONN":       {"JTs", "T9s", "98s", "87s"},
    "OTHER":      {"KTo", "QJo", "JTo", "K9s"},
    # Everything else → "TRASH" (returned as default)
}

# Build reverse lookup: canonical string → group name (built once at import)
HAND_TO_GROUP = {}
for _group, _hands in PREFLOP_GROUPS.items():
    for _h in _hands:
        HAND_TO_GROUP[_h] = _group


def cards_to_preflop_group(card1, card2):
    """
    Map two eval7 Card objects to one of 10 preflop group strings.

    Steps:
      1. Sort so higher rank is always first (for consistent labelling)
      2. Build canonical label: "AKs", "QQ", "72o" etc.
      3. Look up in HAND_TO_GROUP — return "TRASH" if not found
    """
    RANKS = "23456789TJQKA"   # index = eval7 rank integer (0=2, 12=Ace)
    r1, s1 = card1.rank, card1.suit
    r2, s2 = card2.rank, card2.suit

    if r1 < r2:                          # put higher rank first
        r1, r2, s1, s2 = r2, r1, s2, s1

    rc1, rc2 = RANKS[r1], RANKS[r2]

    if r1 == r2:
        canon = f"{rc1}{rc2}"            # pair e.g. "AA"
    elif s1 == s2:
        canon = f"{rc1}{rc2}s"          # suited e.g. "AKs"
    else:
        canon = f"{rc1}{rc2}o"          # offsuit e.g. "AKo"

    return HAND_TO_GROUP.get(canon, "TRASH")


def cards_to_postflop_bucket(hole_cards, board_cards):
    """
    Map hole cards + board into one of 3 strength buckets.

    0 = weak  (bottom third of hands)
    1 = medium (middle third)
    2 = strong (top third — sets, straights, flushes, strong pairs)

    Uses eval7 to score the hand, divides score range into 3 equal parts.
    Returns 1 (middle) as fallback on any error or insufficient cards.
    """
    try:
        h     = [eval7.Card(c) if isinstance(c, str) else c for c in hole_cards]
        b     = [eval7.Card(c) if isinstance(c, str) else c for c in board_cards]
        cards = (h + b)[:7]
        if len(cards) < 5:
            return 1
        score  = eval7.evaluate(cards)    # higher = better hand
        bucket = min(2, int(score / 7462 * 3))
        return bucket
    except Exception:
        return 1


def classify_board(board_cards):
    """
    Classify the board as PAIR, WET, or DRY.

    PAIR — board has a duplicate rank (trips/boat now possible for anyone)
    WET  — flush draw possible OR ranks are connected (lots of draws)
    DRY  — rainbow and disconnected (few draws, bluffs work better)
    """
    if not board_cards:
        return "DRY"
    try:
        b     = [eval7.Card(c) if isinstance(c, str) else c for c in board_cards]
        ranks = [c.rank for c in b]
        suits = [c.suit for c in b]

        if len(set(ranks)) < len(ranks):   # duplicate rank = paired board
            return "PAIR"

        suit_counts = defaultdict(int)
        for s in suits:
            suit_counts[s] += 1

        flush_draw = any(v >= 2 for v in suit_counts.values())
        connected  = (max(ranks) - min(ranks) <= 4)

        if flush_draw or connected:         # either condition = wet enough
            return "WET"
        return "DRY"

    except Exception:
        return "DRY"

# =============================================================================
# SECTION 3 — GAME STATE
#
# Tracks which players are still active, what they've bet this street,
# and whether anyone is all-in. Copyable for branch exploration in CFR.
# =============================================================================

class GameState:
    """Tracks active players, bets, and all-in flags for one street."""

    def __init__(self, num_players):
        self.active           = set(range(num_players))
        self.all_in           = defaultdict(bool)
        self.bets_this_street = defaultdict(float)
        self.max_bet          = 0
        self.acted            = set()

    def copy(self):
        """
        Create an independent deep copy.
        Called before every recursive branch so changes don't propagate back.
        """
        gs                    = GameState.__new__(GameState)
        gs.active             = self.active.copy()
        gs.all_in             = defaultdict(bool,  self.all_in)
        gs.bets_this_street   = defaultdict(float, self.bets_this_street)
        gs.max_bet            = self.max_bet
        gs.acted              = self.acted.copy()
        return gs

    def fold(self, player_id):
        """Remove a player from the active set — they're out of this hand."""
        self.active.discard(player_id)
        self.acted.add(player_id)

    def bet(self, player_id, total_amount_this_street):
        """
        Record that player's total bet this street is total_amount_this_street.
        Updates max_bet so other players know what they need to call.
        """
        self.bets_this_street[player_id] = total_amount_this_street
        self.max_bet                      = max(self.max_bet, total_amount_this_street)
        self.acted.add(player_id)

    def reset_street(self):
        """Clear all betting info at the start of a new street."""
        self.bets_this_street = defaultdict(float)
        self.max_bet          = 0
        self.acted            = set()

    def get_next_player(self, action_order, current):
        """
        Find the next active, non-all-in player clockwise from current.
        Returns None if no such player exists.
        """
        if current not in action_order:
            return None
        idx = action_order.index(current)
        for i in range(1, len(action_order)):
            p = action_order[(idx + i) % len(action_order)]
            if p in self.active and not self.all_in[p]:
                return p
        return None

    def street_complete(self, action_order):
        """
        The street is over when:
          - Every active player has acted (posted their intention), AND
          - Every active non-all-in player has matched max_bet

        If everyone folds to one player, active has 1 member — handled separately.
        """
        for p in self.active:
            if p not in self.acted:
                return False    # this player hasn't acted yet
            if not self.all_in[p] and self.bets_this_street[p] < self.max_bet:
                return False    # this player hasn't matched the bet
        return True

# =============================================================================
# SECTION 4 — CFR DATA TABLES
# =============================================================================

regret_sum   = defaultdict(lambda: defaultdict(float))
strategy_sum = defaultdict(lambda: defaultdict(float))


def get_strategy(info_set_key):
    """
    Regret-matching: convert accumulated regrets into action probabilities.

    Rule: play each action proportionally to how much you regret NOT playing it.
    Negative regrets floored at 0 (they mean "glad I didn't do that").
    Uniform distribution if no data has been accumulated yet for this key.
    """
    regs     = regret_sum[info_set_key]
    positive = {a: max(0.0, regs[a]) for a in ACTIONS}
    total    = sum(positive.values())
    if total > 0:
        return {a: positive[a] / total for a in ACTIONS}
    return {a: 1.0 / len(ACTIONS) for a in ACTIONS}

# =============================================================================
# SECTION 5 — GAME MECHANICS
# =============================================================================

def eval_winner(hole_cards, board, pot, active_players):
    """
    FIX 3: Only evaluate hands for players still in the hand (active_players).
    Previously evaluated all 6 players including those who folded —
    a folded player's strong hand could incorrectly "win" the pot.

    Returns a payoff list indexed by player (0.0 for folded players).
    Pot-proportional: winner receives `pot`, others receive 0.
    """
    try:
        if not board or len(board) < 3:
            # Not enough board cards to evaluate — split pot as fallback
            share   = pot / len(active_players)
            payoffs = [0.0] * NUM_PLAYERS
            for p in active_players:
                payoffs[p] = share
            return payoffs

        scores = {}
        for i in active_players:
            h         = [c if not isinstance(c, str) else eval7.Card(c)
                         for c in list(hole_cards[i]) + list(board)]
            scores[i] = eval7.evaluate(h[:7])

        max_score = max(scores.values())
        winners   = [i for i, s in scores.items() if s == max_score]
        share     = pot / len(winners)

        payoffs = [0.0] * NUM_PLAYERS
        for i in active_players:
            payoffs[i] = share if i in winners else 0.0

        return payoffs

    except Exception:
        return [0.0] * NUM_PLAYERS


def advance_street(current_street, board, hole_cards_all):
    """
    Deal the next community cards and return (new_street_name, new_board).
    Excludes all cards already in play (hole cards + existing board).
    """
    used      = {str(c) for c in board}
    for h in hole_cards_all:
        for c in h:
            used.add(str(c))

    deck      = eval7.Deck()
    available = [c for c in deck.cards if str(c) not in used]
    random.shuffle(available)

    if current_street == "preflop":
        return "flop",  board + available[:3]
    elif current_street == "flop":
        return "turn",  board + available[:1]
    elif current_street == "turn":
        return "river", board + available[:1]
    return "showdown", board


def get_position_name(player_id, button_pos):
    """Map a player index to their position label relative to the button."""
    positions = ["BTN", "SB", "BB", "UTG", "UTG+1", "MP"]
    return positions[(player_id - button_pos) % NUM_PLAYERS]

# =============================================================================
# SECTION 6 — CFR RECURSION
#
# The heart of the algorithm. Recursively explores the complete game tree
# from preflop through to showdown, computing and accumulating regrets.
#
# Key principle: at each decision point, CFR tries EVERY action, recurses
# to see what would happen, then records how much it regrets NOT taking
# each action. Over many iterations, regrets drive the strategy toward Nash.
# =============================================================================

cfr_call_count = 0


def cfr(hole_cards, board, street,
        pot, stacks,
        game_state, action_order, acting_player,
        bet_sequence, reach_probs, button_pos,
        depth=0):
    """
    Recursive CFR for 6-player NLHE. Returns EV for acting_player at this node.

    Parameters:
      hole_cards    — list of 6 × [card, card] (all players' hole cards)
      board         — community cards dealt so far
      street        — "preflop"/"flop"/"turn"/"river"/"showdown"
      pot           — total chips in pot
      stacks        — list of each player's remaining stack
      game_state    — GameState object tracking active players and bets
      action_order  — clockwise player ordering for this street
      acting_player — whose turn it is (0-5)
      bet_sequence  — tuple of actions taken so far THIS street
      reach_probs   — list of each player's reach probability (starts at 1.0)
      button_pos    — which player has the button (for position labelling)
      depth         — recursion depth (safety counter)
    """
    global cfr_call_count
    cfr_call_count += 1

    # Print a heartbeat every 200k calls so you can see progress
    if cfr_call_count % 200_000 == 0:
        print(f"    [CFR {cfr_call_count/1e6:.1f}M calls] "
              f"depth={depth} street={street} active={len(game_state.active)}")

    # ── Safety limit: prevent runaway recursion ───────────────────────────────
    # If we recurse too deep or make too many calls on one hand,
    # estimate the result from hand strength and exit.
    if depth > 40 or cfr_call_count > 15_000_000:
        payoffs = eval_winner(hole_cards, board, pot, game_state.active)
        return payoffs[acting_player]

    # ── Terminal: showdown ────────────────────────────────────────────────────
    if street == "showdown":
        payoffs = eval_winner(hole_cards, board, pot, game_state.active)
        return payoffs[acting_player]

    # ── Terminal: only one player left (everyone else folded) ─────────────────
    # FIX 4: was returning +1/-1 — inconsistent with pot-proportional eval_winner.
    # Now returns actual pot amount to match the scale of all other payoffs.
    if len(game_state.active) == 1:
        winner  = next(iter(game_state.active))
        payoffs = [0.0] * NUM_PLAYERS
        payoffs[winner] = pot          # winner takes the whole pot
        return payoffs[acting_player]

    # ── Street complete: advance to next street ───────────────────────────────
    if game_state.street_complete(action_order):
        if street == "river":
            # River betting done — go to showdown
            return cfr(hole_cards, board, "showdown",
                       pot, stacks, game_state, action_order,
                       acting_player, (), reach_probs, button_pos, depth + 1)

        # Deal the next community cards
        next_street, new_board = advance_street(street, board, hole_cards)
        new_gs = game_state.copy()
        new_gs.reset_street()

        # Postflop: first to act is SB (first active player left of button)
        sb_pos          = (button_pos + 1) % NUM_PLAYERS
        first_postflop  = sb_pos if sb_pos in new_gs.active \
                          else new_gs.get_next_player(action_order, sb_pos)
        if first_postflop is None:
            first_postflop = action_order[0]

        return cfr(hole_cards, new_board, next_street,
                   pot, stacks, new_gs, action_order,
                   first_postflop, (), reach_probs, button_pos, depth + 1)

    # ── Skip inactive or all-in players ──────────────────────────────────────
    if acting_player not in game_state.active or game_state.all_in[acting_player]:
        next_p = game_state.get_next_player(action_order, acting_player)
        if next_p is None:
            # No one left to act — force street completion on next call
            return cfr(hole_cards, board, street, pot, stacks,
                       game_state, action_order, acting_player,
                       bet_sequence + ("call",),
                       reach_probs, button_pos, depth + 1)
        return cfr(hole_cards, board, street, pot, stacks,
                   game_state, action_order, next_p,
                   bet_sequence, reach_probs, button_pos, depth + 1)

    # ── Decision node: build information set key ──────────────────────────────
    h             = hole_cards[acting_player]
    preflop_group = cards_to_preflop_group(h[0], h[1])
    my_stack      = stacks[acting_player]

    if board and street != "preflop":
        bucket  = cards_to_postflop_bucket([str(c) for c in h], [str(c) for c in board])
        texture = classify_board([str(c) for c in board])
    else:
        bucket  = None
        texture = None

    position = get_position_name(acting_player, button_pos)

    # Truncate bet sequence at 3 to cap the key space
    # (longer sequences are rare enough that uniform play is fine)
    bets_str = "-".join(bet_sequence[:3]) if bet_sequence else "none"
    info_key = f"{preflop_group}|{bucket}|{texture}|{street}|{bets_str}|{position}"

    strategy = get_strategy(info_key)

    # ── Evaluate each action by recursing ────────────────────────────────────
    action_values = {}

    for action in ACTIONS:
        state_copy  = game_state.copy()
        stacks_copy = list(stacks)
        reach_copy  = list(reach_probs)
        reach_copy[acting_player] *= strategy[action]

        if action == "fold":
            # Fold: leave the hand — opponent(s) continue without us
            state_copy.fold(acting_player)
            next_player = state_copy.get_next_player(action_order, acting_player)

            if not next_player or not state_copy.active:
                # Everyone else folded too — shouldn't normally happen here
                action_values[action] = 0.0
            else:
                action_values[action] = cfr(
                    hole_cards, board, street, pot, stacks_copy,
                    state_copy, action_order, next_player,
                    bet_sequence + ("fold",),
                    reach_copy, button_pos, depth + 1
                )

        elif action == "call":
            # Call: match the current highest bet, pot grows, move to next player
            owed        = max(0.0, state_copy.max_bet
                              - state_copy.bets_this_street[acting_player])
            call_amount = min(owed, my_stack)

            stacks_copy[acting_player] -= call_amount
            state_copy.bet(acting_player, state_copy.max_bet)

            if call_amount >= my_stack:
                state_copy.all_in[acting_player] = True

            next_player = state_copy.get_next_player(action_order, acting_player) \
                          or acting_player

            action_values[action] = cfr(
                hole_cards, board, street, pot + call_amount, stacks_copy,
                state_copy, action_order, next_player,
                bet_sequence + ("call",),
                reach_copy, button_pos, depth + 1
            )

        else:  # raise_50
            # Raise: put in more chips — opponents must respond
            raise_count = sum(1 for a in bet_sequence if a == "raise_50")
            if raise_count >= MAX_RAISES_PER_STREET:
                # Raise cap hit — treat as a call instead
                action_values[action] = action_values.get("call", 0.0)
                continue

            # FIX 5: raise must be at least 2 × BIG_BLIND (legal minimum)
            raise_amount = max(int(pot * 0.5), BIG_BLIND * 2)
            raise_amount = min(raise_amount, my_stack)  # can't bet more than stack

            stacks_copy[acting_player] -= raise_amount
            state_copy.bet(acting_player, state_copy.max_bet + raise_amount)

            if raise_amount >= my_stack:
                state_copy.all_in[acting_player] = True

            next_player = state_copy.get_next_player(action_order, acting_player) \
                          or acting_player

            action_values[action] = cfr(
                hole_cards, board, street, pot + raise_amount, stacks_copy,
                state_copy, action_order, next_player,
                bet_sequence + ("raise_50",),
                reach_copy, button_pos, depth + 1
            )

    # ── Compute node expected value ───────────────────────────────────────────
    # EV = weighted average of all action values, weighted by action probabilities
    node_ev = sum(strategy[a] * action_values[a] for a in ACTIONS)

    # ── Update regret sums ────────────────────────────────────────────────────
    # Regret for action a = (value if we always took a) − (node EV)
    # Weighted by opponent reach: if opponents were unlikely to be here,
    # this node matters less and the update should be smaller.
    opponent_reach = 1.0
    for i in range(NUM_PLAYERS):
        if i != acting_player:
            opponent_reach *= reach_probs[i]

    my_reach = reach_probs[acting_player]

    for action in ACTIONS:
        regret = action_values[action] - node_ev
        regret_sum[info_key][action]   += opponent_reach * regret
        strategy_sum[info_key][action] += my_reach       * strategy[action]

    return node_ev

# =============================================================================
# SECTION 7 — SAVE AND LOAD THE BLUEPRINT
# =============================================================================

def save_blueprint(label=""):
    """
    Compute the average strategy for every info set and save to disk.

    The AVERAGE strategy (not the current strategy) is what converges to Nash.
    Current strategy oscillates during training; the average smooths over time.

    Saves as a compressed numpy archive (.npz) containing one dict:
      avg_strategy: {info_set_key → {action → probability}}
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    avg_strategy = {}
    for key in strategy_sum:
        total = sum(strategy_sum[key].values())
        if total > 0:
            avg_strategy[key] = {a: strategy_sum[key][a] / total for a in ACTIONS}
        else:
            avg_strategy[key] = {a: 1.0 / len(ACTIONS) for a in ACTIONS}

    np.savez_compressed(
        OUTPUT_FILE,
        avg_strategy=np.array([avg_strategy], dtype=object)
    )

    tag = f" [{label}]" if label else ""
    print(f"    ✓ Saved{tag} — {len(avg_strategy):,} info sets → {OUTPUT_FILE}")
    return len(avg_strategy)


def load_blueprint(path):
    """
    Load the blueprint file in bot.py at module import time.

    Usage in bot.py:
      import numpy as np, os
      DATA_DIR  = os.environ.get("BOT_DATA_DIR", "bots/mybot/data")
      BLUEPRINT = load_blueprint(os.path.join(DATA_DIR, "blueprint.npz"))

    In decide():
      key   = f"{preflop_group}|{bucket}|{texture}|{street}|{bets_str}|{position}"
      strat = BLUEPRINT.get(key)
      if strat:
          action = max(strat, key=strat.get)   # pick highest-probability action
      else:
          action = baseline_action(state, equity)   # fallback to heuristics
    """
    data = np.load(path, allow_pickle=True)
    return data["avg_strategy"][0]

# =============================================================================
# SECTION 8 — TRAINING LOOP
# =============================================================================

def train():
    """
    Main training loop. For each iteration:
      1. Deal a fresh 6-player hand
      2. Initialise game state WITH blinds recorded (FIX 2)
      3. Run CFR — explores the full game tree to showdown
      4. Every CHECKPOINT_EVERY iterations: print progress + save

    The button rotates each iteration so every position gets equal training.
    """
    global cfr_call_count

    print("=" * 70)
    print("  6-PLAYER CFR TRAINER  (FINAL — all bugs fixed)")
    print("=" * 70)
    print(f"  Players       : {NUM_PLAYERS}")
    print(f"  Iterations    : {NUM_ITERATIONS:,}")
    print(f"  Checkpoint    : every {CHECKPOINT_EVERY:,}")
    print(f"  Output        : {OUTPUT_FILE}")
    print(f"  Stop          : Ctrl+C saves and exits cleanly")
    print("=" * 70 + "\n")

    def on_interrupt(sig, frame):
        print("\n\n  Ctrl+C caught — saving progress...")
        save_blueprint("interrupted")
        print("  Done. Goodnight.")
        sys.exit(0)

    signal.signal(signal.SIGINT, on_interrupt)

    for iteration in range(1, NUM_ITERATIONS + 1):
        cfr_call_count = 0

        # ── Deal a fresh hand ─────────────────────────────────────────────────
        deck = eval7.Deck()
        deck.shuffle()
        hole_cards = [deck.cards[i * 2:(i + 1) * 2] for i in range(NUM_PLAYERS)]

        # ── Rotate button so every position gets equal training ───────────────
        button_pos = (iteration - 1) % NUM_PLAYERS
        sb_player  = (button_pos + 1) % NUM_PLAYERS
        bb_player  = (button_pos + 2) % NUM_PLAYERS

        # ── Set up stacks (deduct blinds) ─────────────────────────────────────
        stacks             = [STARTING_STACK] * NUM_PLAYERS
        stacks[sb_player] -= SMALL_BLIND
        stacks[bb_player] -= BIG_BLIND
        initial_pot        = SMALL_BLIND + BIG_BLIND

        # ── Initialise game state WITH blind bets recorded ────────────────────
        # FIX 2: previously GameState started empty (max_bet=0, no bets recorded)
        # which meant UTG could "call" for 0 chips — preflop never really happened.
        # Now we record SB and BB bets so the tree starts correctly.
        game_state = GameState(NUM_PLAYERS)
        game_state.bets_this_street[sb_player] = SMALL_BLIND
        game_state.bets_this_street[bb_player] = BIG_BLIND
        game_state.max_bet                     = BIG_BLIND   # UTG must match BB

        # Preflop action order: UTG → UTG+1 → MP → BTN → SB → BB
        action_order = [(button_pos + i) % NUM_PLAYERS for i in range(3, 3 + NUM_PLAYERS)]

        # ── Run CFR ───────────────────────────────────────────────────────────
        cfr(hole_cards, [], "preflop", initial_pot, stacks,
            game_state, action_order, action_order[0],
            (), [1.0] * NUM_PLAYERS, button_pos)

        # ── Checkpoint ───────────────────────────────────────────────────────
        if iteration % CHECKPOINT_EVERY == 0:
            pct      = 100 * iteration / NUM_ITERATIONS
            n_sets   = len(strategy_sum)
            print(f"  [{pct:5.1f}%]  iter {iteration:>7,}  |  "
                  f"info sets: {n_sets:>8,}  |  "
                  f"CFR calls this iter: {cfr_call_count/1e6:.2f}M")
            save_blueprint(f"{pct:.0f}%")

    print("\n  ✓ All iterations complete!")
    save_blueprint("FINAL")


if __name__ == "__main__":
    train()
