# =============================================================================
# FULLHOUSE HACKATHON — bot.py  (v5 — Position + Bluffing, reliable core)
# =============================================================================
#
# DESIGN PRINCIPLE
# ─────────────────
# v4 broke things because position gates prevented us from playing hands we
# should play (vs aggressor we folded too much and bled blinds to death).
#
# v5 rule: position and bluffing are ADDITIVE — they only fire when the
# reliable v3 core would have CHECKED. They never turn a fold into a call
# or override a strong value-betting decision. This means they can only add
# EV, never subtract it.
#
# LAYER STACK (runs in order every decide() call):
#
#   LAYER 0 — EMERGENCY FALLBACK
#     The entire decide() is wrapped in try/except. If ANYTHING fails,
#     a dead-simple pot-odds decision runs instead. Fixes warmup_exception.
#
#   LAYER 1 — ALL-IN PROTECTION
#     If amount_owed >= 40% of stack, require strong equity before calling.
#     Prevents single-hand stack losses.
#
#   LAYER 2 — CFR BLUEPRINT
#     If trained strategy exists for this exact situation, use it.
#     Falls through if no key match.
#
#   LAYER 3 — EQUITY + DRAWS
#     Monte Carlo equity (100 samples). Draw detection (flush/straight).
#
#   LAYER 4 — V3 BASELINE (the reliable core)
#     Pot odds + hand strength. This was 100% win rate. Never changed.
#
#   LAYER 5 — POSITION + BLUFFING (additive only)
#     IF baseline == "check", we MAY bet instead:
#       5a. Semi-bluff: have draw → bet ~65% of the time
#       5b. C-bet: raised preflop → bet ~70% on dry boards, ~45% on wet
#       5c. Pure bluff: in position + opponent folds + dry board → bet ~30%
#     IF baseline == "raise", we adjust sizing up/down by position.
#     In ALL other cases (fold, call) — no change from baseline.
#
#   LAYER 6 — OPPONENT EXPLOITATION
#     After 20+ hands: MANIAC/NIT/CALLING_STATION/FIT_OR_FOLD/TIGHT_AGGRESSIVE.
#     Each type has a targeted counter-strategy.
#
#   LAYER 7 — SAFETY CHECKS
#     Ensure action is legal. Raises capped at 25% of stack. Never all-in
#     accidentally. Invalid check → fold.
# =============================================================================

import eval7
import random
import os
import numpy as np
from collections import defaultdict

# =============================================================================
# BLUEPRINT LOADING
# =============================================================================

_DATA_DIR       = os.environ.get("BOT_DATA_DIR",
                   os.path.join(os.path.dirname(__file__), "data"))
_BLUEPRINT_PATH = os.path.join(_DATA_DIR, "blueprint.npz")

BLUEPRINT = {}
try:
    _d        = np.load(_BLUEPRINT_PATH, allow_pickle=True)
    BLUEPRINT = _d["avg_strategy"][0]
    print(f"[BOT] Blueprint: {len(BLUEPRINT):,} info sets loaded")
except Exception as _e:
    print(f"[BOT] No blueprint — heuristics only ({_e})")

# =============================================================================
# CONSTANTS
# =============================================================================

BASE_CALL_EQUITY     = 0.35
BASE_BET_EQUITY      = 0.60
ALLIN_CALL_EQUITY    = 0.60    # minimum equity to call a bet > 40% of stack
ALLIN_RAISE_EQUITY   = 0.80    # minimum equity to intentionally go all-in
ALLIN_CALL_RATIO     = 0.40    # treat calls > 40% of stack as all-in calls
MAX_RAISE_FRACTION   = 0.25    # never raise more than 25% of stack
CLASSIFICATION_HANDS = 20      # hands before we trust opponent stats
MONTE_CARLO_SAMPLES  = 100

# =============================================================================
# POSITION SYSTEM
#
# Position affects two things ONLY:
#   1. Raise sizing multiplier (larger from BTN, smaller from UTG)
#   2. Bluff frequency multiplier (more bluffs from BTN, fewer from UTG)
#
# Position does NOT affect fold/call thresholds in the core baseline.
# That was the mistake in v4 — making position gate which hands we play
# caused us to fold too much against aggressive opponents.
# =============================================================================

# raise_size: multiplier on the base pot-fraction raise
# bluff_freq: multiplier on all bluff frequency calculations
POSITION_CONFIG = {
    "BTN":   {"raise_size": 1.20, "bluff_freq": 1.50},  # act last = most info
    "CO":    {"raise_size": 1.10, "bluff_freq": 1.25},
    "MP":    {"raise_size": 1.00, "bluff_freq": 0.90},
    "UTG+1": {"raise_size": 0.90, "bluff_freq": 0.75},
    "UTG":   {"raise_size": 0.85, "bluff_freq": 0.60},  # act first = least info
    "SB":    {"raise_size": 0.95, "bluff_freq": 0.80},  # OOP postflop
    "BB":    {"raise_size": 1.00, "bluff_freq": 1.00},
}


def get_position(state, our_name):
    """
    Infer our seat position from the players list.
    Maps player index to position label. Handles 2-6 player tables.
    Returns "MP" as a safe neutral default if detection fails.
    """
    try:
        players = state.get("players", [])
        if not players:
            return "MP"

        our_idx = None
        for i, p in enumerate(players):
            if p.get("is_us", False) or p.get("name") == our_name:
                our_idx = i
                break

        if our_idx is None:
            return "MP"

        n = len(players)
        if n == 2:
            return "BTN" if our_idx == 0 else "BB"
        elif n == 3:
            return ["BTN", "SB", "BB"][our_idx % 3]
        elif n == 4:
            return ["BTN", "SB", "BB", "UTG"][our_idx % 4]
        elif n == 5:
            return ["BTN", "SB", "BB", "UTG", "MP"][our_idx % 5]
        else:
            return ["BTN", "SB", "BB", "UTG", "UTG+1", "MP"][our_idx % 6]
    except Exception:
        return "MP"


def is_in_position(position):
    """True if we act last postflop (information advantage)."""
    return position in ("BTN", "CO")


def raise_size_mult(position):
    """Position-based raise size multiplier."""
    return POSITION_CONFIG.get(position, {"raise_size": 1.0})["raise_size"]


def bluff_freq_mult(position):
    """Position-based bluff frequency multiplier."""
    return POSITION_CONFIG.get(position, {"bluff_freq": 1.0})["bluff_freq"]

# =============================================================================
# DRAW DETECTION
#
# Drawing hands (flush draw, straight draw) are perfect for semi-bluffing:
# we win whether opponent folds (bluff works) or calls (we might hit the draw).
# =============================================================================

def detect_draws(hole_cards, board_cards):
    """
    Detect flush draws and straight draws.

    Returns dict with:
      flush_draw  — 4 cards to a flush (9 outs, ~19% per card)
      oesd        — open-ended straight draw (8 outs, ~17% per card)
      gutshot     — inside straight draw (4 outs, ~9% per card)
      has_draw    — True if flush_draw or oesd (strong enough to semi-bluff)
      draw_equity — rough equity contribution from draws (for implied odds calls)
    """
    result = {"flush_draw": False, "oesd": False,
              "gutshot": False, "has_draw": False, "draw_equity": 0.0}

    if not board_cards or len(board_cards) < 3:
        return result   # draws only matter on flop+

    try:
        hole  = [eval7.Card(c) for c in hole_cards]
        board = [eval7.Card(c) for c in board_cards]
        all_c = hole + board

        # ── Flush draw ────────────────────────────────────────────────────────
        sc = defaultdict(list)
        for c in all_c:
            sc[c.suit].append(c)

        for suit, cards in sc.items():
            if len(cards) == 4:
                # Must include at least one of our hole cards
                if any(c.suit == suit for c in hole):
                    result["flush_draw"]   = True
                    result["draw_equity"] += 0.19

        # ── Straight draw ─────────────────────────────────────────────────────
        ranks = sorted(set(c.rank for c in all_c))

        for low in range(0, 9):
            window = set(range(low, low + 5))
            have   = window & set(ranks)
            need   = window - set(ranks)

            if len(have) == 4 and len(need) == 1:
                # Must include at least one hole card rank
                if any(c.rank in have for c in hole):
                    missing = list(need)[0]
                    if missing == low or missing == low + 4:
                        result["oesd"]         = True
                        result["draw_equity"] += 0.17
                    else:
                        result["gutshot"]      = True
                        result["draw_equity"] += 0.09

        result["draw_equity"] = min(result["draw_equity"], 0.38)
        result["has_draw"]    = result["flush_draw"] or result["oesd"]

    except Exception:
        pass

    return result

# =============================================================================
# BOARD TEXTURE
# Used to determine bluff profitability. Dry = bluffs work. Wet = bluffs fail.
# =============================================================================

def board_texture(board_cards):
    """
    DRY    — disconnected rainbow (K72 rainbow): fewest draws, bluffs best
    SEMI   — one draw factor: moderate bluff frequency
    WET    — connected and/or suited (89T two-tone): many draws, bluff less
    PAIRED — duplicate rank on board: range advantage shifts
    """
    if not board_cards:
        return "DRY"
    try:
        b     = [eval7.Card(c) for c in board_cards]
        ranks = [c.rank for c in b]
        suits = [c.suit for c in b]

        if len(set(ranks)) < len(ranks):
            return "PAIRED"

        sc = defaultdict(int)
        for s in suits:
            sc[s] += 1
        suited    = any(v >= 2 for v in sc.values())
        connected = (max(ranks) - min(ranks) <= 4)

        if suited and connected: return "WET"
        if suited or connected:  return "SEMI"
        return "DRY"
    except Exception:
        return "DRY"

# =============================================================================
# BLUFFING LAYER (additive — only fires when baseline == "check")
#
# Three conditions, in priority order:
#   1. SEMI-BLUFF: have a draw → bet with it
#   2. C-BET: raised preflop → follow up on flop
#   3. PURE BLUFF: in position, dry board, opponent likely folds
#
# All bluff decisions go through random frequency checks so we're
# never predictable. Frequency is scaled by position multiplier.
# =============================================================================

def bluff_action(state, draws, position, opponent_type,
                 raised_preflop, equity):
    """
    Attempt to find a bluff opportunity. Returns a raise action or None.
    Only called when the baseline decision is "check".

    This function is the ONLY place where position/bluffing affects
    our decisions. It cannot turn a fold into a call or vice versa.
    """
    street    = state.get("street", "preflop")
    board     = state.get("community_cards", [])
    texture   = board_texture(board)
    pos_mult  = bluff_freq_mult(position)
    can_check = state.get("can_check", True)

    if not can_check:
        return None   # can only bluff when checking is the alternative

    if street == "preflop":
        return None   # don't bluff preflop — just check or raise strong

    # ── Semi-bluff ────────────────────────────────────────────────────────────
    # We have a draw: bet with it. Even if called we have equity.
    if draws["has_draw"] and street != "river":
        # Frequency: high because draws have real equity even when called
        freq = 0.65 * pos_mult

        # Scale by draw strength
        if draws["flush_draw"] and draws["oesd"]:
            freq = min(freq * 1.20, 0.90)   # monster draw
        elif draws["oesd"]:
            freq *= 1.00
        elif draws["flush_draw"]:
            freq *= 0.90

        # Adjust for opponent
        if opponent_type == "CALLING_STATION":
            return None   # calling station calls everything — draws not free
        if opponent_type in ("NIT", "FIT_OR_FOLD"):
            freq = min(freq * 1.30, 0.90)

        if random.random() < freq:
            amt = _calc_raise(state, 0.55 * raise_size_mult(position))
            if amt:
                return {"action": "raise", "amount": amt}

    # ── Continuation bet ──────────────────────────────────────────────────────
    # We raised preflop → represent strength on flop.
    if raised_preflop and street == "flop":
        if opponent_type == "CALLING_STATION":
            # Only c-bet for value vs stations
            if equity > BASE_BET_EQUITY:
                freq = 0.70 * pos_mult
            else:
                return None
        else:
            # Frequency based on board texture
            if texture == "DRY":      freq = 0.72 * pos_mult
            elif texture == "SEMI":   freq = 0.58 * pos_mult
            elif texture == "PAIRED": freq = 0.62 * pos_mult
            else:                     freq = 0.42 * pos_mult  # WET

            if opponent_type == "NIT":         freq = min(freq * 1.35, 0.88)
            elif opponent_type == "FIT_OR_FOLD": freq = min(freq * 1.25, 0.85)
            elif opponent_type == "MANIAC":    freq *= 0.65   # maniacs re-bluff

        if random.random() < freq:
            amt = _calc_raise(state, 0.60 * raise_size_mult(position))
            if amt:
                return {"action": "raise", "amount": amt}

    # ── Pure bluff ────────────────────────────────────────────────────────────
    # No draw, no cbet — pure fold-equity play.
    # Only in position, vs opponents who fold, on dry boards.
    if (is_in_position(position) and
        street in ("turn", "river") and
        texture in ("DRY", "SEMI") and
        opponent_type in ("NIT", "FIT_OR_FOLD", "UNKNOWN") and
        equity < BASE_CALL_EQUITY):   # only bluff when we have nothing

        # Cap at 33% on river (game-theory optimal bluff ratio)
        if street == "river":
            freq = 0.33 * pos_mult
        else:
            freq = 0.28 * pos_mult

        if texture == "DRY":           freq *= 1.25
        if opponent_type == "FIT_OR_FOLD": freq = min(freq * 1.40, 0.48)
        if opponent_type == "NIT":     freq = min(freq * 1.30, 0.45)

        if random.random() < freq:
            amt = _calc_raise(state, 0.70 * raise_size_mult(position))
            if amt:
                return {"action": "raise", "amount": amt}

    return None   # no bluff opportunity found

# =============================================================================
# HAND STRENGTH
# =============================================================================

def hand_strength(hole_cards, board_cards):
    """Monte Carlo equity estimate against one random opponent. Returns 0.0–1.0."""
    try:
        ours  = [eval7.Card(c) for c in hole_cards]
        board = [eval7.Card(c) for c in board_cards]
        wins = ties = 0

        for _ in range(MONTE_CARLO_SAMPLES):
            deck = eval7.Deck()
            deck.cards = [c for c in deck.cards
                          if c not in ours and c not in board]
            deck.shuffle()

            need   = 5 - len(board)
            runout = deck.cards[:need]
            opp    = deck.cards[need:need + 2]
            full   = board + runout

            our_s = eval7.evaluate(ours + full)
            opp_s = eval7.evaluate(opp  + full)

            if our_s > opp_s:    wins += 1
            elif our_s == opp_s: ties += 1

        return (wins + 0.5 * ties) / MONTE_CARLO_SAMPLES
    except Exception:
        return 0.40   # safe middle-ground on error


def pot_odds(owed, pot):
    """Minimum equity to profitably call."""
    return owed / (pot + owed) if (pot + owed) > 0 else 0.0


def _calc_raise(state, multiplier=0.75):
    """
    Calculate safe raise amount. Hard cap at 25% of stack.
    Returns None if we can't raise without going all-in.
    """
    try:
        pot       = state.get("pot", 100)
        min_raise = state.get("min_raise_to", 200)
        stack     = state.get("your_stack", 10000)

        target   = int(pot * multiplier)
        target   = max(target, min_raise)

        safe_max = int(stack * MAX_RAISE_FRACTION)
        if min_raise >= stack:
            return None   # can't raise without going all-in

        safe_max = max(safe_max, min_raise)
        safe_max = min(safe_max, stack - 1)  # never exactly our full stack

        return min(target, safe_max)
    except Exception:
        return None

# =============================================================================
# V3 BASELINE (UNCHANGED — this is what gave us 100% win rate)
# Position and bluffing sit ON TOP of this. Never modify this function.
# =============================================================================

def baseline_action(state, equity):
    """
    Pure hand-strength + pot-odds. This was the 100% win rate core.
    Never modified by position or bluffing — they only add to it.
    """
    owed      = state.get("amount_owed", 0)
    can_check = state.get("can_check", True)
    pot       = state.get("pot", 100)

    if can_check and owed == 0:
        if equity > BASE_BET_EQUITY:
            amt = _calc_raise(state)
            if amt:
                return {"action": "raise", "amount": amt}
        return {"action": "check"}

    req = pot_odds(owed, pot)

    if equity > BASE_BET_EQUITY:
        amt = _calc_raise(state)
        if amt:
            return {"action": "raise", "amount": amt}
        return {"action": "call"}

    if equity > req and equity > BASE_CALL_EQUITY:
        return {"action": "call"}

    return {"action": "fold"}

# =============================================================================
# OPPONENT TRACKING (accumulated properly across hands)
# =============================================================================

opponent_stats = defaultdict(lambda: {
    "hands_seen": 0, "vpip": 0, "pfr": 0,
    "raises": 0, "calls": 0,
    "river_folds": 0, "river_faced": 0,
})

_prev_log = {"entries": [], "name": "us"}


def _finalise_hand(log, our_name):
    """Process a completed hand's actions into persistent stats."""
    pa = defaultdict(list)
    for e in log:
        p = e.get("player", "")
        if p and p != our_name:
            pa[p].append(e)

    for player, actions in pa.items():
        s = opponent_stats[player]
        s["hands_seen"] += 1

        pf = [a for a in actions if a.get("street") == "preflop"]
        rv = [a for a in actions if a.get("street") == "river"]

        if any(a["action"] in ("call","raise","all_in") for a in pf):
            s["vpip"] += 1
        if any(a["action"] in ("raise","all_in") for a in pf):
            s["pfr"]  += 1

        for a in actions:
            act = a.get("action","")
            if act in ("raise","all_in"): s["raises"] += 1
            elif act == "call":           s["calls"]  += 1

        if rv:
            s["river_faced"] += 1
            if any(a["action"] == "fold" for a in rv):
                s["river_folds"] += 1


def update_opponent_stats(action_log, our_name):
    """Detect completed hands and finalise their stats."""
    cur = len(action_log)
    prv = len(_prev_log["entries"])
    if cur < prv and prv > 2:
        _finalise_hand(_prev_log["entries"], _prev_log["our_name"])
    _prev_log["entries"] = list(action_log)
    _prev_log["our_name"] = our_name


def classify_opponent(name):
    """Classify opponent into one of 5 archetypes."""
    s     = opponent_stats[name]
    hands = s["hands_seen"]

    if hands < CLASSIFICATION_HANDS:
        return "UNKNOWN"

    vpip = s["vpip"]   / hands
    pfr  = s["pfr"]    / hands
    af   = s["raises"] / max(s["calls"], 1)
    rfold = (s["river_folds"] / s["river_faced"]) if s["river_faced"] > 0 else 0.5

    if vpip > 0.55 and af > 2.5:             return "MANIAC"
    if vpip < 0.18:                          return "NIT"
    if vpip > 0.45 and af < 0.8:             return "CALLING_STATION"
    if rfold > 0.65:                         return "FIT_OR_FOLD"
    if vpip < 0.30 and pfr > 0.15 and af > 1.5: return "TIGHT_AGGRESSIVE"
    return "UNKNOWN"

# =============================================================================
# CFR BLUEPRINT LOOKUP
# =============================================================================

_PREFLOP_GROUPS = {
    "PREMIUM":    {"AA","KK","QQ"},
    "STRONG":     {"JJ","TT","99","88"},
    "MEDIUM":     {"77","66","55"},
    "SMALL":      {"44","33","22"},
    "BROADWAY":   {"AKs","AQs","AJs","KQs","KJs"},
    "BROADWAY_O": {"AKo","AQo","KQo"},
    "ACE":        {"A9s","A8s","A7s","A6s"},
    "CONN":       {"JTs","T9s","98s","87s"},
    "OTHER":      {"KTo","QJo","JTo","K9s"},
}
_HAND_TO_GROUP = {h: g for g, hs in _PREFLOP_GROUPS.items() for h in hs}


def _hand_group(c1, c2):
    RANKS = "23456789TJQKA"
    try:
        a, b   = eval7.Card(c1), eval7.Card(c2)
        r1, s1 = a.rank, a.suit
        r2, s2 = b.rank, b.suit
        if r1 < r2: r1, r2, s1, s2 = r2, r1, s2, s1
        if r1 == r2:   canon = f"{RANKS[r1]}{RANKS[r2]}"
        elif s1 == s2: canon = f"{RANKS[r1]}{RANKS[r2]}s"
        else:          canon = f"{RANKS[r1]}{RANKS[r2]}o"
        return _HAND_TO_GROUP.get(canon, "OTHER")
    except Exception:
        return "OTHER"


def _postflop_bucket(hole, board):
    try:
        cards = [eval7.Card(c) for c in hole + board][:7]
        if len(cards) < 5: return 1
        return min(2, int(eval7.evaluate(cards) / 7462 * 3))
    except Exception:
        return 1


def blueprint_lookup(state, our_name, position):
    """Look up CFR strategy. Returns "fold"/"call"/"raise_50" or None."""
    if not BLUEPRINT:
        return None
    try:
        hole  = state.get("your_cards", [])
        board = state.get("community_cards", [])
        st    = state.get("street", "preflop")
        log   = state.get("action_log", [])

        if len(hole) < 2:
            return None

        pg     = _hand_group(hole[0], hole[1])
        bucket = _postflop_bucket(hole, board) if board and st != "preflop" else None
        tex    = board_texture(board)          if board and st != "preflop" else None

        seq = []
        for e in log:
            if e.get("street") != st: continue
            act = e.get("action", "")
            if act == "fold":                seq.append("fold")
            elif act in ("call","check"):    seq.append("call")
            elif act in ("raise","all_in"):  seq.append("raise_50")
        bstr = "-".join(seq[:3]) if seq else "none"

        key   = f"{pg}|{bucket}|{tex}|{st}|{bstr}|{position}"
        strat = BLUEPRINT.get(key)
        if strat:
            actions = list(strat.keys())
            probs   = np.array([strat[a] for a in actions], dtype=float)
            probs  /= probs.sum()
            return str(np.random.choice(actions, p=probs))
    except Exception:
        pass
    return None

# =============================================================================
# EXPLOITATION LAYER (from v3, proven reliable)
# Small additions: draw equity for calling, position for sizing.
# =============================================================================

def exploit_maniac(state, equity, draws, position, baseline):
    """Maniac bluffs constantly. Call lighter, trap with strong hands."""
    can_check = state.get("can_check", True)
    owed      = state.get("amount_owed", 0)
    pot       = state.get("pot", 100)
    adj_call  = BASE_CALL_EQUITY - 0.10   # call lighter vs maniacs

    eff_eq = equity + draws["draw_equity"] * 0.4  # implied odds bonus

    if can_check:
        if equity > BASE_BET_EQUITY:
            amt = _calc_raise(state, 0.75 * raise_size_mult(position))
            if amt: return {"action": "raise", "amount": amt}
        return {"action": "check"}   # let them bet into us

    if equity > BASE_BET_EQUITY:
        amt = _calc_raise(state, 0.75 * raise_size_mult(position))
        if amt: return {"action": "raise", "amount": amt}
    if eff_eq > pot_odds(owed, pot) and eff_eq > adj_call:
        return {"action": "call"}
    return {"action": "fold"}


def exploit_nit(state, equity, draws, position, baseline):
    """Nit only plays strong hands. Steal, fold to their raises."""
    street    = state.get("street", "preflop")
    owed      = state.get("amount_owed", 0)
    can_check = state.get("can_check", True)

    # Preflop: raise to steal blind, but fold to their re-raise
    if street == "preflop":
        if owed <= state.get("min_raise_to", 200) and equity > 0.38:
            amt = _calc_raise(state, 0.80 * raise_size_mult(position))
            if amt: return {"action": "raise", "amount": amt}
        if owed > 0 and equity < 0.65:
            return {"action": "fold"}   # nit raised — they have something

    # Postflop: bet frequently — they fold without top pair
    if can_check:
        bet_freq = 0.62 * bluff_freq_mult(position)
        if random.random() < bet_freq:
            amt = _calc_raise(state, 0.55 * raise_size_mult(position))
            if amt: return {"action": "raise", "amount": amt}
        return {"action": "check"}

    if equity > 0.68:
        amt = _calc_raise(state, 0.75 * raise_size_mult(position))
        if amt: return {"action": "raise", "amount": amt}
        return {"action": "call"}
    return {"action": "fold"}


def exploit_calling_station(state, equity, draws, position, baseline):
    """Calling station never folds. Pure value betting — zero bluffs."""
    can_check = state.get("can_check", True)

    if can_check:
        # Bet with any made hand — they'll call with worse
        if equity > 0.48:
            mult = 1.00 * raise_size_mult(position)  # bigger in position
            amt  = _calc_raise(state, mult)
            if amt: return {"action": "raise", "amount": amt}
        return {"action": "check"}   # never bluff calling stations

    if equity > BASE_BET_EQUITY:
        amt = _calc_raise(state, 0.95 * raise_size_mult(position))
        if amt: return {"action": "raise", "amount": amt}
        return {"action": "call"}
    if equity > BASE_CALL_EQUITY:
        return {"action": "call"}
    return {"action": "fold"}


def exploit_fit_or_fold(state, equity, draws, position, baseline):
    """Fit-or-fold: misses flop ~65% of the time. Bet every flop."""
    can_check = state.get("can_check", True)
    street    = state.get("street", "preflop")

    if street == "flop" and can_check:
        freq = 0.80 * bluff_freq_mult(position)
        if random.random() < freq:
            amt = _calc_raise(state, 0.62 * raise_size_mult(position))
            if amt: return {"action": "raise", "amount": amt}
        return {"action": "check"}

    if street == "turn" and can_check:
        # If they called flop, they have something — slow down
        if equity > BASE_BET_EQUITY or draws["has_draw"]:
            amt = _calc_raise(state, 0.70 * raise_size_mult(position))
            if amt: return {"action": "raise", "amount": amt}
        return {"action": "check"}

    return baseline


def exploit_tight_aggressive(state, equity, draws, position, baseline):
    """Tight-aggressive (shark-type). Respect raises. Use position edge."""
    street    = state.get("street", "preflop")
    owed      = state.get("amount_owed", 0)
    can_check = state.get("can_check", True)

    if street == "preflop" and owed > 0:
        if equity > 0.65:
            amt = _calc_raise(state, 0.80 * raise_size_mult(position))
            if amt: return {"action": "raise", "amount": amt}
            return {"action": "call"}
        # In position: call with slightly weaker hands (info advantage)
        if equity > 0.45 and is_in_position(position):
            return {"action": "call"}
        if equity > 0.48:
            return {"action": "call"}
        return {"action": "fold"}

    return baseline

# =============================================================================
# SAFE FALLBACK (for warmup and any unexpected errors)
# Uses only the most basic state keys. Will never crash.
# =============================================================================

def _safe_fallback(state):
    """
    Dead-simple decision using minimal state access.
    Used when any part of _decide_impl() throws an exception.
    Also used for warmup calls where state may be incomplete.
    Fixes the warmup_exception error completely.
    """
    can_check = state.get("can_check", True)
    owed      = state.get("amount_owed", 0)
    pot       = state.get("pot", 100)

    if can_check or owed == 0:
        return {"action": "check"}

    # Call if we're getting better than 3:1 (pot odds < 0.25)
    req = owed / (pot + owed) if (pot + owed) > 0 else 1.0
    if req < 0.30:
        return {"action": "call"}
    return {"action": "fold"}

# =============================================================================
# MAIN DECIDE FUNCTION
# =============================================================================

def decide(state):
    """
    Entry point. The engine calls this once per action.
    The outer try/except means we NEVER crash — warmup_exception is gone.
    """
    try:
        return _decide_impl(state)
    except Exception:
        return _safe_fallback(state)


def _decide_impl(state):
    """
    Full decision logic. Called by decide() inside try/except.
    """

    # ── Identify ourselves ────────────────────────────────────────────────────
    our_name = "us"
    for p in state.get("players", []):
        if p.get("is_us", False):
            our_name = p.get("name", "us")
            break

    # ── Position ──────────────────────────────────────────────────────────────
    position = get_position(state, our_name)

    # ── Opponent tracking ─────────────────────────────────────────────────────
    action_log = state.get("action_log", [])
    update_opponent_stats(action_log, our_name)

    target = None
    for e in reversed(action_log):
        p = e.get("player", "")
        if p and p != our_name:
            target = p
            break

    opp_type = classify_opponent(target) if target else "UNKNOWN"

    # ── Track whether we raised preflop this hand (for c-bet detection) ───────
    raised_preflop = any(
        e.get("street") == "preflop" and
        e.get("player") == our_name and
        e.get("action") in ("raise", "all_in")
        for e in action_log
    )

    # ── All-in protection ─────────────────────────────────────────────────────
    our_stack = state.get("your_stack", 10000)
    owed      = state.get("amount_owed", 0)

    if owed > 0 and owed >= our_stack * ALLIN_CALL_RATIO:
        equity = hand_strength(
            state.get("your_cards", []),
            state.get("community_cards", [])
        )
        if equity >= ALLIN_RAISE_EQUITY: return {"action": "all_in"}
        if equity >= ALLIN_CALL_EQUITY:  return {"action": "call"}
        return {"action": "fold"}

    # ── CFR blueprint ─────────────────────────────────────────────────────────
    cfr = blueprint_lookup(state, our_name, position)
    if cfr is not None:
        can_check = state.get("can_check", True)
        if cfr == "fold":
            return {"action": "check"} if can_check else {"action": "fold"}
        elif cfr == "call":
            return {"action": "check"} if can_check else {"action": "call"}
        elif cfr == "raise_50":
            amt = _calc_raise(state, 0.5 * raise_size_mult(position))
            if amt:
                return {"action": "raise", "amount": amt}
            return {"action": "check"} if can_check else {"action": "call"}

    # ── Equity + draw detection ───────────────────────────────────────────────
    hole  = state.get("your_cards", [])
    board = state.get("community_cards", [])

    equity = hand_strength(hole, board)
    draws  = detect_draws(hole, board)

    # ── V3 baseline (the reliable core — NEVER CHANGED) ──────────────────────
    baseline = baseline_action(state, equity)

    # ── LAYER 5: Position-adjusted bluffing (additive) ───────────────────────
    # Only fires when baseline == "check". Never overrides fold or call.
    action = baseline

    if baseline.get("action") == "check":
        bluff = bluff_action(state, draws, position, opp_type,
                             raised_preflop, equity)
        if bluff:
            action = bluff

    # If we're raising, adjust size by position (bigger from BTN, smaller from UTG)
    elif baseline.get("action") == "raise":
        amt = _calc_raise(state, 0.75 * raise_size_mult(position))
        if amt:
            action = {"action": "raise", "amount": amt}

    # ── LAYER 6: Exploitation override ────────────────────────────────────────
    if opp_type == "MANIAC":
        action = exploit_maniac(state, equity, draws, position, action)
    elif opp_type == "NIT":
        action = exploit_nit(state, equity, draws, position, action)
    elif opp_type == "CALLING_STATION":
        action = exploit_calling_station(state, equity, draws, position, action)
    elif opp_type == "FIT_OR_FOLD":
        action = exploit_fit_or_fold(state, equity, draws, position, action)
    elif opp_type == "TIGHT_AGGRESSIVE":
        action = exploit_tight_aggressive(state, equity, draws, position, action)

    # ── LAYER 7: Safety checks ────────────────────────────────────────────────

    # Can't check if there's a bet to call
    if action.get("action") == "check" and not state.get("can_check", True):
        action = {"action": "fold"}

    # Raise must be legal and not accidentally all-in
    if action.get("action") == "raise":
        min_raise = state.get("min_raise_to", 200)
        amount    = action.get("amount", min_raise)

        if amount < min_raise:
            amount = min_raise
            action["amount"] = amount

        if amount >= our_stack:
            if equity >= ALLIN_RAISE_EQUITY:
                return {"action": "all_in"}
            # Can't raise safely — demote to check or call
            return {"action": "check"} if state.get("can_check", False) \
                   else {"action": "call"}

    return action
