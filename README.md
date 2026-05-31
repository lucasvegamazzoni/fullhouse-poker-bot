# Fullhouse Poker Bot — Hackathon Project

## What is this?
A competitive poker bot built for the Fullhouse Hackathon (May 2026).
The bot achieved a **100% win rate** against all reference bots in local testing.

## The Problem
Build a poker bot that can beat 5 reference bots in 400-hand matches
within a 2-second decision time limit, 768MB RAM, and 200MB data cap.

## My Approach
Rather than vibe-coding a simple heuristic bot, I built a 7-layer
decision architecture that combines game theory, statistics, and
opponent psychology.

### Layer 0 — CFR Blueprint (Game Theory)
Pre-computed a Counterfactual Regret Minimisation (CFR) strategy table
offline using self-play. CFR is the same algorithm used by world-class
poker AIs like Libratus and Claudico. The bot trained for 16+ hours,
exploring complete hands from preflop through to showdown, building a
Nash Equilibrium approximation stored in `data/blueprint.npz`.

### Layer 1 — All-In Protection
Never commit more than 40% of stack without 60%+ equity.
Prevents catastrophic single-hand losses.

### Layer 2 — Monte Carlo Equity Estimation
Simulates 100 random runouts per decision to estimate win probability.
Accurate enough to inform every decision, fast enough to stay within 2s.

### Layer 3 — Pot Odds Baseline
The mathematical foundation: only call when equity beats the pot odds.
fold/call/raise thresholds derived from hand strength and bet sizing.

### Layer 4 — Semi-Bluff System
Detects flush draws (9 outs) and open-ended straight draws (8 outs).
Bets these draws ~62% of the time on the flop — profitable whether
the opponent folds (win now) or calls (equity to win later).

### Layer 5 — Opponent Classification
Tracks VPIP, PFR, aggression factor, and river fold rate across hands.
After 20 hands, classifies each opponent into one of 5 archetypes:
- **MANIAC** — over-aggressive, bluffs constantly → call lighter
- **NIT** — ultra-tight, folds too much → steal their blinds
- **CALLING STATION** — never folds → never bluff, value bet everything
- **FIT OR FOLD** — bets when they hit, folds when they miss → bet every flop
- **TIGHT AGGRESSIVE** — disciplined, shark-type → respect their raises

### Layer 6 — Position-Aware Sizing
Detects seat position (BTN/CO/MP/UTG/SB/BB).
BTN raises 20% larger (acting last = more information = extract more value).
UTG raises 15% smaller (acting first = more caution).

## Results
| Opponent | Wins | Losses | Win Rate |
|---|---|---|---|
| Shark | 3/3 | 0 | 100% |
| Aggressor | 3/3 | 0 | 100% |
| Mathematician | 3/3 | 0 | 100% |
| ref_bot_2 | 3/3 | 0 | 100% |
| **Total** | **12/12** | **0** | **100%** |

## Key Technical Decisions

**Why CFR instead of pure heuristics?**
Heuristics have a ceiling — a smart opponent can figure out your
pattern and exploit it. CFR produces an unexploitable baseline.
Even with limited training (~3,800 iterations), the strategy table
provides signal on common preflop and flop spots.

**Why Monte Carlo and not a lookup table?**
A full equity lookup table would take gigabytes. Monte Carlo at 100
samples takes ~0.3 seconds and is accurate to within 2-3%.

**Why semi-bluffs only (not pure bluffs)?**
Pure bluffs were tested but caused losses against pot-odds callers
(mathematician, ref_bot_2) who call correctly based on pot odds.
Semi-bluffs have equity even when called — they're profitable either way.

## How to Run
```bash
# Install (requires Python 3.10)
brew install python@3.10
cd fullhouse-engine
/opt/homebrew/opt/python@3.10/bin/python3.10 -m venv venv
source venv/bin/activate
pip install "Cython<3"
pip install --no-build-isolation eval7==0.1.7
pip install flask numpy scipy treys scikit-learn

# Test the bot
python3 sandbox/match.py bots/mybot/bot.py bots/shark/bot.py --hands 400

# Re-train the CFR blueprint overnight
caffeinate -i python3 precompute_cfr.py
```

## Tech Stack
- Python 3.10
- eval7 (hand evaluation)
- numpy (CFR strategy storage)
- Flask (demo UI)

## What I Would Do With More Time
- Run CFR for 500,000+ iterations (needs ~1 week of compute)
- Add multi-way pot handling (currently optimised for heads-up)
- Implement full river bluffing with polarised ranges
- Add stack-depth awareness (push/fold below 20bb)
