# Fullhouse Hackathon — Poker Bot

## Architecture
A 6-layer poker bot built for the Fullhouse hackathon.

- **Layer 0** — CFR Blueprint (pre-trained strategy table)
- **Layer 1** — All-in protection (never commit >40% of stack without 60%+ equity)
- **Layer 2** — Monte Carlo equity estimation (100 simulations per decision)
- **Layer 3** — V3 baseline (pot odds + hand strength)
- **Layer 4** — Semi-bluff system (flush draw / OESD only)
- **Layer 5** — Opponent classification (MANIAC/NIT/CALLING_STATION/FIT_OR_FOLD/TAG)
- **Layer 6** — Position-aware sizing (BTN raises 20% bigger than UTG)

## Results
100% win rate against all reference bots in local testing.

## Files
- `bots/mybot/bot.py` — the main bot
- `precompute_cfr.py` — CFR training script (run overnight to generate blueprint)
- `bots/mybot/data/blueprint.npz` — pre-trained CFR strategy table (12,550 info sets)

## How to run
```bash
brew install python@3.10
cd fullhouse-engine
/opt/homebrew/opt/python@3.10/bin/python3.10 -m venv venv
source venv/bin/activate
pip install "Cython<3"
pip install --no-build-isolation eval7==0.1.7
pip install flask numpy scipy treys scikit-learn

# Test your bot
python3 sandbox/match.py bots/mybot/bot.py bots/shark/bot.py --hands 400

# Retrain CFR overnight
caffeinate -i python3 precompute_cfr.py
```
