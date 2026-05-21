# Contributing

This is the official engine for Fullhouse Hackathon. The engine itself is frozen before the event — we don't want rule changes mid-competition.

## What you CAN contribute

- Bug reports (open an issue)
- Improvements to the local demo UI
- Additional reference bots in `bots/`
- Documentation improvements

## What stays frozen

- `engine/game.py` — the poker rules
- `sandbox/runner.py` — the bot protocol
- `db/schema.sql` — the database schema

## Adding a reference bot

Copy `bots/template/bot.py` into a new folder under `bots/`, implement `decide()`, and open a PR. Good reference bots that demonstrate interesting strategies are welcome.

## Need a Python library that isn't preinstalled?

The sandbox container ships with `eval7`, `numpy`, `scipy`, `treys`, and `scikit-learn` plus the full Python 3.10 stdlib. If you need something else, **open a GitHub Issue with the title prefix `[library-request]`** by **25 May 2026 (23:59 UTC)**. We will review batched requests once and rebuild the sandbox image before the 1 June qualifier.

Requests after 25 May 2026 will not make it in. Plan accordingly.

What we'll generally accept:
- Numerical/scientific (e.g. `pandas`, `networkx`, `scikit-learn` extensions)
- Algorithm/data-structure libraries with no network or filesystem behaviour

What we'll generally reject:
- Anything that needs network access (`requests`, `httpx`, etc. are blocked at the OS level anyway)
- Heavy ML training frameworks (`pytorch`, `tensorflow`, `jax`) — train offline, ship weights as `.npz` in `data/`
- Native binaries that can't be `pip install`-ed cleanly

## Questions

Open an issue or reach out via [fullhousehackathon.com](https://fullhousehackathon.com).
