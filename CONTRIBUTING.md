# Contributing

Contributions should improve the educational value, statistical validity, or
operational safety of the project. A higher backtested return by itself is not
evidence that a change is an improvement.

## Local setup

```bash
git clone YOUR_REPOSITORY_URL
cd pairs-trading-lab
make setup
make check
```

Without `make`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all]"
python -m ruff check .
python -m pytest
python -m build
```

## Development workflow

1. Create a focused branch from `main`.
2. Add or update tests for behavioral changes.
3. Explain any statistical assumptions in the relevant docstring or document.
4. Run `make check` before opening a pull request.
5. Keep generated market data, API keys, and research outputs out of Git.

## Research requirements

Changes to strategy logic should document:

- Which data is available at the decision timestamp
- Which parameters are estimated in sample
- Which observations are genuinely out of sample
- How turnover and execution costs are modeled
- Whether multiple hypotheses or parameter combinations were tried
- What new failure modes the change introduces

Do not tune a feature against the existing test period and continue reporting
that same period as unseen performance. Create a new validation period or use a
proper walk-forward design.

## Pull requests

Keep pull requests narrow enough to review. Include:

- The problem being solved
- The design choice and alternatives considered
- Tests performed
- Before/after research results when relevant
- Limitations that remain

Never include Alpaca keys or any other credential in an issue, commit, test
fixture, screenshot, or pull request.
