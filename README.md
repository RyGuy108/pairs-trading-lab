# Pairs Trading Lab

A project that finds related stocks, models their equilibrium relationship, 
trades temporary spread divergences, and evaluates the result on unseen data.

The project is designed to teach research discipline—not to manufacture an
impressive backtest. It includes deterministic sample data, pair discovery,
cointegration testing, a one-bar-delayed backtester, transaction costs,
diagnostic plots, and optional Alpaca data and **paper-only** order submission.

> This is educational software, not investment advice. A profitable historical
> simulation does not establish that a strategy will remain profitable.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all]"
pytest
pairs-trading demo
```

Or use the convenience targets:

```bash
make setup
make check
make demo
make walkthrough
```

The demo writes:

```text
outputs/demo/
├── synthetic_prices.csv
├── summary.csv
├── aaa_bbb_history.csv
└── aaa_bbb_diagnostics.png
```

Also run the narrated code walkthrough:

```bash
python examples/walkthrough.py
```

## Research workflow

```text
wide price data
      │
      ├─ chronological split ───────────────┐
      │                                     │
      ▼                                     ▼
training window                        test window
      │                                     │
return-correlation screen                   │
      │                                     │
Engle-Granger cointegration test            │
      │                                     │
regression + frozen spread parameters ──────┘
                                            │
                                      delayed signals
                                            │
                                  costs + performance report
```

All selection and estimation occur in the training window. Only frozen model
parameters are applied to the test window.

## Mathematical model

For stocks `Y` and `X`, the training regression is:

```text
log(Y_t) = alpha + beta * log(X_t) + epsilon_t
```

The residual is the spread:

```text
spread_t = log(Y_t) - alpha - beta * log(X_t)
z_t      = (spread_t - training_mean) / training_std
```

If the spread is cointegrated, it may revert toward its historical equilibrium.
The default state machine:

- `z >= +2`: short the spread—short Y and buy the beta-adjusted X leg
- `z <= -2`: long the spread—buy Y and short the beta-adjusted X leg
- `|z| <= 0.5`: exit
- `|z| >= 4`: stop out or refuse a new entry

The next bar holds the position selected from the current bar's z-score. This
delay is essential: a strategy cannot observe a closing price and also earn the
return that ended at that same close.

## Use your own CSV

Input files are wide, with a timestamp in the first column:

```csv
timestamp,KO,PEP,WMT,COST
2022-01-03,58.44,169.61,144.65,558.58
2022-01-04,59.18,171.15,142.00,564.22
```

Run:

```bash
pairs-trading research \
  --csv data/downloaded/prices.csv \
  --output-dir outputs/real-data \
  --train-fraction 0.65 \
  --minimum-correlation 0.55 \
  --maximum-pvalue 0.05
```

Do not choose thresholds after examining the test results. Use another validation
window or walk-forward experiment when tuning parameters.

## Alpaca integration

The project uses the maintained `alpaca-py` interfaces described in Alpaca's
[market-data documentation](https://alpaca.markets/sdks/python/market_data.html)
and [trading documentation](https://alpaca.markets/sdks/python/trading.html).
The historical-data client and trading client are separate.

1. Copy `.env.example` to `.env` and add paper-account credentials.
2. Download adjusted daily bars from the free IEX feed:

```bash
pairs-trading download \
  --symbols KO PEP WMT COST HD LOW \
  --start 2020-01-01 \
  --end 2025-01-01 \
  --output data/downloaded/consumer_stocks.csv
```

3. Research the downloaded universe:

```bash
pairs-trading research \
  --csv data/downloaded/consumer_stocks.csv \
  --output-dir outputs/consumer-research
```

4. Generate an order plan without submitting it:

```bash
pairs-trading paper-plan \
  --csv data/downloaded/consumer_stocks.csv \
  --symbol-y KO \
  --symbol-x PEP \
  --gross-notional 1000
```

Add `--submit-paper` only after inspecting the plan. The implementation hardcodes
`paper=True`; it contains no live-trading mode. Alpaca market orders use typed
request objects and fractional quantities, following the official
[order API](https://alpaca.markets/sdks/python/api_reference/trading/orders.html).

Pair legs are not atomic. One order may fill while the other fails or fills later.
Production execution would need fill monitoring, short-availability checks,
reconciliation, retry policy, and an emergency flattening procedure.

## Project layout

```text
src/pairs_trading/
├── data.py        # validation, simulation, Alpaca historical bars
├── model.py       # correlation, cointegration, regression, half-life
├── strategy.py    # stateful z-score rules
├── backtest.py    # delayed execution, costs, metrics
├── research.py    # chronological train/test experiment
├── reporting.py   # CSVs and diagnostic plots
├── execution.py   # sizing and paper-only Alpaca submission
└── cli.py         # reproducible commands
```

## Important limitations

- A single train/test split does not establish robustness.
- Cointegration relationships can break structurally.
- Daily closing bars hide intraday execution and bid/ask behavior.
- The backtest uses modeled costs, not actual quotes or fills.
- Multiple pair tests create false discoveries unless corrected.
- The pair portfolio is not yet constrained at an aggregate risk level.
- Alpaca paper fills are simulations and will not reproduce live execution.

The next worthwhile improvements are in the exercises, especially walk-forward
re-estimation, false-discovery control, and quote-based cost modeling.
