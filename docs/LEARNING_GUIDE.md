# Learning guide

Use this guide in order. Each lesson points to a small part of the implementation
and ends with a change you can make yourself.

## Lesson 1: Establish a known truth

Open `src/pairs_trading/data.py` and read `simulate_prices`.

`BBB` is a random walk. `AAA` uses the same stochastic trend plus a stationary
AR(1) residual. Their prices can wander indefinitely, but the modeled residual
stays bounded. `CCC` and `DDD` are distractors.

Run:

```bash
pairs-trading demo
```

Checkpoint: confirm that `AAA/BBB` appears in `outputs/demo/summary.csv`.

Exercise: change the residual persistence from `0.90` to `0.98`. Run the demo
again and explain why the estimated half-life increases.

## Lesson 2: Correlation does not define the spread

Open `src/pairs_trading/model.py` and locate `select_candidate_pairs`.

Correlation measures co-movement of returns. It does not guarantee that a linear
combination of price levels is stationary. Two unrelated trending price series
can have a high level correlation. This implementation:

1. Uses absolute log-return correlation as a computational prefilter.
2. Applies the Engle-Granger cointegration test to log-price levels.
3. Keeps pairs whose training-period p-value is below the chosen cutoff.

A p-value is not the probability that a pair is valid. Testing hundreds of pairs
at 5% significance will produce false positives. That is a later extension.

Exercise: set `--minimum-correlation 0` and inspect how many additional
cointegration tests run. Then add ten independent synthetic assets and observe
whether any false candidates appear.

## Lesson 3: Estimate the hedge ratio

`fit_pair` uses scikit-learn's `LinearRegression` to estimate:

```text
log(Y) = alpha + beta * log(X) + residual
```

The spread is the regression residual. `beta` controls relative leg exposure; it
is not merely the number of X shares to trade. The backtest converts it to gross-
normalized dollar weights:

```text
w_y = 1 / (1 + |beta|)
w_x = |beta| / (1 + |beta|)
```

Exercise: print the training spread mean and standard deviation, then verify that
the training z-score has approximately mean zero and standard deviation one.

## Lesson 4: Build a state machine

Open `src/pairs_trading/strategy.py`.

A naive expression such as `position = -sign(z)` changes the portfolio every day.
The state machine remembers whether it already holds a spread:

- Enter only after crossing an outer boundary.
- Hold while the divergence remains unresolved.
- Exit near equilibrium.
- Stop when the relationship moves implausibly far away.

Run:

```bash
pytest tests/test_strategy.py -q
```

Exercise: add a cooldown state that prevents re-entry until the z-score has first
returned inside the entry threshold after a stop.

## Lesson 5: Find the look-ahead trap

Open `src/pairs_trading/backtest.py` and compare `target_position` with
`held_position`.

At date `t`, the close creates `z_t`. That information can determine exposure for
the next return, but not the return that just occurred. Therefore:

```python
held = target.shift(1)
```

Exercise: temporarily remove the shift and compare performance. The difference
is an estimate of how much same-bar look-ahead can distort this experiment.
Restore the shift afterward.

## Lesson 6: Account for trading friction

Turnover is the absolute change in gross-normalized pair exposure. Entering or
exiting has turnover 1; flipping directly from long to short has turnover 2.
Modeled transaction cost and slippage are deducted in basis points.

Try:

```bash
pairs-trading demo --transaction-cost-bps 5 --slippage-bps 10
```

Exercise: plot total return across total costs of 0, 5, 10, 20, and 50 basis
points. Identify the strategy's approximate break-even cost.

## Lesson 7: Protect the test period

Open `src/pairs_trading/research.py`.

The chronological training window selects the pair and estimates every model
parameter. The test window is used once for evaluation. Random splitting would
mix later market regimes into earlier training data.

Exercise: implement expanding-window walk-forward testing:

1. Estimate using the first 252 observations.
2. Trade the next 21 observations.
3. Expand training by 21 observations and repeat.
4. Concatenate only the out-of-sample returns.

This is the highest-value next feature.

## Lesson 8: Use Alpaca safely

`src/pairs_trading/data.py` downloads adjusted bars. `execution.py` builds a
beta-adjusted plan and can submit it only through `TradingClient(..., paper=True)`.

First generate a dry run. Read each quantity and confirm that the two notionals
sum to the requested gross amount. Paper submission is still not atomic.

Exercise: write a paper-account reconciliation function that compares intended
quantities with filled quantities and reports leg imbalance. Do not add live
trading until you can safely handle partial fills and failed second legs.

## Suggested portfolio extensions

In priority order:

1. Walk-forward re-estimation with no overlapping training/test leakage
2. Benjamini-Hochberg false-discovery correction across pair tests
3. Rolling stability checks for beta, p-value, and spread half-life
4. Bid/ask quote data and an empirical slippage model
5. Portfolio allocation across pairs with symbol-level exposure limits
6. Paper fill monitoring, reconciliation, and a kill switch

For every extension, add a focused test and a short design note explaining what
bias or operational risk it addresses.
