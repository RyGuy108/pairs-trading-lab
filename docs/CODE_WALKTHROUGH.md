# Complete code walkthrough

This document explains the reasoning and implementation from raw prices to a
paper-order plan. Read it beside the source files and run the examples as you go.

## 1. The central idea

Pairs trading does not try to predict whether the entire market rises tomorrow.
It tries to identify two securities with a historically stable relationship and
trade temporary deviations from that relationship.

Suppose stocks `Y` and `X` are exposed to similar economic forces. Their prices
may both trend, which means neither price is stationary. A particular linear
combination of their log prices may nevertheless fluctuate around a stable mean.
That is the relationship this project models.

The algorithm has four distinct jobs:

1. **Discover:** find plausible pairs using training data.
2. **Estimate:** calculate the hedge ratio and normal spread behavior.
3. **Trade:** enter when the spread is unusually far from equilibrium and exit
   when it reverts.
4. **Evaluate:** apply frozen parameters to unseen data with realistic timing and
   modeled trading costs.

Keeping these jobs separate is one of the most important design choices. If the
test period helps choose the pair, hedge ratio, or thresholds, it is no longer a
true test period.

## 2. End-to-end program flow

Running `pairs-trading demo` follows this path:

```text
cli.main
  └─ _run_research_command
       ├─ data.simulate_prices
       ├─ data.save_prices
       ├─ research.run_research
       │    ├─ research.chronological_split
       │    ├─ model.select_candidate_pairs
       │    │    └─ model.fit_pair
       │    └─ backtest.run_backtest
       │         └─ strategy.generate_target_positions
       └─ reporting.write_research_outputs
            └─ reporting.plot_backtest
```

The package uses a `src/` layout. Importable code lives under
`src/pairs_trading`, while `tests/`, `examples/`, and `docs/` remain outside the
installed package.

## 3. Statistics and financial logic

### 3.1 Prices versus returns

Price series usually trend. A stock at $100 can become $120 without having a
stable long-run mean of $110. Daily returns are closer to stationary and are more
appropriate for measuring short-run co-movement.

The project therefore uses:

- **Log returns** for the initial correlation screen
- **Log-price levels** for the cointegration relationship

For price `P_t`, the one-period log return is:

```text
r_t = log(P_t) - log(P_{t-1})
```

Log prices make multiplicative price changes additive and let the regression
describe a proportional relationship rather than a fixed-dollar relationship.

### 3.2 Correlation is only a screen

`select_candidate_pairs` calculates the correlation between the two log-return
series. It discards pairs whose absolute correlation is below the configured
threshold.

Correlation answers: “Do these returns often move together?” It does **not**
answer: “Does their price relationship return to equilibrium?” Two assets can be
highly correlated yet drift permanently apart. That is why the code does not
trade correlation by itself.

The correlation step is a computational prefilter. The cointegration test is the
statistical test of the spread relationship.

### 3.3 Hedge-ratio regression

For each surviving pair, `fit_pair` estimates:

```text
log(Y_t) = alpha + beta * log(X_t) + epsilon_t
```

The implementation uses scikit-learn:

```python
regression = LinearRegression().fit(
    log_x.to_numpy().reshape(-1, 1),
    log_y.to_numpy(),
)
intercept = regression.intercept_
hedge_ratio = regression.coef_[0]
```

`LinearRegression` expects a two-dimensional feature matrix, so the one `X`
series is reshaped into `(number_of_rows, 1)`.

Interpretation:

- `alpha` is the fitted intercept.
- `beta` is the sensitivity of `log(Y)` to `log(X)`.
- `epsilon_t` is the part of `Y` not explained by the fitted relationship.

The residual becomes the spread:

```text
spread_t = log(Y_t) - alpha - beta * log(X_t)
```

Notice that this is not simply `Y - X`. Assets can have different price levels
and different proportional responses. Regression estimates that scaling.

### 3.4 Cointegration test

`statsmodels.tsa.stattools.coint` performs the Engle-Granger test. Its null
hypothesis is that the two series are **not** cointegrated. A small p-value is
evidence against that null on the training sample.

The default maximum p-value is 0.05. This is not a guarantee that a pair is valid,
and it does not mean there is a 95% probability the trade will work. If hundreds
of pairs are tested, some will pass by chance. Multiple-testing correction is a
documented next step.

The test is asymmetric in finite samples: testing `Y` on `X` can differ slightly
from testing `X` on `Y`. This project uses the deterministic column order so runs
are reproducible.

### 3.5 Spread mean, standard deviation, and z-score

The training spread supplies two frozen parameters:

```text
mu    = mean(training spread)
sigma = sample standard deviation(training spread)
```

Each later spread observation is standardized:

```text
z_t = (spread_t - mu) / sigma
```

A z-score of `+2` says the current spread is two training standard deviations
above its estimated equilibrium. It does not say there is exactly a 2.3% tail
probability; that familiar normal-distribution interpretation requires stronger
distribution assumptions than the strategy makes.

### 3.6 Mean-reversion half-life

`estimate_half_life` approximates the spread as an Ornstein-Uhlenbeck-style
mean-reverting process. It regresses the spread change on its lagged level:

```text
Delta spread_t = constant + lambda * spread_{t-1} + error_t
```

When `lambda < 0`, deviations tend to shrink. The estimated half-life is:

```text
half_life = -log(2) / lambda
```

It is a descriptive diagnostic, not a holding-period guarantee. A nonnegative
lambda returns infinity because the fitted process does not show mean reversion.

## 4. Signal state machine

`StrategyConfig` holds the rules and validates this ordering:

```text
0 <= exit_z < entry_z < stop_z
```

The default values are:

```text
entry = 2.0
exit  = 0.5
stop  = 4.0
```

Positions use three integer states:

| Position | Meaning | Y leg | X leg when beta is positive |
|---:|---|---|---|
| `+1` | Long spread | Long | Short |
| `0` | Flat | Flat | Flat |
| `-1` | Short spread | Short | Long |

When flat:

- `z <= -entry_z` produces `+1`; the spread is unusually low.
- `z >= +entry_z` produces `-1`; the spread is unusually high.
- `abs(z) >= stop_z` refuses a new position because the deviation may represent
  a broken relationship rather than an opportunity.

When invested, the state is held until either:

- `abs(z) <= exit_z`, meaning the spread returned near equilibrium, or
- `abs(z) >= stop_z`, meaning the divergence became too extreme.

This state machine prevents the strategy from repeatedly entering on every bar
while a position is already open. `generate_target_positions` iterates in time
order and never reads a future z-score.

The final target is forced to zero so the backtest includes an exit cost instead
of silently leaving an open position at the end of the report.

## 5. Information timing and look-ahead prevention

This is the most important section of the backtester.

At the close of day `t`, the program observes `price_t` and calculates `z_t`.
That information can decide the position for the next period. It cannot earn the
return from `t-1` to `t`, because that return has already happened.

The code represents that distinction with two columns:

```python
target = generate_target_positions(zscore, config)
held = target.shift(1).fillna(0)
```

| Time | Newly observed | Decision | Position earning this bar's return |
|---|---|---|---|
| `t-1` close | `z_{t-1}` | `target_{t-1}` | previous position |
| `t` close | `z_t` | `target_t` | `target_{t-1}` |
| `t+1` close | `z_{t+1}` | `target_{t+1}` | `target_t` |

Therefore `held_t = target_{t-1}`. Removing the shift lets the strategy use a
closing price to earn the same price movement that created its signal—a classic
same-bar look-ahead error.

## 6. Position sizing and pair return

The regression beta is converted to gross-normalized leg weights:

```text
gross_scale = 1 + |beta|
w_y = 1 / gross_scale
w_x = |beta| / gross_scale
```

The absolute weights sum to one. If `beta = 1.5`, then:

```text
w_y = 1 / 2.5 = 0.40
w_x = 1.5 / 2.5 = 0.60
```

For positive beta, the unpositioned long-spread return is:

```text
pair_return_t = w_y * return_y_t - w_x * return_x_t
```

The held signal supplies direction:

```text
gross_strategy_return_t = held_t * pair_return_t
```

Thus a long spread earns the pair return, while a short spread earns its negative.
This normalization controls gross exposure; it does not promise exact market,
sector, or factor neutrality.

## 7. Turnover and trading costs

`StrategyConfig.total_cost_rate` converts basis points to a decimal:

```text
(transaction_cost_bps + slippage_bps) / 10,000
```

With the defaults, total modeled cost is `3 / 10,000 = 0.0003`, or 0.03% of
traded gross exposure.

Turnover is the absolute change in the target state:

```python
turnover = target.diff().abs()
```

Examples:

- Flat `0` to long `+1`: turnover `1`
- Long `+1` to flat `0`: turnover `1`
- Long `+1` directly to short `-1`: turnover `2`
- Continue holding `+1`: turnover `0`

Net return is:

```text
net_return_t = gross_return_t - turnover_t * total_cost_rate
```

The cost model is intentionally simple. Real trading costs depend on bid/ask
spreads, liquidity, order size, volatility, market impact, borrow availability,
and the timing difference between the two fills.

## 8. Equity and performance metrics

The equity curve compounds net returns:

```python
equity = (1 + net_return).cumprod()
```

Starting from one dollar, an equity value of `1.12` represents a 12% cumulative
return in the simulation.

Drawdown compares current equity with its historical peak:

```text
drawdown_t = equity_t / running_max_equity_t - 1
```

The metrics are:

- **Total return:** ending equity minus one
- **Annualized return:** compounded return scaled to 252 periods per year
- **Annualized volatility:** daily standard deviation times `sqrt(252)`
- **Sharpe ratio:** mean daily return divided by daily volatility, times
  `sqrt(252)`, with a zero risk-free-rate assumption
- **Maximum drawdown:** worst peak-to-trough decline
- **Trades:** transitions from flat into a nonzero target
- **Exposure fraction:** percentage of bars with an active held position
- **Positive-day fraction:** profitable fraction of nonzero-return days
- **Total cost:** sum of modeled costs as a fraction of capital

These are strategy-level diagnostics. The Sharpe ratio is not corrected for
autocorrelation, non-normal returns, selection bias, or repeated trials.

## 9. Module-by-module code tour

### `data.py`

`validate_prices` enforces the input contract:

1. Input must be a DataFrame with at least two unique columns.
2. The index is converted to UTC timestamps and sorted.
3. Duplicate timestamps keep the last value.
4. Values are converted to numbers.
5. Rows missing either leg are removed jointly.
6. Enough complete rows must remain.
7. Every price must be finite and strictly positive.

Strict positivity matters because `log(0)` and logs of negative values are not
defined for this model.

`simulate_prices` creates a controlled experiment. `BBB` is a log-price random
walk. `AAA` is built from `BBB` with a known beta and a stationary AR(1) residual.
`CCC` and `DDD` are independent random walks. The algorithm should recover
`AAA/BBB`, which tests the research pipeline against known ground truth.

`load_price_csv` and `save_prices` define the wide CSV boundary.

`download_alpaca_daily_bars`:

- Loads `.env` if available.
- Imports Alpaca only inside the function, keeping offline use independent of the
  optional SDK.
- Reads credentials from environment variables.
- Creates a `StockBarsRequest` for adjusted daily bars.
- Uses the IEX or SIP feed.
- Pivots Alpaca's long/multi-index response into timestamp-by-symbol prices.
- Validates the result with the same rules as every other data source.

### `model.py`

`PairModel` is an immutable dataclass containing every fitted parameter. Keeping
the model immutable makes it harder to accidentally recalculate the test-period
mean or standard deviation.

Its `spread` and `zscore` methods apply the frozen equation to any compatible
price frame. `to_dict` makes models easy to serialize into reports.

`estimate_half_life` uses NumPy least squares for the AR-style regression.

`fit_pair`:

1. Selects and validates the two symbols.
2. Calculates log prices.
3. Fits scikit-learn linear regression.
4. Builds the residual spread.
5. Rejects an almost constant spread, which cannot produce meaningful z-scores.
6. Calculates log-return correlation.
7. Runs the statsmodels cointegration test.
8. Returns one frozen `PairModel`.

`select_candidate_pairs` uses `itertools.combinations` to generate each unique
pair once. For `N` symbols there are `N(N-1)/2` pairs, so the correlation screen
can significantly reduce expensive tests in a large universe.

### `strategy.py`

`StrategyConfig` centralizes thresholds, costs, and annualization. Its
`__post_init__` method fails immediately on nonsensical settings.

`signal_for_flat_position` is reusable by both the historical strategy and the
paper-order planner.

`generate_target_positions` is the stateful chronological loop. A vectorized
Boolean expression would be shorter, but a loop makes holding, exit, and stop
behavior explicit and easier to extend with cooldowns.

### `backtest.py`

`BacktestResult` keeps the fitted model, configuration, full daily history, and
summary metrics together.

`run_backtest` is deliberately deterministic:

1. Apply the frozen model to test prices.
2. Generate target positions from z-scores.
3. Shift targets one bar into held positions.
4. Calculate simple asset returns.
5. Convert beta into normalized leg weights.
6. Calculate the pair return and directional gross return.
7. Deduct turnover-based costs.
8. Compound equity and calculate drawdown.
9. Concatenate every intermediate series into `history` for inspection.

The detailed `history` table is important. A backtest that exposes only a Sharpe
ratio is hard to audit. Here you can inspect the exact price, signal, position,
turnover, cost, return, and equity for every timestamp.

`_performance_metrics` consumes that table rather than independently rebuilding
the strategy, so the report and exported history use the same accounting.

### `research.py`

`chronological_split` prohibits random splitting and keeps between 50% and 85% of
observations in training. It also guarantees a minimally useful test window.

`run_research` performs candidate selection only on `training`, truncates to the
configured maximum number of pairs, and sends each frozen model plus only the
`testing` prices to `run_backtest`.

`ResearchResult.summary` combines training diagnostics and test metrics in one
table without losing their labels.

### `reporting.py`

The reporting module writes machine-readable CSVs plus a three-panel diagnostic:

1. Spread z-score and entry thresholds
2. Held strategy position
3. Growth of one dollar

The noninteractive Matplotlib backend allows plots to render in CI or a server
without a display. A writable temporary font cache handles sandboxed systems.

### `execution.py`

`OrderInstruction` is an API-independent order plan. Separating planning from
submission lets tests verify sides, quantities, and gross exposure without any
network call or credentials.

`build_entry_order_plan` converts normalized dollar weights into fractional share
quantities using the latest prices. A `+1` signal buys Y and, for positive beta,
sells X. A `-1` signal reverses both sides.

`submit_paper_order_plan`:

- Requires exactly two instructions and both credentials.
- Hardcodes `TradingClient(..., paper=True)`.
- Creates typed `MarketOrderRequest` objects.
- Adds related client order IDs.
- Submits the two legs sequentially.

Sequential submission creates leg risk. If the first order fills and the second
fails, the account has an unintended directional exposure. The code surfaces this
limitation instead of pretending the pair is atomic.

### `cli.py`

`argparse` creates four commands:

- `demo`: deterministic offline end-to-end experiment
- `research`: run the same process on a wide CSV
- `download`: retrieve Alpaca adjusted daily closes
- `paper-plan`: estimate a recent pair, create a plan, and optionally submit it
  to the paper account

Research and reporting imports are delayed where useful so a simple order plan
does not pay Matplotlib's startup cost.

The `--signal long` and `--signal short` overrides exist for paper-execution
experiments. They should not be interpreted as research signals.

## 10. What each test proves

### `test_data.py`

- Synthetic generation is deterministic for a fixed seed.
- Every generated price is finite and positive.
- Validation rejects zero prices before logarithms are calculated.

### `test_model.py`

- Regression approximately recovers the known synthetic beta.
- The known spread passes the cointegration filter.
- Candidate selection finds the intended `AAA/BBB` pair.

### `test_strategy.py`

- A hand-written z-score sequence enters, holds, exits, and stops as expected.
- The final target is closed for backtest accounting.

### `test_backtest.py`

- The first target signal does not become a held position until the next row.
- Costs equal turnover times the configured basis-point rate.
- Every reported metric is finite in the controlled experiment.

### `test_execution.py`

- A long-spread plan creates opposing order sides for a positive-beta pair.
- Leg notionals sum to the requested gross notional.
- Quantities are positive.

Tests establish software behavior, not market profitability.

## 11. Reading the demo result correctly

The synthetic demo normally selects `AAA/BBB` because that relationship was
deliberately planted in the data. Its positive test result demonstrates that the
pipeline can detect and trade the simulated mean reversion.

It does **not** mean:

- Real stocks will follow the same process.
- The displayed Sharpe ratio is a live expectation.
- A 5% cointegration p-value ensures future stationarity.
- Modeled costs reproduce real fills.
- Six trades are enough to estimate a stable distribution.

The synthetic result is a unit/integration test with economically interpretable
output—not investment evidence.

## 12. Important limitations and their code-level remedies

### One fixed split

Regimes change, and one test window can be lucky. Implement expanding or rolling
walk-forward evaluation where each block is predicted using only earlier data.

### Multiple testing

Testing many pairs increases false discoveries. Retain all training p-values and
apply Benjamini-Hochberg false-discovery control before selection.

### Fixed hedge ratio

Beta can drift. Add rolling stability diagnostics or controlled re-estimation at
scheduled boundaries. Do not update beta on every test observation and call the
result fixed out of sample.

### Close-to-close execution

Daily closes do not show the bid/ask spread, intraday path, or whether both legs
could fill. Download quotes, define an executable timestamp, and simulate each
leg at its bid or ask plus market impact.

### Shorting assumptions

The backtest assumes short exposure is available. Real execution needs asset
shortability, borrow fees, locate constraints, and forced-buy-in handling.

### Portfolio overlap

Several selected pairs may share symbols and silently concentrate risk. Add an
allocator that constrains symbol, sector, factor, gross, and net exposure across
the entire portfolio.

### Order lifecycle

Paper submission currently sends two orders and returns their initial responses.
Production-quality paper infrastructure should monitor fills, reconcile intended
and actual positions, prevent duplicate submission, cancel stale orders, and
flatten imbalances.

## 13. Recommended way to study the code

Use this sequence:

1. Run `python examples/walkthrough.py` and inspect its output.
2. Read `data.py`, then change one synthetic parameter and rerun.
3. Read `model.py`; manually reproduce one spread in a Python shell.
4. Read `strategy.py`; draw the state transitions for `test_strategy.py`.
5. Read `backtest.py`; inspect the first 20 rows containing a trade.
6. Temporarily remove the one-bar shift, observe the difference, and restore it.
7. Increase costs and find the synthetic strategy's break-even cost.
8. Read `execution.py`; calculate the two notionals by hand for one plan.
9. Implement the walk-forward exercise in `LEARNING_GUIDE.md` with new tests.

At each stage, predict the output before running the code. That habit turns the
project from something you can execute into something you genuinely understand.
