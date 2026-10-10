# Can a language model read earnings reports well enough to predict stock returns?

Every three months, each public company publishes an earnings report: a press
release with its results and what management expects next. This project asks a
language model (Claude, the same kind of artificial intelligence as ChatGPT) to
read 676 of these reports and rate how upbeat or gloomy management sounds. It
then checks whether those ratings predict how the stock does over the following
month, and whether the model does better than a classic method that simply
counts positive and negative words.

---

## In short

**The main test, chosen before any data were analysed:** does the model's tone
score rank stocks by how well they do over the next 20 trading days (about one
month)? It did, a little. Across 676 reports from July 2023 to August 2026, the
rank correlation was **+0.06** (0 means no link), and it was positive in 9 of
13 earnings seasons. The classic word-counting method scored **−0.05**, no
better than chance. But the result falls short of the usual bar for
statistical significance (p = 0.09), and the pre-chosen trading strategy,
rebalanced after each earnings season, lost a little after costs.

**How the sample grew.** The first run used 525 reports through December 2025
(information coefficient +0.082, p = 0.08). The study was then extended to
August 2026 as new reports came out (+0.064, p = 0.09). The test, the code and
the settings stayed the same; only the end date moved, and the first 10 seasons
give exactly the same numbers in both runs. The 151 added reports therefore
work as a fresh out-of-sample check, and on their own they showed almost no
link (+0.02, p = 0.83). That is why the headline number went down.

**The most interesting pattern (exploratory):** the effect appears in the first
few days after a report and then fades. Trades placed right after each report
did better than the seasonal strategy, which trades about five weeks later.
That version holds only a few stocks at a time, so it is fragile.

**The clean test does not confirm a lasting effect.** A model that learned from
internet text written after a report may have read how the stock moved. Claude
learned from text up to June 2026, so only the 48 reports from July and August
2026 are truly new to it. On those 48, Claude's correlation was +0.03
(p = 0.83): no sign of skill, but far too few reports to rule it out. (On the
same 48, Llama scored +0.29 with p = 0.05. That is most likely noise: one
result among many, on a tiny sample, from a model that agrees only loosely with
Claude and found nothing on the 520 reports before them.) As a
second check, the same reports were scored by Llama 3.2, a smaller model whose
training data ends in December 2023. On the 520 reports from January 2024 to
June 2026, which Llama cannot have read about, no score predicted returns
(Claude +0.05, Llama +0.01, the word list −0.08), none of them statistically
significant. Most of Claude's overall result comes from the 108 reports of late
2023 (+0.23). Memory, a period when tone mattered more, or chance could each
explain that; this sample cannot tell them apart. The two models also read the
reports quite differently (their scores correlate only 0.39), so Llama's weak
result may partly reflect weaker reading rather than missing memory. See the
[clean test](#clean-test-an-older-model-that-cannot-have-seen-what-happened)
and the [next steps](#next-steps).

![Growth of $1 in the long/short portfolios](results/figures/cumulative_long_short.png)

## Key results in plain words

| Question | Answer |
|---|---|
| Does the model's tone score line up with the next month's returns? | Yes, a little: a rank correlation of +0.06, positive in 9 of 13 earnings seasons. The finance word list scored −0.05. |
| Is that just luck? | It might be. The t-statistic was 1.83; with only 13 seasons, the usual 5% significance bar is about 2.18. The p-value was 0.09, so there is roughly a 9% chance of a result this strong with no real link. |
| How big was the difference in returns? | Small. Over 20 days the most upbeat fifth of reports roughly matched the market (−0.02%), while the most gloomy fifth trailed it by 0.41%. |
| Could you have made money trading on it? | Not with the pre-chosen strategy: rebalancing after each earnings season gave a Sharpe ratio of −0.14 after costs. Trading right after each report looked better (Sharpe ratio 0.73), but that is exploratory and rests on just 2 stocks on each side on a typical day, so its margin of error is about ±0.64. |
| How long does the effect last? | A few days. Starting the same trades 5 trading days later cut the Sharpe ratio from 0.73 to 0.21, and 10 days later it turned negative. |
| Which part of the report mattered most? | What management said about **future guidance**, and how that changed since the previous quarter, had the strongest results. Comments on **customer demand** showed nothing. None of the topics is significant on its own. |
| Did it hold up on reports a model could not have read about? | No. On the 48 reports after Claude's training cutoff (July and August 2026), Claude's correlation was +0.03 (p = 0.83). On the 520 reports from January 2024 to June 2026, new to Llama, Claude scored +0.05 (p = 0.31) and Llama +0.01 (p = 0.89). The overall result rests mostly on late 2023. |
| Does the model give the same answer twice? | Yes. Scoring the same reports a second time gave the identical guidance rating 80% of the time, and it was never more than one step apart. |

## What the numbers mean

| Term | Plain meaning |
|---|---|
| Earnings report | A company's press release with its quarterly results, filed with the U.S. Securities and Exchange Commission as a "Form 8-K". |
| Tone score | The model's rating from −2 (clearly negative, for example guidance cut) to +2 (clearly positive, for example guidance raised). |
| Return above the market | The stock's return minus the return of an S&P 500 index fund over the same days. It removes the effect of the whole market going up or down. |
| Information coefficient | How well a score *ranks* stocks by their later returns. A rank correlation from −1 to +1; 0 means no link. In investing, +0.05 to +0.10 is considered useful. |
| t-statistic | How far a result is from zero, measured in units of its own uncertainty. The bar for "unlikely to be chance" depends on the amount of data: with 13 earnings seasons it is about 2.18. |
| p-value | The chance of seeing a result at least this strong if there were really no link. Below 0.05 is the usual bar for "statistically significant". |
| Standard error | The typical size of the error in an estimate. A Sharpe ratio of 0.7 with a standard error of 0.6 could easily be anywhere from about 0.1 to 1.3. |
| Hit rate | How often the most upbeat and most gloomy fifths moved the way their score predicted. 50% is a coin flip. |
| Long/short portfolio | Buy the stocks with the most upbeat reports and sell short (bet against) those with the most gloomy reports. It makes money if the first group beats the second, whatever the overall market does. |
| Sharpe ratio | Return earned per unit of risk taken. Above 1 is good; the stock market as a whole is usually around 0.4 to 0.6. |
| Maximum drawdown | The worst fall from a high point to a later low point. |
| Fama-French factors | Well-known drivers of stock returns (the overall market, company size, cheap versus expensive stocks, profitability, investment, and recent winners or "momentum"). Checking against them shows whether a strategy is just repeating a known pattern. |
| Alpha | The part of a strategy's return that those known factors cannot explain. |
| Training cutoff | The date up to which the language model learned from text. Anything before it, the model may already have read about. |

---

## Detailed results

The section below is written automatically by `earnsig report` from the files in
[`results/`](results/).

<!-- RESULTS:START -->

> [!NOTE]
> 628 of 676 reports are older than the language model's training cutoff (2026-06-30); the last row of the table uses only the 48 after it.

_Sample: 676 earnings reports from 50 companies, 2023-07-07 to 2026-08-27. Scored by: Claude (claude-haiku-5-5). Returns are measured above the S&P 500 index fund (SPY). Trading costs: 0.10% per trade plus 0.50% a year to borrow shares for selling short._

### Main test (chosen before the data were analysed)

The `llm` score (future-guidance tone, with the other topics as a tie-breaker), its information coefficient against the 20-day return above the market, and a long/short portfolio rebalanced after each earnings season. These choices were fixed in the settings before any real report was scored.

| Main test | Language model tone | Finance word list |
|---|---:|---:|
| Information coefficient: rank correlation of the score with the 20-day return above the market | +0.064 | -0.054 |
| t-statistic across 13 seasons (5% significance needs about 2.18 with this few seasons) | 1.83 | -1.33 |
| p-value (chance of a result this strong if there were no real link; below 0.05 is the usual bar) | 0.09 | 0.21 |
| Earnings seasons where the correlation was positive | 69% of 13 | 38% of 13 |
| Portfolio rebalanced each season, after costs: Sharpe ratio (± one standard error) | -0.14 (± 0.57) | -0.80 (± 0.65) |

**Verdict on the main test:** the language model's score was positively linked to later returns (information coefficient +0.064), but the link is **not** statistically significant at the usual 5% level (p = 0.09), and the pre-chosen seasonal portfolio made little after costs. Treat the result as suggestive, not proven.

### Exploratory results

Everything below was examined after seeing the data. With this many variations, some will look good by luck, so none of it should be read as a finding on its own.

**The 20-day portfolio is very thin.** Holding each report for 20 trading days looks much better than the main portfolio, but on a typical day it holds only 2 stocks long and 2 short, and on 25% of days only one side (the other side is then the market fund). A single stock can swing the result, which is why the delay table below jumps around and why its Sharpe ratio of 0.73 has a margin of error of about ±0.64.

**Why the two portfolios differ.** The link between tone and returns shows up in the first few days after a report. Starting the same trades later shows how fast it fades; the seasonal portfolio typically trades 35 calendar days after a report, after the effect has mostly gone. These five numbers are themselves noisy, so read the pattern, not each value.

| Trades start this many trading days later | 0 | 5 | 10 | 20 | 40 |
|---|---:|---:|---:|---:|---:|
| Sharpe ratio after costs | 0.73 | 0.21 | -0.40 | 0.35 | 0.25 |

#### All measures

| Measure | Language model tone | Finance word list |
|---|---:|---:|
| Reports with a score | 676 | 676 |
| Information coefficient: rank correlation of score with the 20-day return above the market | +0.064 | -0.054 |
| t-statistic of the information coefficient across seasons | 1.83 | -1.33 |
| p-value | 0.09 | 0.21 |
| Earnings seasons where the correlation was positive | 69% of 13 | 38% of 13 |
| Hit rate: top and bottom fifth that moved the predicted way | 55.6% | 48.9% |
| Average 20-day return above the market, most upbeat fifth / most gloomy fifth | -0.02% / -0.41% | -1.01% / 0.13% |
| **Long/short portfolio, rebalanced each season, after costs:** yearly return | -1.9% | -10.0% |
| Yearly volatility (typical size of ups and downs) | 13.6% | 12.5% |
| Sharpe ratio (return per unit of risk), after costs (before costs) | -0.14 (-0.03) | -0.80 (-0.72) |
| Standard error of that Sharpe ratio | 0.57 | 0.65 |
| Maximum drawdown (worst fall from a peak) | -26.8% | -32.3% |
| Turnover per rebalance (replacing every holding = 400%) | 215% | 123% |
| Holding periods that made money | 46% | 46% |
| **Long/short portfolio, each report held 20 trading days, after costs:** Sharpe ratio (± one standard error) | 0.73 (± 0.64) | -1.39 (± 0.80) |
| Maximum drawdown | -27.4% | -68.6% |
| Stocks held on a typical day, long / short | 2 / 2 | 3 / 2 |
| Days with only one side held (other side hedged with the market fund) | 25% | 21% |
| 20-day portfolio: alpha against the Fama-French five factors plus momentum, yearly (t-statistic) | 13.2% (1.13) | -37.8% (-2.96) |
| **Fama-French five factors plus momentum, seasonal portfolio:** alpha, yearly return not explained by the factors (t-statistic) | 2.8% (0.40) | -9.6% (-1.53) |
| Market beta (sensitivity to the overall stock market) | -0.14 | +0.22 |
| Size / value / profitability / investment betas | -0.12 / -0.36 / +0.02 / +0.01 | +0.13 / -0.35 / +0.17 / -0.02 |
| Momentum beta (tendency to hold recent winners) | +0.05 | +0.05 |
| R-squared (share of the ups and downs explained by the factors) | 0.14 | 0.20 |
| Information coefficient using only reports after the model's training cutoff (number of reports) | +0.032 (48) | -0.013 (48) |

_A note on noise: this page reports dozens of statistics, so one or two with a t-statistic above 2 are expected by chance alone. Treat the finance word list 20-day portfolio alpha (t = -2.96) as likely noise, not a finding; it was not part of the main test._

![Return after the report, by tone group](results/figures/car_by_quintile.png)

#### Which topic matters most?

Each report also got separate scores for what management said about future guidance, profit margins and customer demand.

| Score | Information coefficient | t-statistic | Hit rate | Sharpe ratio, seasonal portfolio | Sharpe ratio, 20-day portfolio |
|---|---:|---:|---:|---:|---:|
| Language model tone | +0.064 | 1.83 | 55.6% | -0.14 | 0.73 |
| Language model: guidance | +0.060 | 2.15 | 55.9% | -0.49 | 0.90 |
| Language model: margins | +0.035 | 0.66 | 58.9% | 0.21 | 0.44 |
| Language model: demand | -0.034 | -0.89 | 49.3% | 0.91 | -0.40 |
| Language model: overall tone | +0.040 | 0.71 | 53.7% | 0.57 | 0.38 |
| Language model: change since last quarter | +0.094 | 2.84 | 56.0% | 0.24 | 0.05 |
| Finance word list | -0.054 | -1.33 | 48.9% | -0.80 | -1.39 |

![Information coefficient by score](results/figures/ic_by_signal.png)

#### Does the model give the same answer twice?

The same 40 reports were scored twice with identical inputs.

| Score | Same score both times | Within one step | Rank correlation | Agreement beyond chance (weighted kappa, 1 = perfect) |
|---|---:|---:|---:|---:|
| Guidance tone | 80% | 100% | 0.84 | 0.86 |
| Overall tone | 78% | 100% | 0.86 | 0.83 |
| Margins tone | 90% | 100% | 0.95 | 0.95 |
| Demand tone | 80% | 100% | 0.91 | 0.92 |

<!-- RESULTS:END -->

---

<!-- CLEAN_TEST:START -->

## Clean test: an older model that cannot have seen what happened

The main results use Claude (claude-haiku-5-5), which learned from text written after every report in the sample. Here the same reports are scored by Llama (llama3.2:3b, run with Ollama), whose training data ends on 2023-12-31, so for reports published after that date it cannot have read how the stock moved. If its score still predicts returns on those reports, the effect is more likely to be real reading skill; if it does not, memory is the likelier explanation for the main result. One caution: the second model is much smaller, so a weaker result can also mean it simply reads less well.

| Measure | Claude (claude-haiku-5-5) | Llama (llama3.2:3b, run with Ollama) | Finance word list |
|---|---:|---:|---:|
| Training data ends | 2026-06-30 | 2023-12-31 | not applicable |
| Reports scored | 676 | 676 | 676 |
| Reports published after the model's training data ends | 48 | 568 | not applicable |
| Information coefficient, all reports (rank correlation with the 20-day return above the market) | +0.064 | +0.029 | -0.054 |
| t-statistic across seasons (5% significance needs about 2.18) | 1.83 | 0.85 | -1.33 |
| p-value | 0.09 | 0.41 | 0.21 |
| Hit rate (top and bottom fifth that moved the predicted way) | 55.6% | 50.7% | 48.9% |
| Sharpe ratio, each report held 20 trading days, after costs | 0.73 | -0.08 | -1.39 |
| Sharpe ratio, rebalanced each season, after costs | -0.14 | 0.07 | -0.80 |
| **Information coefficient using only reports the model could not have read about** | **+0.032 (48 reports)** | **+0.032 (568 reports)** | not applicable |

**The same comparison split at each model's training cutoff** (rank correlation of each score with the 20-day return above the market, pooled across reports, with its p-value):

| Period | Reports | Claude (claude-haiku-5-5) | Llama (llama3.2:3b, run with Ollama) | Finance word list |
|---|---:|---:|---:|---:|
| Up to the end of December 2023 (both models may have read about these) | 108 | +0.229 (p 0.02) | +0.085 (p 0.38) | +0.036 (p 0.71) |
| January 2024 to June 2026 (new to Llama only) | 520 | +0.045 (p 0.31) | +0.006 (p 0.89) | -0.078 (p 0.08) |
| After June 2026 (new to both models) | 48 | +0.032 (p 0.83) | +0.290 (p 0.05) | -0.013 (p 0.93) |

**Small periods are noisy.** A period with fewer than 100 reports is too small to judge on its own: with nine numbers in this table, one of them can reach p = 0.05 by chance alone. That is the most likely reading of Llama's +0.290 on 48 reports: it is a single result on a tiny sample, it is not backed by the larger periods, and the two models agree only loosely, so it should not be read as that score having found something.

**Do the two models read the reports alike?** On the same 676 reports they gave the identical guidance score 26% of the time and were within one step 72% of the time; the rank correlation of their scores is 0.39. That is well below the 0.7 or so that would show the two models read the reports alike, so the second model may simply be reading worse, and this test **cannot tell memory apart from weaker reading**. A larger model with an equally early cutoff would settle it.

Full tables and charts for the second model: [`results/llama/`](results/llama/).

<!-- CLEAN_TEST:END -->

### Reading the clean test

* **Most of Claude's edge comes from a small early sample.** Claude's rank
  correlation was +0.23 on the 108 reports up to December 2023 but only +0.05 on
  the 520 from January 2024 to June 2026, even though Claude's training data
  covers both periods. Three
  explanations fit: **noise** (108 reports over about two earnings seasons is a
  small sample), **a change in market conditions** (tone may have mattered more
  in late 2023), or **better memory of older events** (more has been written
  about them, so a model may recall them more easily). This study cannot tell
  these apart.
* **The test cannot separate memory from reading quality.** Llama 3.2 3B is a
  small model, and its scores agree only loosely with Claude's (rank correlation
  0.39). If the two agreed closely and Llama still found nothing, memory would
  be the likelier explanation for Claude's result. As it is, Llama may simply be
  reading worse. Running the same test with the larger Llama 3.1 8B (same
  December 2023 cutoff) would weaken that objection.
* **Claude's own clean test is too small to settle anything.** Only 48 reports
  (July and August 2026) are newer than Claude's training data. Claude scored
  +0.03 on them (p = 0.83). Llama scored +0.29 (p = 0.05) on the same 48, but
  with so few reports, and with Llama finding nothing on the 520 before them,
  that is most likely chance.
* **What can be said with confidence:** on the reports published after December
  2023, none of the three scores reliably predicted the following month's
  returns.

---

## How the study works

```mermaid
flowchart LR
  A["Earnings reports<br/>from the U.S. Securities and<br/>Exchange Commission"] --> C["Language model<br/>rates the tone<br/>from -2 to +2"]
  A --> D["Finance word list<br/>counts positive and<br/>negative words"]
  B["Daily stock prices<br/>from Yahoo Finance"] --> E["Return above the market<br/>over the next 20 trading days"]
  C --> F[Scores]
  D --> F
  E --> G["Does the score rank<br/>later returns?"]
  F --> G
  F --> H["Long/short portfolios"]
  B --> H
  H --> I["Return, risk, Sharpe ratio,<br/>check against known factors"]
```

### 1. Collecting the reports
`earnsig collect` downloads every earnings report for each company in
[`config/universe.csv`](config/universe.csv) (50 large U.S. companies across all
11 sectors) from EDGAR, the Securities and Exchange Commission's public filing
database. It looks for Form 8-K filings marked "Item 2.02, Results of Operations
and Financial Condition", takes the attached press release (exhibit 99.1),
removes the web formatting and records the exact time the filing went public.
Earnings-call transcripts can be used instead with
`earnsig collect --source local --manifest transcripts.csv` (columns
`ticker, published_at, path`), if you have access to them.

### 2. Scoring the tone with a language model
Each report is read once by the model using a fixed set of instructions
([`llm_score.py`](src/earnsig/llm_score.py)). The model must answer in a fixed
structured format (JSON, a standard text format for data) with whole-number
scores from −2 to +2 for **future guidance**, **profit margins**, **customer
demand** and **overall tone**, a yes/no flag for whether each topic is
mentioned, and a one-line reason. Before scoring, the company name, stock
ticker and all dates are hidden, to make it harder for the model to recognise
the company and recall what happened next. Every answer is saved in
`data/llm_cache.jsonl`, so re-running costs nothing and every score can be
checked. `earnsig consistency` scores a random sample a second time and
measures how often the answers match.

**No paid account is needed.** Choose how the reports are scored with
`llm.provider` in [`config/config.yaml`](config/config.yaml):

| Option | What you need | Cost |
|---|---|---|
| `claude_code` (default) | [Claude Code](https://code.claude.com), signed in with a Claude Pro or Max subscription | Included in the subscription; uses its normal usage limits |
| `ollama` | [Ollama](https://ollama.com) and a downloaded model, for example `ollama pull llama3.2:3b` (about 2 gigabytes; runs on a Mac with 8 gigabytes of memory) | Free; runs on your own computer |
| `anthropic` | A key for Anthropic's paid programming interface | Pay per use (optional, not needed) |

With `claude_code`, a large batch of reports may not fit within one usage
period. When the limit is reached, scoring stops cleanly, and running
`earnsig score` again after the limit resets carries on where it stopped. The
Haiku model uses the least of the allowance.

With `ollama`, an older model is an advantage: Llama 3.2 learned from text up to
December 2023, so it cannot have read about 2024 and 2025 stock moves. The
ready-made settings in [`config/llama.yaml`](config/llama.yaml) score the same
reports with it and keep its results separate in `results/llama/`. Small local
models read less accurately than Claude, so comparing the two is a useful
experiment in itself.

The score used for trading (`llm`) is the guidance score, with the average of
the other three scores added at one tenth of the weight to break ties, because
five possible values produce many ties. `llm_delta` is the change in guidance
score since the same company's previous report.

### 3. The comparison method: a finance word list
`lm` counts words from the Loughran-McDonald word lists, a standard set of
positive and negative words built specifically for financial documents. The
score is (positive − negative) ÷ (positive + negative), and a positive word
preceded by "not", "no" or "never" counts as negative. To use the official list,
put its file at `data/external/lm_master_dictionary.csv`; otherwise the copy
included with the `pysentiment2` Python package is used.

### 4. Measuring returns without peeking ahead
The first day a trade is possible ("day 0") is **the first trading day whose
4 p.m. New York closing price comes after the report was published**: a report
released at 7 a.m. can be traded at that day's close, one released at 4:05 p.m.
at the next day's close. Positions start at the day-0 close, so the study
**never counts the immediate price jump on the day of the report**; it only
counts what happens over the next 20 trading days. The return above the market
is the stock's return minus that of the S&P 500 index fund (SPY), or minus the
stock's sector fund with `event.benchmark: sector`. These rules are checked by
automated tests in [`tests/test_events.py`](tests/test_events.py).

### 5. The two portfolios
* **Rebalanced after each earnings season (main test).** On the first trading
  day of March, June, September and December, rank the stocks by the score of
  their most recent report published *before* that day. Buy the top fifth, sell
  short the bottom fifth, in equal amounts, and hold until the next rebalance.
* **Each report held for 20 trading days (exploratory).** Every report opens a
  position for the 20 trading days after it. Whether it counts as "upbeat" or
  "gloomy" is decided by comparing it only with reports published *earlier*,
  because later reports in the same season are not yet known at that point.

The two answer different questions. The seasonal portfolio often trades weeks
after a report (a report from late January is first traded on March 1), so it
only works if tone predicts returns for several months. The 20-day portfolio
tests the month right after the report directly. In this study the effect
showed up in the 20-day portfolio and mostly faded in the seasonal one, which
trades a median of 35 days after each report. Delaying the 20-day trades by 5
to 10 trading days removes most of the effect, so the market seems to catch up
within about a week.

The 20-day portfolio is also thin: with 50 companies reporting over a few weeks,
only about a fifth of the reports open at any time fall in each extreme group,
so on a typical day it holds 2 stocks on each side, and on about a quarter
of days only one side (the other side is then hedged with the S&P 500 fund).
Each side is equally weighted across the stocks it holds, and days with nothing
held count as zero return.

Trading costs: 0.10% of the amount traded each time a position changes, plus
0.50% a year to borrow the shares sold short. Both can be changed in the
settings.

### 6. How the results are measured
| Measure | How it is calculated |
|---|---|
| Information coefficient | Rank correlation (Spearman) between the score and the 20-day return above the market, calculated within each earnings season and then averaged; the t-statistic uses the spread across seasons |
| Hit rate | Share of top-fifth reports followed by a return above the market, plus bottom-fifth reports followed by a return below it |
| Sharpe ratio | Yearly average return divided by yearly volatility of the daily long/short returns (no risk-free rate is subtracted, because buying and selling short in equal amounts needs no net money) |
| Maximum drawdown | Largest fall from a high point to a later low point of the long/short portfolio's value |
| Factor check | Ordinary least squares regression of the daily long/short returns on the Fama-French five factors plus momentum, with standard errors adjusted for patterns across neighbouring days (Newey-West, 5 days) |
| After-cutoff information coefficient | The same correlation, using only reports published after the model's training cutoff |

---

## Run it yourself

```bash
git clone https://github.com/pooja003-cloud/llm-earnings-signal && cd llm-earnings-signal
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                      # 64 automated tests, about 3 seconds

earnsig demo --readme          # practice run on made-up data, no internet needed, about 20 seconds
```

Full run (free; no paid account):

```bash
cp .env.example .env           # set SEC_USER_AGENT="Your Name you@email.com" (the SEC asks every downloader to identify itself)
claude                         # once: sign in with your Claude Pro account, then type /exit
earnsig prices                 # daily stock prices from Yahoo Finance
earnsig factors                # Fama-French factor data from Kenneth French's website
earnsig collect                # earnings reports from the SEC (about 15 to 20 minutes)
earnsig score --dry-run        # shows how many reports will be scored
earnsig score --limit 20       # try a few first; answers are saved in data/llm_cache.jsonl
earnsig score                  # run again after each usage reset until everything is scored
earnsig consistency            # score 40 reports a second time
earnsig baseline               # finance word list scores
earnsig backtest               # all calculations and charts
earnsig report                 # writes the "Detailed results" section of this page
```

Clean test with an older model, free on your own computer (needs
[Ollama](https://ollama.com); about 4 to 5 hours for 676 reports on a Mac with
8 gigabytes of memory; the Colab notebook in `notebooks/` does it free on a
cloud graphics card instead):

```bash
brew install ollama && brew services start ollama
ollama pull llama3.2:3b
earnsig --config config/llama.yaml score        # resumes if stopped
earnsig --config config/llama.yaml consistency
earnsig --config config/llama.yaml backtest     # results go to results/llama/
earnsig --config config/llama.yaml compare      # adds the "Clean test" section to this page
```

No suitable computer? The notebook
[`notebooks/llama_clean_test_colab.ipynb`](notebooks/llama_clean_test_colab.ipynb)
does the scoring on a free Google Colab machine with a graphics card in about
half an hour ([open it in Colab](https://colab.research.google.com/github/pooja003-cloud/llm-earnings-signal/blob/main/notebooks/llama_clean_test_colab.ipynb));
then run the last two commands above on your own computer.

Or run everything with `make all`. All settings are in
[`config/config.yaml`](config/config.yaml): the scoring model, the period
studied (`analysis_start`, `end_date`), the comparison fund, the holding period,
trading costs, rebalance months and the model's **training cutoff**, which you
should set to the published date for whichever model you use.

## What is in this repository

```
config/            list of companies (universe.csv) and all settings (config.yaml)
src/earnsig/
  collect.py       downloads earnings reports from the SEC; loads your own transcripts
  market_data.py   stock prices from Yahoo Finance; Fama-French factor data
  llm_score.py     scoring instructions, answer format, hiding names and dates, saved answers, consistency check
  lm_baseline.py   finance word list scores (Loughran-McDonald)
  events.py        finds the first day a trade is possible and measures returns above the market
  portfolio.py     the two long/short portfolios
  metrics.py       information coefficient, hit rate, Sharpe ratio, drawdown, factor check
  analysis.py      runs every calculation and writes results/
  figures.py       the charts on this page
  report.py        writes the "Detailed results" section of this page
  synthetic.py     made-up data for the practice run
tests/             automated checks, including the no-peeking-ahead rules
results/           all result tables, daily returns, holdings, scored reports and charts
```

---

## Next steps

* **A larger clean-test model (the most useful remaining step).** The two
  models agree only loosely (rank correlation 0.39), so the clean test cannot
  tell "Claude remembered what happened" apart from "Llama reads worse". Score
  the same reports with Llama 3.1 8B, which has the same December 2023 training
  cutoff but reads better. If its scores agree closely with Claude's and still
  predict nothing after 2023, memory becomes the likelier explanation. The settings are
  ready in [`config/llama8b.yaml`](config/llama8b.yaml); in the Colab notebook,
  set `CONFIG = "config/llama8b.yaml"` and run it (about one to two hours on the free
  graphics card), then run `earnsig --config config/llama8b.yaml backtest` and
  `earnsig --config config/llama8b.yaml compare`, which adds its own section.
* **All 1,192 reports since 2021.** They are already downloaded. Scoring the 516
  reports from before July 2023 with Claude (set `analysis_start: "2021-01-01"`
  in `config/config.yaml`) would roughly double the number of earnings seasons
  in the main test. It adds nothing to the clean test, because all of those
  reports are older than every model's training cutoff.
* **More reports after June 2026.** Claude's own clean test has only 48
  reports. Each new earnings season adds about 50; rerunning `earnsig collect`,
  `earnsig score` and the later steps with a later `end_date` grows it.
* **Historical index membership** instead of today's large companies, to remove
  survivorship bias.

## Biases and limitations

Read this before trusting any number above.

**The language model may already know what happened.** This is the biggest
problem. A model that learned from text written after a report may have read
news about how the stock reacted, and can "rate" the report with that hindsight.
Hiding the company name and dates helps a little, but distinctive products,
figures and wording can still give a company away. The only clean test uses
reports published after the model's training cutoff, shown in the last row of
the main results table.

For Claude that test is possible only on a few reports: Claude Haiku 5.5
learned from text up to June 2026, so only the 48 reports from July and August
2026 are new to it, and on them its score showed no clear link to returns
(+0.03, p = 0.83). So the same reports were also scored by an older model,
Llama 3.2 (3 billion parameters, training data up to December 2023), run free
through Ollama; 568 of the 676 reports are new to it. The
[clean test](#clean-test-an-older-model-that-cannot-have-seen-what-happened)
shows that on the 520 reports from January 2024 to June 2026 no score predicts
returns, and that Claude's own result on them is also weak (+0.05). Llama did
better on the reports it could have read about (+0.09) than on those 520
(+0.01), which fits the memory explanation, but with only 108 earlier reports
that difference could also be chance. Llama is also a much smaller model, so
part of its weaker result may be weaker reading.

**A small sample.** 50 companies over about 13 quarters is 676 reports and only
13 earnings seasons, so every number has a wide margin of error. With 13
seasons, a t-statistic needs to reach about 2.18 (not the familiar 2) for 5%
significance. A Sharpe ratio measured over about three years has a standard
error of about 0.6, as the results tables show.
All 1,192 reports since 2021 have been downloaded; set `analysis_start` equal
to `start_date` in the settings to use them all.

**Only today's survivors.** The list is today's large companies. Companies that
failed, were bought or became smaller since 2021 are missing, which can make
results look better or worse than they really were. The effect on this study
is probably modest, because both sides of every trade come from the same list
of survivors, and all 50 were already large, established companies in 2021.
But it cannot be measured without the historical list. A stricter study would
use the S&P 500 membership as it stood at each date.

**Only the weeks after the report are tested.** Skipping the price jump on the
report day is deliberate, because nobody can trade before the report is public.
It does mean that a weak result here would not show that the model reads tone
badly, only that the market reacts quickly.

**Timing of filings.** The time a filing appears on the SEC's website can be
minutes to hours later than the press release on news services. That makes the
study's "first chance to trade" a little late, if anything, which is the safe
direction to be wrong in.

**Simplified trading costs.** A flat 0.10% per trade and 0.50% a year for
borrowing are reasonable for large, heavily traded companies, but ignore the
extra cost of trading large amounts or on busy earnings days. Results are shown
both before and after costs.

**Many variations.** Six language model scores, one word-list score, two
portfolios and several settings add up to many possible versions, and some
will look good by luck. The main test was fixed in the settings before any real
report was scored: the `llm` score (guidance tone, with the other topics as a
tie-breaker), its information coefficient against the 20-day return above the
market, and the portfolio rebalanced after each earnings season. Everything else
is labelled exploratory. The sample was later extended from December 2025 to
August 2026 with the same test and code (see "How the sample grew" above); the
headline number fell from +0.082 to +0.064 as a result, and both are reported.

**Sensitivity to the instructions and the model.** Scores depend on how the
instructions are worded and which model version reads them. The instruction
version (`prompt_version`) is stored with every saved answer; change it whenever
the instructions change. The main results used `claude-haiku-5-5`, the model the
`haiku` setting pointed to on 9 October 2026; new runs record the exact model
identifier with every saved answer, so a later change of model is visible.

**Data terms.** Filings from the Securities and Exchange Commission are public.
Yahoo Finance data is for personal use, so the downloaded prices and reports
(the `data/` folder) are not uploaded to this repository. The Loughran-McDonald
word lists have their own terms of use (free for academic use).
