# Causality-Aware Query Answering on Incomplete Databases
A framework for answering SQL queries over incomplete databases with **Missing Not At Random (MNAR)** data. Rather than discarding incomplete tuples or imputing values, the system rewrites queries into probabilistic forms that estimate tuple membership in the query result with quantified uncertainty.
## Full paper: (https://research.engr.oregonstate.edu/idea/sites/research.engr.oregonstate.edu.idea/files/causal_aware.pdf)
## Overview

Given a relational table with missing values and a SQL query, the framework:

1. **Decomposes** query predicates using ordered separating sets into conditionally independent factors.
2. **Estimates** each factor P(φ(A) | X\_A = x) from the observed data via a single-pass aggregation.
3. **Scores** each tuple with a probability of belonging to the query result.
4. **Returns** a set (certain + possible tuples) along with confidence intervals.

The system compiles all estimation logic into standard SQL (CTEs, conditional aggregation, joins) and delegates execution to PostgreSQL.

## Approaches

| Approach | Description |
|----------|-------------|
| **No-optimization(mGraph-QE)** (`SetQueryRewriterExecuter.py`) | Base query rewriting per §4.2. Implements ordered separating sets, cell-based distribution estimation, and rewriting for selection, projection, GROUP BY, and joins. |
| **Optimized mGraph-QE** (`SetQueryRewriterExecuterOptimized.py`) | All base rewriting plus estimation-aware optimizations from §4.3: selective filter pushdown, deterministic guards, zero-mass pruning, separator-aware filtering, shared computation via merged statistics CTEs, and optional stratified sampling. |
| **Probabalistic Pattrena mGraph-QE Ranking** (`RankingQueryExecuter.py`) | Pattern-based top-fraction approach. Groups tuples by missingness pattern, ranks by estimated probability mass, and returns the top-*f* fraction. Uses merged stats with theta-based grouping for fast execution on large tables. |
| **Certain answers** (`RunnerCertainAnswers.py`) | Conservative baseline: returns only tuples with no missing values in query-relevant attributes (IS NOT NULL guards). |

## Metrics


- **TV\_prob**: Normalized total variation over conditional probability distributions: (1/2) Σ\_t |P̃(t)/Z̃ − P\*(t)/Z\*|.
- **Confidence intervals**: Hoeffding, CLT/Wald, Wilson, and Delta method.
- **Interval metrics**: Empirical coverage, mass-weighted mean width, Winkler score.
- **Bias**: Jensen-Shannon divergence between predicted and ground-truth distributions.

## Project Structure

### Queries Without Aggregation (Set Queries)

Handles SELECT-WHERE, projection, GROUP BY, and join queries that return **sets of tuples** with associated membership probabilities.

```
├── SetQueryRewriterExecuter.py          # Base query rewriter/executor (mGraph-QE, §4.2)
├── SetQueryRewriterExecuterOptimized.py # Optimized rewriter with selective pushdown (§4.3)
├── RankingQueryExecuter.py              # Top-fraction pattern-based mGraph-QE Ranking approach
├── RunnerSetQueriy.py                   # Set query runner with metric computation
├── RunnerCertainAnswers.py              # Certain answers baseline (IS NOT NULL guards)
├── CompareSetQueryApproaches.py         # Main experiment runner and comparison
├── mnar_set_queries.json                # Query config (datasets, queries, metadata)
└── certain_answers_queries.json         # Certain answers query config
```

### Queries With Aggregation (AVG, SUM, COUNT)

Compare mGraph-QE againist two baselines 1)**distribution-based estimates** (full probability distribution over possible aggregate values) or 2) **interval-based bounds** (wide [lower, upper] ranges for the aggregate result).

```
├── QueryRewriterExecuter.py             # Aggregate query rewriter: rewrites AVG/SUM/COUNT
│                                        #   queries under MCAR/MAR/MNAR into SQL with
│                                        #   group-level estimation and variance computation
├── QueryDistributionBasedEstimator.py   # Distribution recovery: builds joint probability
│                                        #   tables P(X_o, X_m, R) and computes full
│                                        #   distributions over possible aggregate outcomes
├── QueryIntervalBasedEstimator.py       # Interval-based estimator: implements Algorithm 2
│                                        #   (Zhang et al. 2019) for tight [a, b] bounds on
│                                        #   AVG queries via greedy substitution of missing
│                                        #   values with domain endpoints
└── real_mnar_agg.json                   # Aggregate query config
```

### Shared Utilities and Data Repair

```
├── bias_utils.py                        # JSD and bias metrics
├── myDataAnalyzer.py                    # Result logging utilities
├── RunnerRepair.py                      # Data repair via imputers (MICE, MissForest, etc.)
├── create_safe_discretized_temp_json.py # Discretize numeric columns for safe evaluation
├── requirements_venv311.txt             # Python dependencies
└── max-mmd/                             # Max-MMD optimization (Frank-Wolfe, projected gradient)
```

## Requirements

- **Python** 3.11+
- **PostgreSQL** (tested with 13+)
- Dependencies listed in `requirements_venv311.txt`

Core dependencies:

```
psycopg2==2.9.10
pandas==2.1.4
numpy==1.26.4
scipy==1.11.4
scikit-learn==1.4.2
func_timeout==4.3.5
tabulate==0.9.0
```

## Setup

1. **Create a virtual environment and install dependencies:**

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements_venv311.txt
```

2. **Configure PostgreSQL connection** in `RunnerSetQuery.py` (or whichever runner you use):

```python
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="your_db",
    user="your_user",
    password="your_password"
)
```

3. **Prepare data:** Place MNAR and complete CSV files in the paths referenced by the JSON config files (e.g., `mnar_set_queries.json`). The runner will automatically load CSVs into PostgreSQL tables.

## Usage

### Run the full comparison

```bash
python CompareSetQueryApproaches.py
```

This evaluates all approaches (mGraph-QE, Optimized mGraph-QE, mGraph-QE Ranking) on the queries defined in the JSON config, computes TV distances, confidence intervals, and timing, and writes results to `results_final.txt`.

### Configure experiments

Edit the constants at the top of `CompareSetQueryApproaches.py`:

```python
JSON_PATH = "mnar_set_queries.json"     # Query config file
TIMEOUT_PER_QUERY = 300                  # Per-approach timeout (seconds)
RUN_CERTAIN_ANSWERS = False              # Include certain answers baseline
INTERVAL_MODE = "wilson"                 # Confidence interval method
INTERVAL_ALPHA = 0.05                    # Significance level
```

### JSON config format

Each JSON config defines dataset groups, where each group contains blocks of queries:

```json
{
  "group_name": [
    {
      "csv": "path/to/mnar_data.csv",
      "table": "table_name",
      "complete_csv": "path/to/ground_truth.csv",
      "complete_table": "gt_table_name",
      "missing_attrs_single": ["attr1", "attr2"],
      "ordering_single": {
        "attr1": { "name": "attr1", "conditioning": ["X1", "X2"], "condition": "attr1 > 0" },
        "attr2": { "name": "attr2", "conditioning": ["X1"], "condition": "attr2 = 'yes'" }
      },
      "queries": [
        "SELECT col1, col2 FROM table_name WHERE col1 > 10 AND attr1 > 0"
      ]
    }
  ]
}
```


### Selective Filter Pushdown

Predicates on complete attributes (φ(B)) are placed in `WHERE` clauses of the tuple-level scan for early filtering. Predicates on incomplete attributes (φ(A)) are embedded as conditional aggregation expressions inside the statistics CTE, preserving the full sample for probability estimation.

### Confidence Intervals

For each tuple's probability estimate, the framework computes confidence intervals using one of four methods (Hoeffding, CLT/Wald, Wilson, Delta), with Šidák correction for multiple factors. Intervals are propagated through the product of independent factors.
