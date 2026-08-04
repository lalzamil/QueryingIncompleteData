# Causality-Aware Query Answering on Incomplete Data

<!-- Add the technical-report link here:
**Technical report:** [Technical report](PASTE_LINK_HERE)
-->

This repository contains the PostgreSQL and Python implementation used to evaluate queries over incomplete data in the paper.


## Implemented methods

- **CADE** implements the causal-aware query-evaluation methods from Sections 5 and 6. The main full-data runner is `src/RunSectionComparisonsFullData.py`. Aggregation queries use `src/QueryRewriterExecuter.py`, and non-aggregation queries use `src/nonAgg_direct.py`.
- **QE** constructs factor distributions and evaluates the union of the queries obtained from the sampled valuations. Its full-data runners are `src/RunQEFromFactorDistributionsFullData.py`, `src/RunQEMCARMARSelectedFullData.py`, and `src/RunRealFactorizableQE.py`.
- **Certain** is implemented in `src/RunnerCertainAnswersBagTVD.py` and the full-data runners whose names begin with `RunCertainAnswers`.
- **Interval-Based** and **Distribution-Centric** are implemented in `src/QueryIntervalBasedEstimator.py` and `src/QueryDistributionBasedEstimator.py`.

The marked-null implementations and runners are identified by `marked` in their file names.

## Repository structure

```text
analysis/       Figure and table generation
configs/        Query definitions and factorization metadata
scripts/        Commands used for full-data and multi-relation runs
src/            Implementations, data preparation, and experiment runners
tests/          PostgreSQL correctness tests
DATA.md         Required dataset layout
```

Generated datasets, result files, logs, virtual environments, and credentials are excluded from Git.

## Requirements

The experiments were run with Python 3.11.7 and PostgreSQL 15.8.
The Python package versions are recorded in `requirements.txt`.

Create an environment and install the packages:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

Set the PostgreSQL connection without placing a password in the repository:

```bash
export PGHOST=127.0.0.1
export PGPORT=5433
export PGDATABASE=mydb
export PGUSER=postgres
export PGPASSWORD='your-password'
```

The command-line runners also accept `--db-host`, `--db-port`, `--db-name`, `--db-user`, and `--db-password`.

## Data and query configurations

Download the datasets using the links in `DATA.md` and place the prepared files below `data/`.
The files in `configs/` define the injected MCAR, MAR, and factorizable MNAR queries, the real-world queries, and the meaningful two- and four-relation join queries.
All paths in these files are relative to the repository root.

The full-data runs use `--rows 0`.
The QE comparisons use `H=783` sampled valuations, and each query has a 300-second timeout in the final comparison runners.

## Running the methods

Run CADE on the injected factorizable MNAR data:

```bash
python src/RunSectionComparisonsFullData.py \
  --methods CADE \
  --datasets bank,nyc,bitcoin \
  --rates 5,10,20 \
  --workloads set,aggregate \
  --rows 0 \
  --timeout 300 \
  --output results/caex_cade.csv
```

Run QE on the same non-repeating-null queries:

```bash
python src/RunQEFromFactorDistributionsFullData.py \
  --null-semantics nonrepeating \
  --datasets bank,nyc,bitcoin \
  --rates 5,10,20 \
  --workloads set,aggregate \
  --rows 0 \
  --h 783 \
  --timeout 300 \
  --output results/qe_nonrepeating.csv
```

Run QE on marked nulls:

```bash
python src/RunLikeApxMarkedFullData.py \
  --datasets bank,nyc,bitcoin \
  --rates 5,10,20 \
  --workloads set,aggregate \
  --rows 0 \
  --h 783 \
  --timeout 300 \
  --output results/qe_marked.csv
```

The scripts in `scripts/` run the multi-relation comparisons for separate PostgreSQL instances.
Use the `PYTHON` environment variable to select a Python executable.
