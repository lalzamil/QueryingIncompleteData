# Querying Incomplete Data

This repository contains the PostgreSQL and Python implementation used to evaluate queries over incomplete data in the paper.
The snapshot includes the corrected code used for the full-data experiments in July and August 2026.

## Implemented methods

- **CAEX** implements the causal-aware extensional evaluation from Section 5. The main full-data runner is `src/RunSectionComparisonsFullData.py`.
- **CADE** implements the causal-aware direct estimation from Section 7. Aggregation queries use `src/QueryRewriterExecuter.py`, and non-aggregation queries use `src/nonAgg_direct.py`.
- **QE** constructs factor distributions and evaluates the union of the queries obtained from the sampled valuations. Its full-data runners are `src/RunQEFromFactorDistributionsFullData.py`, `src/RunQEMCARMARSelectedFullData.py`, and `src/RunRealFactorizableQE.py`.
- **Certain** is implemented in `src/RunnerCertainAnswersBagTVD.py` and the full-data runners whose names begin with `RunCertainAnswers`.
- **Interval-Based** and **Distribution-Centric** are implemented in `src/QueryIntervalBasedEstimator.py` and `src/QueryDistributionBasedEstimator.py`.

The repository also retains the PostgreSQL MCDB implementation and factor sampler used during method validation in `src/MCDBPostgresNative.py` and `src/FactorSamplerPostgres.py`.
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

Place the prepared datasets below `data/` as described in `DATA.md`.
The files in `configs/` define the injected MCAR, MAR, and factorizable MNAR queries, the real-world queries, and the meaningful two- and four-relation join queries.
All paths in these files are relative to the repository root.

The full-data runs use `--rows 0`.
The QE comparisons use `H=783` sampled valuations, and each query has a 300-second timeout in the final comparison runners.

## Running the methods

Run CAEX and CADE on the injected factorizable MNAR data:

```bash
python src/RunSectionComparisonsFullData.py \
  --methods CAEX,CADE \
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

## Tests

The tests create temporary PostgreSQL tables and therefore require a running database configured through the PostgreSQL environment variables.

```bash
PYTHONPATH=src python tests/TestFactorSamplerPostgres.py
PYTHONPATH=src python tests/TestMCDBPostgresNative.py
PYTHONPATH=src python tests/TestLazyCAMC.py
```

The tests compare optimized tuple-bundle evaluation with explicit repair-indexed evaluation and verify the factor sampler on small relations.

## Reproducing Figure 3

`analysis/generate_figure3.py` reads the CSV results placed below `analysis/inputs/` and creates the execution-time figure and query-quality table.
It expects the following layout:

```text
analysis/inputs/
├── cade/
│   ├── figure3_cade_mechanism_runtime.csv
│   └── table3_cade_mechanism_quality.csv
├── caex_mnar.csv
├── caex_selected/{MCAR,MAR}/{bank,nyc,bitcoin}.csv
├── certain/{bank,nyc,bitcoin}.csv
├── certain_runtime.csv
├── qe_mnar/{bank,nyc,bitcoin}.csv
└── qe_selected/{MCAR,MAR}/{bank,nyc,bitcoin}.csv
```

`analysis/generate_without_caex.py` creates the version that omits CAEX.
The scripts validate the expected query counts and reject row-limited measurements.
