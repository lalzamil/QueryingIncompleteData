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
The files in `configs/` define the injected MCAR, MAR, and factorizable MNAR queries, the real-world queries, and the meaningful multi-relation join queries.
All paths in these files are relative to the repository root.

The full-data runs use `--rows 0`.
The QE comparisons use `H=783` sampled valuations, and each query has a 300-second timeout in the final comparison runners.

## Multi-relation schemas

The join queries use the following meaningful relation decompositions.
The join key appears in every relation and preserves the association between attributes from the same source tuple.
The decomposition does not introduce additional missing values.

### Bank Marketing

Join key: `tuple_id`.

```text
customer_profile(tuple_id, age, job, marital, education)
account_status(tuple_id, default, balance, housing, loan)
current_contact(tuple_id, contact, day, month, duration, campaign, y)
campaign_history(tuple_id, pdays, previous, poutcome)
```

Configuration: `configs/bank_semantic_join_queries.json`.

### NYC Taxi Trip Duration

Join key: `tuple_id`.

```text
trip(tuple_id, vendor_id, passenger_count, store_and_fwd_flag, trip_duration)
pickup(tuple_id, pickup_datetime, pickup_longitude, pickup_latitude)
dropoff(tuple_id, id, dropoff_datetime, dropoff_longitude, dropoff_latitude)
```

Configuration: `configs/nyc_semantic_join_queries.json`.

### Bitcoin Heist Ransomware Address

Join key: `tuple_id`.

```text
address_class(tuple_id, address, label)
observation_time(tuple_id, year, day)
transaction_graph(tuple_id, ID, length, weight, count, looped, neighbors, income)
```

Configuration: `configs/bitcoin_semantic_join_queries.json`.

### Student Admission Records

Join key: `id`.

```text
student_applicant_profile(id, age, gender)
student_academic_record(id, admission_test_score, high_school_percentage)
student_admission_record(id, name, city, admission_status)
```

### Aircraft Performance

Join key: `id`.

```text
aircraft_description(id, model, company, engine_type, fuel_gal_lbs, gross_weight_lbs, empty_weight_lbs, length_ft_in, height_ft_in, wing_span_ft_in)
aircraft_flight_performance(id, max_speed_knots, rcmnd_cruise_knots, stall_knots_dirty, range_n_m)
aircraft_climb_ceiling(id, all_eng_service_ceiling, eng_out_service_ceiling, all_eng_rate_of_climb, eng_out_rate_of_climb)
aircraft_takeoff_landing(id, takeoff_over_50ft, takeoff_ground_run, landing_over_50ft, landing_ground_roll)
```

### Medical Condition Prediction

Join key: `id`.

```text
medical_patient(id, full_name, age, gender, smoking_status)
medical_measurements(id, bmi, blood_pressure, glucose_levels)
medical_diagnosis(id, condition)
```

The Student Admission, Aircraft Performance, and Medical Condition schemas are defined in `configs/real_factorizable_10_queries.json`.

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
