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
QUERY_CATALOG.md Query lists and missingness metadata
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
The correspondence among datasets, query lists, missingness types, incomplete attributes, and their causes is given in [`QUERY_CATALOG.md`](QUERY_CATALOG.md).
All paths in these files are relative to the repository root.

The full-data runs use `--rows 0`.
The QE comparisons use `H=783` sampled valuations, and each query has a 300-second timeout in the final comparison runners.

## Multi-relation schemas

The join queries use the following meaningful relation decompositions.
The join key appears in every relation and preserves the association between attributes from the same source tuple.
The decomposition does not introduce additional missing values.
The relation counts match the maximum number of relations reported for each dataset in Table 2 of the paper.

| Dataset | Number of relations |
|---|---:|
| Bank Marketing | 4 |
| NYC Taxi Trips | 4 |
| Bitcoin Heist | 4 |
| Building Permits | 2 |
| Street Construction Permits | 2 |
| Employees Info | 2 |
| SF Salaries | 2 |
| Heart Health | 2 |
| Student Admission | 3 |
| Aircraft Performance | 4 |
| Medical Condition | 3 |
| Communities & Crime | 2 |
| NHANES | 2 |

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
trip_route [r1](tuple_id, vendor_id, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, trip_duration)
trip_passengers [r2](tuple_id, passenger_count)
trip_handling [r3](tuple_id, store_and_fwd_flag)
trip_time [r4](tuple_id, id, pickup_datetime, dropoff_datetime)
```

The labels in brackets identify the physical partitions in `configs/nyc_four_relation_queries.json`.

### Bitcoin Heist Ransomware Address

Join key: `tuple_id`.

```text
transaction_graph [r1](tuple_id, label, weight, day, length, neighbors, count, looped)
transaction_income [r2](tuple_id, income)
observation_year [r3](tuple_id, year)
address_identity [r4](tuple_id, address)
```

The labels in brackets identify the physical partitions in `configs/bitcoin_four_relation_queries.json`.

### Building Permits

Join key: `customer_id`.

```text
permit_record(customer_id, id, permit#, permit_status, permit_milestone, permit_type, review_type, application_start_date, issue_date, processing_time, work_type, work_description)
permit_fees(customer_id, id, building_fee_paid, zoning_fee_paid, other_fee_paid, subtotal_paid, building_fee_unpaid, zoning_fee_unpaid, other_fee_unpaid, subtotal_unpaid, building_fee_waived, building_fee_subtotal, zoning_fee_subtotal, other_fee_subtotal, zoning_fee_waived, other_fee_waived, subtotal_waived, total_fee)
```

### Street Construction Permits

Join key: `permitnumber`.

```text
permit_record(permitnumber, applicationtrackingid, sequencenumber, applicationtypeshortdesc, permitstatusid, permitstatusshortdesc, permitseriesid, permitseriesshortdesc, permittypeid, permittypedesc, permitnumberofzones, permitlinearfeet, permittotalsqfeet, permitestimatednumberofcuts, equipmenttypedesc, numberofcontainers, numberofminicontainers, specificstipulations, previouspermitnumber, nextpermitnumber, emergencyissuedate, permitissuedate, issuedworkstartdate, issuedworkenddate, boroughname, permitpurposecomments, permitlocationcomments, pavementshortdesc, sidewalkshortdesc, createdon, modifiedon, oftcode)
permittee(permitnumber, permitteename)
```

### Employees Info

Join key: `employeeid`.

```text
employee_profile(employeeid, name, department, full_or_part_time)
employee_compensation(employeeid, job_titles, salary_or_hourly, typical_hours, annual_salary, hourly_rate)
```

### SF Salaries

Join key: `id`.

```text
employee_record(id, employeename, agency, status)
employee_compensation(id, jobtitle, basepay, overtimepay, otherpay, benefits, totalpay, totalpaybenefits, year, notes)
```

### Heart Health

Join key: `person_id`.

```text
person_profile(person_id, State, Sex, RaceEthnicityCategory, HeightInMeters, WeightInKilograms, BMI, SmokerStatus, ECigaretteUsage, AlcoholDrinkers)
health_record(person_id, GeneralHealth, PhysicalHealthDays, MentalHealthDays, LastCheckupTime, PhysicalActivities, SleepHours, RemovedTeeth, HadHeartAttack, HadAngina, HadStroke, HadAsthma, HadSkinCancer, HadCOPD, HadDepressiveDisorder, HadKidneyDisease, HadArthritis, HadDiabetes, DeafOrHardOfHearing, BlindOrVisionDifficulty, DifficultyConcentrating, DifficultyWalking, DifficultyDressingBathing, DifficultyErrands, ChestScan, HIVTesting, FluVaxLast12, PneumoVaxEver, TetanusLast10Tdap, HighRiskLastYear, CovidPos)
```

The Building Permits, Street Construction Permits, Employees Info, SF Salaries, and Heart Health schemas are defined by the relation files referenced in `configs/all_queries.json`.

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

### Communities & Crime

Join key: `row_id`, assigned during relation preparation.

```text
community_characteristics(row_id, feat_001 through feat_100)
policing_and_crime(row_id, feat_101 through feat_146)
```

The feature numbers preserve the column order of the prepared Communities & Crime relation.

### NHANES

Join key: `SEQN`, the NHANES sample-person identifier.

```text
participant_profile(SEQN, RIDAGEYR, RIAGENDR, RIDRETH3, DMDEDUC2, INDHHIN2)
health_record(SEQN, BMXBMI, BMXWAIST, BMXHT, BMXWT, BPXSY1, BPXDI1, LBXTC, LBDHDD, LBXTR, LBDLDL, LBXGH, LBXGLU, LBXSCR, LBXSUA, LBXSGL, BPQ020, BPQ080, BPQ100D, DIQ010, DIQ160, MCQ160B, MCQ160C, MCQ160E)
```

The prepared single-relation file drops `SEQN` after joining the original NHANES files.

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
  --output results/cade.csv
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
