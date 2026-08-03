#!/usr/bin/env bash
set -euo pipefail

dataset=$1
method=$2
port=$3
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
result_root=$repo/psql_results/section_comparisons/full_four_relation_20260728
python_bin=${PYTHON:-python3}
config=$repo/configs/${dataset}_four_relation_queries_supported.json
method_dir=$result_root/$dataset/$method

mkdir -p "$method_dir"
cd "$repo"

"$python_bin" src/RunFourRelationSetComparisons.py \
  --config "$config" \
  --set-config "$repo/configs/mnar_set_queries.json" \
  --rates 5,10,20 \
  --rows 0 \
  --h 783 \
  --timeout 300 \
  --seed 20260722 \
  --only-queries 13,14 \
  --methods "$method" \
  --db-host 127.0.0.1 \
  --db-port "$port" \
  --db-name "${PGDATABASE:-mydb}" \
  --db-user "${PGUSER:-postgres}" \
  --db-password "${PGPASSWORD:-}" \
  --output "$method_dir/set.csv"

"$python_bin" src/RunFourRelationAggregateComparisons.py \
  --config "$config" \
  --rates 5,10,20 \
  --rows 0 \
  --h 783 \
  --timeout 300 \
  --seed 20260722 \
  --only-queries 13,14 \
  --methods "$method" \
  --db-host 127.0.0.1 \
  --db-port "$port" \
  --db-name "${PGDATABASE:-mydb}" \
  --db-user "${PGUSER:-postgres}" \
  --db-password "${PGPASSWORD:-}" \
  --output "$method_dir/aggregate.csv"
