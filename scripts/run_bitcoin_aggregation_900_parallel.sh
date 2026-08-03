#!/usr/bin/env bash
set -euo pipefail

repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python=${PYTHON:-python3}
result_root="$repo/psql_results/section_comparisons/full_data_20260728_bitcoin_aggregation_900s_parallel"
method=$1
rate=$2
port=$3

method_dir="$result_root/$method"
mkdir -p "$method_dir" "$result_root/logs"

cd "$repo"

"$python" src/RunSectionComparisonsFullData.py \
  --datasets bitcoin \
  --rates "$rate" \
  --workloads aggregate \
  --methods "$method" \
  --rows 0 \
  --h 783 \
  --timeout 900 \
  --query-limit 8 \
  --db-host 127.0.0.1 \
  --db-port "$port" \
  --force-reload \
  --output "$method_dir/rate${rate}_q1_q8.csv"

"$python" src/RunFourRelationAggregateComparisons.py \
  --config configs/bitcoin_four_relation_queries_supported.json \
  --rates "$rate" \
  --rows 0 \
  --h 783 \
  --timeout 900 \
  --only-queries 13,14 \
  --methods "$method" \
  --db-host 127.0.0.1 \
  --db-port "$port" \
  --output "$method_dir/rate${rate}_q13_q14.csv"
