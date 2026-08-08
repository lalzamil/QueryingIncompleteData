# MNAR-Repair

This directory contains the implementation of MNAR-Repair from Section 8 of *Causality-Aware Query Answering on Incomplete Data*.

MNAR-Repair takes an incomplete relation, its m-graph metadata, and optional positive repair costs. It identifies a set of incomplete attributes whose repair eliminates every edge from an incomplete attribute to the missingness mechanism of another incomplete attribute. It then imputes all missing cells in the selected attributes and writes the repaired relation and updated m-graph.

The implementation follows the weighted Set Cover formulation in the paper. For an MNAR edge `A -> R_B`, repairing either `A` or `B` eliminates the edge. At each iteration, the algorithm selects the attribute with the largest ratio between the number of remaining MNAR edges it eliminates and its repair cost. Ties are resolved by the lower missingness rate and then by attribute name so that runs are reproducible.

## Directory contents

```text
mnar_repair/              Implementation and command-line interface
tests/                    Unit and command-line tests
tests/data/               Small incomplete relation, m-graph, and repair costs
requirements.txt          Python dependencies
```

## Requirements

The code requires Python 3.11 or later. From the repository root, create an environment and install the dependencies:

```bash
python3.11 -m venv .venv-repair
source .venv-repair/bin/activate
pip install -r MNAR-Repair/requirements.txt
```

## Input format

The relation is a CSV file. An empty CSV field denotes a missing value.

The m-graph metadata is a JSON object with one entry per relation attribute:

```json
{
  "education": {"mechanism": "MCAR", "parents": []},
  "tax": {"mechanism": "MNAR", "parents": ["education"]}
}
```

`parents` lists the relation attributes with edges into the attribute's missingness mechanism. Every listed attribute must also occur in the relation and in the JSON object. The supported mechanism labels are `FullyObserved`, `MCAR`, `MAR`, and `MNAR`.

Repair costs are optional. When supplied, they are a JSON object that maps incomplete attributes to positive numbers. Unspecified costs default to one.

## Run the included example

The `simple` imputer uses the observed median for a numerical attribute and the observed mode for a categorical attribute. It is included to make the example and tests deterministic.

```bash
cd MNAR-Repair
python -m mnar_repair \
  tests/data/example_relation.csv \
  tests/data/example_mgraph.json \
  --costs tests/data/example_costs.json \
  --imputer simple \
  --output-dir example_output
```

The command writes:

- `example_output/repaired_relation.csv`
- `example_output/repaired_mgraph.json`
- `example_output/repair_set.json`

The command refuses to replace an existing output file unless `--force` is specified.

## Markov Blanket Imputation

The experiments use Markov Blanket Imputation (MBI) as one model-based option. The following command repairs the selected attributes with the random-forest MBI implementation:

```bash
python -m mnar_repair \
  tests/data/example_relation.csv \
  tests/data/example_mgraph.json \
  --imputer mbi \
  --output-dir example_output_mbi
```

For each selected attribute, MBI uses the available attributes in its Markov blanket. If the supplied m-graph metadata gives an empty blanket, MBI uses the observed marginal distribution through the median or mode. The Python interface also accepts a user-provided imputation function, which permits the selected attributes to be repaired by an expert or another trained imputation model.

## Python interface

```python
import json
import pandas as pd

from mnar_repair import repair_relation

relation = pd.read_csv("tests/data/example_relation.csv")
with open("tests/data/example_mgraph.json", encoding="utf-8") as handle:
    mgraph = json.load(handle)

result = repair_relation(relation, mgraph, imputer="simple")
print(result.repair_set)
result.relation.to_csv("repaired_relation.csv", index=False)
```

`repair_relation` does not modify the input DataFrame or m-graph object. Its result contains the selected repair set, the repaired relation, and the updated m-graph.

## Tests

Run all tests from `MNAR-Repair/`:

```bash
python -m unittest discover -s tests -v
```

The tests verify the incident-edge gain used by MNAR-Repair, positive repair costs, deterministic tie resolution, preservation of observed cells, elimination of all MNAR edges, command-line outputs, and protection against accidental output replacement.
