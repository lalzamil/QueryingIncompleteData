# Data layout

The datasets are not distributed in this repository.
Download each dataset from its original source and place the prepared files below `data/`.
The JSON files in `configs/` give the exact relative path expected for every query.

The experiment code uses the following directories when all comparisons are run:

```text
data/
├── CompleteFourRelationsData/
├── CompleteSemanticJoinData/
├── CompleteSemanticTwoRelationData/
├── Injected_JoinsData/
├── MarJoinsData/
├── MI/
├── MNAR1Data/
├── Mnar1FourRelationsData/
├── Mnar1JoinsData/
├── SemanticJoinData/
├── SemanticTwoRelationData/
├── joinsData/
├── mcdb_test_data/
└── rwDatasets/
```

`src/PrepareBankFourRelationWorkload.py`, `src/PrepareNYCBitcoinFourRelationWorkloads.py`, and `src/PrepareRealFactorizableRelations.py` construct the meaningful relation decompositions used by the join queries.
These scripts preserve the original tuple identifier and do not inject additional missing values.

The query configurations include the relative paths, relation names, incomplete attributes, causes, factorization order, and SQL query text used in the experiments.
Run commands from the repository root so that these paths resolve consistently.
