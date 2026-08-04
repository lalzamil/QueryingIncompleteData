# Data layout

The datasets are not distributed in this repository.
Download each dataset from its original source and place the prepared files below `data/`.
The JSON files in `configs/` give the exact relative path expected for every query.

## Dataset downloads

The following links correspond to the datasets reported in the paper.
The Download links return the source archive or CSV when the provider supports a direct download.

| Dataset | Official source | Download |
|---|---|---|
| Bank Marketing | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) | [ZIP](https://archive.ics.uci.edu/static/public/222/bank+marketing.zip) |
| NYC Taxi Trip Duration | [Kaggle](https://www.kaggle.com/datasets/parisrohan/nyc-taxi-trip-duration) | [ZIP](https://www.kaggle.com/api/v1/datasets/download/parisrohan/nyc-taxi-trip-duration) |
| Bitcoin Heist Ransomware Address | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset) | [ZIP](https://archive.ics.uci.edu/static/public/526/bitcoinheistransomwareaddressdataset.zip) |
| Building Permits | [City of Chicago Data Portal](https://data.cityofchicago.org/Buildings/Building-Permits/ydr8-5enu) | [CSV](https://data.cityofchicago.org/api/views/ydr8-5enu/rows.csv?accessType=DOWNLOAD) |
| Street Construction Permits | [NYC Open Data](https://data.cityofnewyork.us/Transportation/Street-Construction-Permits-2022-Present-/tqtj-sjs8) | [CSV](https://data.cityofnewyork.us/api/views/tqtj-sjs8/rows.csv?accessType=DOWNLOAD) |
| Employees Info | [City of Chicago Data Portal](https://data.cityofchicago.org/Administration-Finance/Current-Employee-Names-Salaries-and-Position-Title/xzkq-xp2w/about_data) | [CSV](https://data.cityofchicago.org/api/views/xzkq-xp2w/rows.csv?accessType=DOWNLOAD) |
| SF Salaries | [Kaggle](https://www.kaggle.com/datasets/kaggle/sf-salaries) | [ZIP](https://www.kaggle.com/api/v1/datasets/download/kaggle/sf-salaries) |
| Heart Health | [Kaggle](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) | [ZIP](https://www.kaggle.com/api/v1/datasets/download/kamilpytlak/personal-key-indicators-of-heart-disease) |
| Student Admission Records | [Kaggle](https://www.kaggle.com/datasets/zeeshier/student-admission-records) | [ZIP](https://www.kaggle.com/api/v1/datasets/download/zeeshier/student-admission-records) |
| Aircraft Performance | [Kaggle](https://www.kaggle.com/datasets/heitornunes/aircraft-performance-dataset-aircraft-bluebook) | [ZIP](https://www.kaggle.com/api/v1/datasets/download/heitornunes/aircraft-performance-dataset-aircraft-bluebook) |
| Medical Condition Prediction | [Kaggle](https://www.kaggle.com/datasets/marius2303/medical-condition-prediction-dataset) | [ZIP](https://www.kaggle.com/api/v1/datasets/download/marius2303/medical-condition-prediction-dataset) |

Kaggle may require sign-in or API credentials before a download begins.
The injected missingness files and the normalized relations are derived from these source datasets and are not separate external datasets.
After downloading the source files, prepare them using the repository scripts and retain the relative paths specified in `configs/`.

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
