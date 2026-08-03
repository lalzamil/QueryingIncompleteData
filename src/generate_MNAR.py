import numpy as np
import pandas as pd
import json
# -----------------------------
# Synthetic MNAR / MAR / MCAR dataset generator
# -----------------------------
np.random.seed(42)

# ---- Parameters you can tweak ----
n = 1000000         # hard-coded dataset size
p_mcar_education = 0.10

# ---- 1. Generate fully-observed latent data (no missing values yet) ----
# education: 0 (low), 1 (mid), 2 (high)
education = np.random.choice([0, 1, 2], size=n, p=[0.3, 0.5, 0.2])

# job depends on education (edge: education ➜ job)
p_job_given_edu = {0: 0.25, 1: 0.55, 2: 0.85}
job = np.array([np.random.binomial(1, p_job_given_edu[e]) for e in education])

# income depends on education & job (edge: education, job ➜ income)
income_mean = 30_000 + education * 15_000 + job * 10_000
income = np.random.normal(loc=income_mean, scale=5_000)

# loan depends on education & income (edges: education, income ➜ loan)
loan = np.random.normal(loc=income * 0.30 + education * 4_000, scale=5_000)

# tax depends on income (edge: income ➜ tax)
tax = income * 0.25 + np.random.normal(0, 1_000)

df_complete = pd.DataFrame(
    dict(
        education=education,
        job=job,
        income=income,
        loan=loan,
        tax=tax,
    )
)

# ---- 2. Inject missingness according to the m-graph ----
# MCAR: education v
R_education = np.random.binomial(1, p_mcar_education, n)
education_obs = education.astype("float")
education_obs[R_education == 1] = np.nan

# MAR: tax | job
p_miss_tax = np.where(job == 1, 0.05, 0.20)
R_tax = np.random.binomial(1, p_miss_tax)
tax_obs = tax.copy()
tax_obs[R_tax == 1] = np.nan

# MNAR: income | education + job   (education may itself be missing)
logit_income = -2 + 0.8 * education + 1.0 * job
p_miss_income = 1 / (1 + np.exp(-logit_income))
R_income = np.random.binomial(1, p_miss_income)
income_obs = income.copy()
income_obs[R_income == 1] = np.nan

# MNAR: loan | education + income  (income may be missing)
logit_loan = -3 + 0.6 * education + 0.00005 * income
p_miss_loan = 1 / (1 + np.exp(-logit_loan))
R_loan = np.random.binomial(1, p_miss_loan)
loan_obs = loan.copy()
loan_obs[R_loan == 1] = np.nan

df_mnar = pd.DataFrame(
    dict(
        education=education_obs,
        job=job,
        income=income_obs,
        loan=loan_obs,
        tax=tax_obs,
    )
)

# ---- 3. Save both versions to disk ----
complete_path = "mnarData/complete_data_for_MNAR.csv"
mnar_path = "mnarData/mnar_data.csv"
df_complete.to_csv(complete_path, index=False)
df_mnar.to_csv(mnar_path, index=False)


# --- Mechanism + parent mapping -------------------------------------------
mechanism_meta = {
    "education": {"mechanism": "MCAR", "parents": []},
    "job":       {"mechanism": "FullyObserved", "parents": []},
    "income":    {"mechanism": "MNAR", "parents": ["education", "job"]},
    "loan":      {"mechanism": "MNAR", "parents": ["education", "income"]},
    "tax":       {"mechanism": "MAR",  "parents": ["job"]},
}

# ---- Save as JSON side-car -----------------------------------------------
meta_parents_path = "mnarData/missingness_mechanisms_with_parents.json"
with open(meta_parents_path, "w") as f:
    json.dump(mechanism_meta, f, indent=2)

# ---- Also export an easy-to-read table ------------------------------------
meta_df = (
    pd.DataFrame.from_dict(mechanism_meta, orient="index")
      .reset_index()
      .rename(columns={"index": "attribute"})
)
# stringify parents list for display
meta_df["parents"] = meta_df["parents"].apply(lambda lst: ", ".join(lst))



print("Mechanism metadata (with parents) saved →", meta_parents_path)


# # ---- 4. Quick sanity check ----
# summary = df_mnar.isna().mean().round(3).rename("missing_fraction")
# display_df = pd.DataFrame(summary)



# print("Complete dataset shape:", df_complete.shape)
# print("MNAR dataset shape    :", df_mnar.shape)


# metadata = {
#     "education": "MCAR",
#     "job": "FullyObserved",
#     "income": "MNAR",
#     "loan": "MNAR",
#     "tax": "MAR",
# }

# # Save as a side-car JSON file
# meta_path = "/mnt/data/missingness_metadata.json"
# with open(meta_path, "w") as f:
#     json.dump(metadata, f, indent=2)

# # Add R-indicator columns so the mechanism itself is recorded row-wise
# df_with_R = df_mnar.copy()
# df_with_R["R_education"] = R_education
# df_with_R["R_tax"] = R_tax
# df_with_R["R_income"] = R_income
# df_with_R["R_loan"] = R_loan

# indicators_path = "/mnt/data/mnar_with_indicators.csv"
# df_with_R.to_csv(indicators_path, index=False)

# print("Side-car metadata JSON saved →", meta_path)
# print("Dataset with R-indicator columns saved →", indicators_path)
