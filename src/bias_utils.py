from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

class BiasUtil:
    def __init__(self, dedup_joint: bool = False, smooth_alpha: float = 0.0):
        """
        dedup_joint=True switches the joint bias metric to set semantics (ignores multiplicities).
        smooth_alpha>0 applies tiny Laplace smoothing to stabilize tiny samples.
        """
        self.dedup_joint = dedup_joint
        self.smooth_alpha = smooth_alpha

    @staticmethod
    def _counts(series: pd.Series) -> pd.Series:
        return series.dropna().value_counts()

    def _coverage_jsd_1d(self, s_pred: pd.Series, s_gt: pd.Series) -> float:
        """
        Symmetric coverage-aware JSD for one column (returns divergence in [0,1]).
        Penalizes both missing and extra mass by normalizing to max(n_p, n_q)
        and adding a single 'deficit' bucket to the smaller side.
        """
        vc_gt   = self._counts(s_gt)
        vc_pred = self._counts(s_pred)
        cats = vc_gt.index.union(vc_pred.index)

        p = vc_gt.reindex(cats, fill_value=0).to_numpy(dtype=float)  # GT counts
        q = vc_pred.reindex(cats, fill_value=0).to_numpy(dtype=float)  # Pred counts

        n_p, n_q = float(p.sum()), float(q.sum())
        if n_p == 0.0 and n_q == 0.0:
            return 0.0
        if n_p == 0.0 or n_q == 0.0:
            return 1.0

        M = max(n_p, n_q)
        P_aug = np.append(p / M, (M - n_p) / M)  # GT + deficit bucket
        Q_aug = np.append(q / M, (M - n_q) / M)  # Pred + deficit bucket

        if self.smooth_alpha > 0.0:
            k = P_aug.size
            P_aug = (P_aug + self.smooth_alpha) / (1.0 + self.smooth_alpha * k)
            Q_aug = (Q_aug + self.smooth_alpha) / (1.0 + self.smooth_alpha * k)

        js = jensenshannon(P_aug, Q_aug, base=2)
        return float(js**2 if js is not None and not np.isnan(js) else 0.0)

    def _coverage_jsd_joint(self, df_pred: pd.DataFrame, df_gt: pd.DataFrame, cols: List[str]) -> float:
        """
        Coverage-aware JSD on the JOINT distribution of the selected columns (single score in [0,1]).
        Treats each row's selected columns as one categorical tuple. Works for len(cols) >= 1.
        """
        if not cols:
            return 0.0
        if self.dedup_joint:
            s_gt   = df_gt[cols].drop_duplicates().apply(tuple, axis=1)
            s_pred = df_pred[cols].drop_duplicates().apply(tuple, axis=1)
        else:
            s_gt   = df_gt[cols].apply(tuple, axis=1)
            s_pred = df_pred[cols].apply(tuple, axis=1)
        return self._coverage_jsd_1d(s_pred, s_gt)

    # ------- existing helpers you already have (tuples_to_dataframe, calculate_f1, calculate_jsd, calculate_wasserstein, _jsd_with_missing) -------

    def tuples_to_dataframe(self, tuples: List[Tuple], columns: List[str]) -> pd.DataFrame:
        return pd.DataFrame(tuples, columns=columns)

    def calculate_fbeta(self, df_runner: pd.DataFrame, df_gt: pd.DataFrame, beta: float = 1.0) -> Tuple[float, float, float, float]:
        """
        Calculates Precision, Recall, F1-score, and F-beta score.
        beta=1.0 for F1, beta=2.0 for F2 (favors recall).
        Returns: P, R, F1, Fbeta
        """
        if df_runner.empty and df_gt.empty:
            return 1.0, 1.0, 1.0, 1.0
        if df_gt.empty:
            # P=0, R=1, F=0
            return 0.0, 1.0, 0.0, 0.0
        if df_runner.empty:
            # P=1, R=0, F=0
            return 1.0, 0.0, 0.0, 0.0

        common_cols = [col for col in df_gt.columns if col in df_runner.columns]
        if not common_cols:
            return 0.0, 0.0, 0.0, 0.0

        def to_hashable_set(df):
            placeholder = "__BIAS_PLACEHOLDER_NULL__"
            # Use placeholder for NaNs/None and convert to string for robust set comparison
            return set([tuple(row) for row in df[common_cols].fillna(placeholder).astype(str).to_numpy()])

        set_runner = to_hashable_set(df_runner)
        set_gt     = to_hashable_set(df_gt)

        TP = len(set_runner & set_gt)
        FP = len(set_runner - set_gt)
        FN = len(set_gt - set_runner)

        # Denominators are guaranteed > 0 here because DFs are not empty
        precision = TP / (TP + FP)
        recall    = TP / (TP + FN)

        # Calculate F1 (always calculate F1 for the return tuple)
        f1 = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.0

        # Calculate F-beta
        beta_sq = beta * beta
        fbeta_denom = (beta_sq * precision) + recall
        fbeta = ((1 + beta_sq) * precision * recall) / fbeta_denom if fbeta_denom > 0 else 0.0

        return precision, recall, f1, fbeta

    # keep your calculate_jsd / calculate_wasserstein / _jsd_with_missing unchanged if you still want diagnostics

    def measure_bias(self,
                     results_runner: List[Tuple],
                     results_gt: List[Tuple],
                     columns_runner: List[str],
                     columns_gt: List[str]) -> Dict[str, Any]:
        """
        Returns one single-score joint bias metric (coverage-aware JSD on the SELECT tuple),
        along with optional per-column diagnostics if you still print them.
        """
        df_runner = self.tuples_to_dataframe(results_runner, columns_runner)
        df_gt     = self.tuples_to_dataframe(results_gt,     columns_gt)

        # 1) set-based accuracy. Use beta=2.0 to calculate F2-Score.
        # The function returns P, R, F1, and F2 (since beta=2.0).
        precision, recall, f1, f2 = self.calculate_fbeta(df_runner, df_gt, beta=2.0)

        # 2) joint bias (JSDc - still useful for comparison, behaves like F1)
        cols_common = [c for c in columns_gt if c in df_runner.columns]
        joint_jsdc  = self._coverage_jsd_joint(df_runner, df_gt, cols_common) if cols_common else 0.0

        # 3) (optional) per-column diagnostics — (Keep your existing implementation here)
        distributional_details: Dict[str, float] = {}
        jsd_scores: List[float] = []
        cov_jsd_scores: List[float] = []
        wass_scores: List[float] = []

        if not df_gt.empty and not df_runner.empty and cols_common:
            for col in cols_common:
                # Classic JSD (shape-only)
                try:
                    dv = self.calculate_jsd(df_runner, df_gt, col)
                    if dv is not None and not np.isnan(dv):
                        distributional_details[f'{col}_jsd'] = float(dv)
                        jsd_scores.append(float(dv))
                except Exception:
                    pass
                # Coverage-aware (missing-bucket) per column
                try:
                    cv = self._jsd_with_missing(df_runner, df_gt, col)
                    if cv is not None and not np.isnan(cv):
                        distributional_details[f'{col}_jsd_coverage'] = float(cv)
                        cov_jsd_scores.append(float(cv))
                except Exception:
                    pass
                # Wasserstein if you re-enable it later:
                # try:
                #     w = self.calculate_wasserstein(df_runner, df_gt, col)
                #     if w is not None and not np.isnan(w):
                #         distributional_details[f'{col}_wasserstein'] = float(w)
                #         wass_scores.append(float(w))
                # except Exception:
                #     pass

        return {
            "precision": float(precision),
            "recall":    float(recall),
            "f1_score":  float(f1),
            "f2_score":  float(f2), # Added F2 score

            # Distributional metric (JSDc)
            "joint_jsd_coverage": float(joint_jsdc),

            # Primary bias metric: 1 - F2-score (Lower is better, prioritizing recall)
            "bias":               1.0 - float(f2),

            # optional summaries
            "avg_jsd":            float(np.mean(jsd_scores)) if jsd_scores else 0.0,
            "avg_jsd_coverage":   float(np.mean(cov_jsd_scores)) if cov_jsd_scores else 0.0,
            "avg_wasserstein":    float(np.mean(wass_scores)) if wass_scores else 0.0,
            "distributional_details": distributional_details,
        }
