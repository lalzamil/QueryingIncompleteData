from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
INPUT = OUT / "inputs"
QE_SELECTED = INPUT / "qe_selected"
QE_MNAR = INPUT / "qe_mnar"
CAEX_SELECTED = INPUT / "caex_selected"
CAEX_MNAR = INPUT / "caex_mnar.csv"
CADE_SOURCE = INPUT / "cade"
CURRENT_FIGURE3 = INPUT / "certain_runtime.csv"
CERTAIN_SOURCE = INPUT / "certain"
DATASETS = ("bank", "nyc", "bitcoin")
RATES = (5, 10, 20)
METHODS = ("CAEX", "CADE", "QE", "Certain", "Interval-Based", "Distribution-Centric")
ESTIMATOR_METHODS = ("CAEX", "CADE", "QE", "Certain")
AGGREGATION_METHODS = ("CAEX", "CADE", "QE", "Interval-Based", "Distribution-Centric")
NONAGGREGATION_METHODS = ("CAEX", "CADE", "QE", "Certain")
DATASET_LABELS = {
    "bank": "Bank Marketing",
    "nyc": "NYC Taxi Trips",
    "bitcoin": "Bitcoin Heist",
}
LABEL_TO_DATASET = {label: dataset for dataset, label in DATASET_LABELS.items()}
WORKLOAD_LABELS = {"aggregate": "Aggregation", "set": "Non-aggregation"}
LABEL_TO_WORKLOAD = {label: workload for workload, label in WORKLOAD_LABELS.items()}
SELECTED_MNAR = {
    "aggregate": (4, 7, 2, 5, 1),
    "set": (4, 7, 6, 3, 9),
}

BASELINE_RUNTIME = {
    "Interval-Based": {
        "MCAR": {
            "bank": {5: 0.008, 10: 0.008, 20: 0.008},
            "nyc": {5: 1.339, 10: 1.299, 20: 2.133},
            "bitcoin": {5: 300.0, 10: 300.0, 20: 300.0},
        },
        "MAR": {
            "bank": {5: 0.005, 10: 0.005, 20: 0.007},
            "nyc": {5: 1.113, 10: 1.091, 20: 1.023},
            "bitcoin": {5: 300.0, 10: 300.0, 20: 300.0},
        },
    },
    "Distribution-Centric": {
        "MCAR": {
            "bank": {5: 0.153, 10: 0.152, 20: 0.146},
            "nyc": {5: 1.148, 10: 0.475, 20: 0.959},
            "bitcoin": {5: 16.90, 10: 16.75, 20: 13.93},
        },
        "MAR": {
            "bank": {5: 0.154, 10: 0.236, 20: 0.150},
            "nyc": {5: 0.654, 10: 0.573, 20: 0.675},
            "bitcoin": {5: 24.07, 10: 22.52, 20: 20.88},
        },
    },
}

BASELINE_QUALITY = {
    "Interval-Based": {
        "MCAR": {
            "bank": {5: (100.0, 0.879), 10: (100.0, 1.242), 20: (100.0, 1.678)},
            "nyc": {5: (100.0, 0.918), 10: (100.0, 1.054), 20: (100.0, 1.258)},
            "bitcoin": {5: None, 10: None, 20: None},
        },
        "MAR": {
            "bank": {5: (100.0, 0.303), 10: (100.0, 0.523), 20: (100.0, 0.805)},
            "nyc": {5: (100.0, 0.837), 10: (100.0, 0.900), 20: (100.0, 1.016)},
            "bitcoin": {5: None, 10: None, 20: None},
        },
    },
    "Distribution-Centric": {
        "MCAR": {
            "bank": {5: (90.24, None), 10: (88.30, None), 20: (85.30, None)},
            "nyc": {5: (99.80, None), 10: (99.70, None), 20: (99.83, None)},
            "bitcoin": {5: (88.90, None), 10: (87.60, None), 20: (84.90, None)},
        },
        "MAR": {
            "bank": {5: (92.13, None), 10: (92.16, None), 20: (92.18, None)},
            "nyc": {5: (99.80, None), 10: (99.84, None), 20: (99.80, None)},
            "bitcoin": {5: (90.90, None), 10: (90.80, None), 20: (90.80, None)},
        },
    },
}
def read_method(path: Path, method: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame = frame[frame["method"].eq(method)].copy()
    frame["rate"] = pd.to_numeric(frame["rate"], errors="raise").astype(int)
    frame["query_index"] = pd.to_numeric(frame["query_index"], errors="raise").astype(int)
    frame["time_s"] = pd.to_numeric(frame["time_s"], errors="raise")
    if "row_limit" in frame and not pd.to_numeric(frame["row_limit"], errors="raise").eq(0).all():
        raise RuntimeError(f"{path} contains a row-limited measurement")
    return frame


def read_qe() -> pd.DataFrame:
    frames = []
    expected = {"MCAR": {"set": 4, "aggregate": 5}, "MAR": {"set": 4, "aggregate": 5}, "MNAR": {"set": 5, "aggregate": 5}}
    for mechanism in ("MCAR", "MAR"):
        for dataset in DATASETS:
            frame = read_method(QE_SELECTED / mechanism / f"{dataset}.csv", "QE")
            frame["mechanism"] = mechanism
            frames.append(frame)
    for dataset in DATASETS:
        frame = read_method(QE_MNAR / f"{dataset}.csv", "QE")
        selected = []
        for workload, indices in SELECTED_MNAR.items():
            selected.append(frame[frame["workload"].eq(workload) & frame["query_index"].isin(indices)].copy())
        frame = pd.concat(selected, ignore_index=True)
        frame["mechanism"] = "MNAR"
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True, sort=False)
    counts = result.groupby(["mechanism", "dataset", "rate", "workload"])["query_index"].nunique()
    if len(counts) != 54:
        raise RuntimeError("QE mechanism groups are incomplete")
    for key, count in counts.items():
        mechanism, _dataset, _rate, workload = key
        if count != expected[mechanism][workload]:
            raise RuntimeError(f"QE group {key} contains {count} queries")
    if not pd.to_numeric(result["h"], errors="raise").eq(783).all():
        raise RuntimeError("QE does not consistently use H=783")
    return result


def read_caex() -> pd.DataFrame:
    frames = []
    expected = {"MCAR": {"set": 4, "aggregate": 5}, "MAR": {"set": 4, "aggregate": 5}, "MNAR": {"set": 5, "aggregate": 5}}
    for mechanism in ("MCAR", "MAR"):
        for dataset in DATASETS:
            frame = read_method(CAEX_SELECTED / mechanism / f"{dataset}.csv", "CAEX")
            frame["mechanism"] = mechanism
            frames.append(frame)
    mnar = read_method(CAEX_MNAR, "CAEX")
    mnar["plot_query_index"] = pd.to_numeric(mnar["plot_query_index"], errors="raise").astype(int)
    selected = []
    for workload, indices in SELECTED_MNAR.items():
        selected.append(mnar[mnar["workload"].eq(workload) & mnar["plot_query_index"].isin(indices)].copy())
    mnar = pd.concat(selected, ignore_index=True)
    mnar["mechanism"] = "MNAR"
    frames.append(mnar)
    result = pd.concat(frames, ignore_index=True, sort=False)
    result["selected_query_index"] = pd.to_numeric(result.get("plot_query_index"), errors="coerce")
    result.loc[result["selected_query_index"].isna(), "selected_query_index"] = result.loc[result["selected_query_index"].isna(), "query_index"]
    result["selected_query_index"] = result["selected_query_index"].astype(int)
    counts = result.groupby(["mechanism", "dataset", "rate", "workload"])["selected_query_index"].nunique()
    if len(counts) != 54:
        raise RuntimeError("CAEX mechanism groups are incomplete")
    for key, count in counts.items():
        mechanism, _dataset, _rate, workload = key
        if count != expected[mechanism][workload]:
            raise RuntimeError(f"CAEX group {key} contains {count} queries")
    return result


def weighted_summary(frame: pd.DataFrame, groups: list[str], value: str, count: str) -> pd.DataFrame:
    data = frame.copy()
    data["_weighted"] = data[value] * data[count]
    result = data.groupby(groups, as_index=False).agg(_weighted=("_weighted", "sum"), query_count=(count, "sum"))
    result[value] = result["_weighted"] / result["query_count"]
    return result.drop(columns="_weighted")


def cade_runtime() -> pd.DataFrame:
    frame = pd.read_csv(CADE_SOURCE / "figure3_cade_mechanism_runtime.csv")
    frame["dataset"] = frame["dataset"].map(LABEL_TO_DATASET)
    frame["workload"] = frame["query_type"].map(LABEL_TO_WORKLOAD)
    result = weighted_summary(frame, ["dataset", "missingness_rate", "workload"], "mean_time_s", "query_count")
    result = result.rename(columns={"missingness_rate": "rate"})
    result["method"] = "CADE"
    result["completed_query_count"] = result["query_count"]
    return result


def qe_runtime(qe: pd.DataFrame) -> pd.DataFrame:
    result = qe.groupby(["dataset", "rate", "workload"], as_index=False).agg(
        mean_time_s=("time_s", "mean"),
        query_count=("query_index", "size"),
        completed_query_count=("status", lambda values: int(values.eq("ok").sum())),
    )
    expected = result["workload"].map({"set": 13, "aggregate": 15})
    if not result["query_count"].eq(expected).all():
        raise RuntimeError("QE runtime means have incorrect query counts")
    result["method"] = "QE"
    return result


def estimator_runtime(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    result = frame.groupby(["dataset", "rate", "workload"], as_index=False).agg(
        mean_time_s=("time_s", "mean"),
        query_count=("query_index", "size"),
        completed_query_count=("status", lambda values: int(values.eq("ok").sum())),
    )
    expected = result["workload"].map({"set": 13, "aggregate": 15})
    if not result["query_count"].eq(expected).all():
        raise RuntimeError(f"{method} runtime means have incorrect query counts")
    result["method"] = method
    return result


def certain_runtime() -> pd.DataFrame:
    frame = pd.read_csv(CURRENT_FIGURE3)
    frame = frame[frame["method"].eq("Certain Answers")].copy()
    frame["_weighted"] = frame["mean_time_s"] * frame["query_count"]
    rows = []
    for (dataset, rate), group in frame.groupby(["dataset", "rate"]):
        count = int(group["query_count"].sum())
        if count != 10:
            raise RuntimeError(f"Certain for {dataset} at {rate}% does not contain ten queries")
        rows.append({"dataset": dataset, "rate": int(rate), "workload": "set", "method": "Certain", "mean_time_s": group["_weighted"].sum() / count, "query_count": count, "completed_query_count": count})
    return pd.DataFrame(rows)


def baseline_runtime() -> pd.DataFrame:
    rows = []
    for method, mechanisms in BASELINE_RUNTIME.items():
        for dataset in DATASETS:
            for rate in RATES:
                values = [mechanisms[mechanism][dataset][rate] for mechanism in ("MCAR", "MAR")]
                values.append(300.0)
                completed = sum(value < 300.0 for value in values) * 5
                rows.append({"dataset": dataset, "rate": rate, "workload": "aggregate", "method": method, "mean_time_s": float(np.mean(values)), "query_count": 15, "completed_query_count": completed})
    return pd.DataFrame(rows)


def build_runtime(caex: pd.DataFrame, qe: pd.DataFrame) -> pd.DataFrame:
    result = pd.concat([estimator_runtime(caex, "CAEX"), cade_runtime(), qe_runtime(qe), certain_runtime(), baseline_runtime()], ignore_index=True, sort=False)
    result["dataset_label"] = result["dataset"].map(DATASET_LABELS)
    result["query_type"] = result["workload"].map(WORKLOAD_LABELS)
    result.to_csv(OUT / "figure3_runtime.csv", index=False)
    return result


def cade_quality() -> pd.DataFrame:
    frame = pd.read_csv(CADE_SOURCE / "table3_cade_mechanism_quality.csv")
    frame["dataset"] = frame["dataset"].map(LABEL_TO_DATASET)
    result = weighted_summary(frame, ["dataset", "missingness_rate", "metric"], "value", "query_count")
    result = result.rename(columns={"missingness_rate": "rate"})
    result["method"] = "CADE"
    result["completed_query_count"] = result["query_count"]
    return result


def qe_quality(qe: pd.DataFrame) -> pd.DataFrame:
    completed = qe[qe["status"].eq("ok")].copy()
    completed["metric"] = pd.to_numeric(completed["metric"], errors="coerce")
    completed["delta_w"] = pd.to_numeric(completed["delta_w"], errors="coerce")
    rows = []
    for dataset in DATASETS:
        for rate in RATES:
            for workload in ("set", "aggregate"):
                part = completed[completed["dataset"].eq(dataset) & completed["rate"].eq(rate) & completed["workload"].eq(workload)]
                total = 13 if workload == "set" else 15
                if workload == "set":
                    values = (("Non-aggregation (TVD)", part["metric"].mean()), ("Non-aggregation ($\\Delta w$)", part["delta_w"].mean()))
                else:
                    values = (("Aggregation (Accuracy \\%)", 100.0 * (1.0 - part["metric"].mean())), ("Aggregation ($\\Delta w$)", part["delta_w"].mean()))
                for metric, value in values:
                    rows.append({"dataset": dataset, "rate": rate, "method": "QE", "metric": metric, "value": value, "completed_query_count": len(part), "query_count": total})
    return pd.DataFrame(rows)


def estimator_quality(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    completed = frame[frame["status"].eq("ok")].copy()
    completed["metric"] = pd.to_numeric(completed["metric"], errors="coerce")
    completed["delta_w"] = pd.to_numeric(completed["delta_w"], errors="coerce")
    rows = []
    for dataset in DATASETS:
        for rate in RATES:
            for workload in ("set", "aggregate"):
                part = completed[completed["dataset"].eq(dataset) & completed["rate"].eq(rate) & completed["workload"].eq(workload)]
                total = 13 if workload == "set" else 15
                if workload == "set":
                    values = (("Non-aggregation (TVD)", part["metric"].mean()), ("Non-aggregation ($\\Delta w$)", part["delta_w"].mean()))
                else:
                    values = (("Aggregation (Accuracy \\%)", 100.0 * (1.0 - part["metric"].mean())), ("Aggregation ($\\Delta w$)", part["delta_w"].mean()))
                for metric, value in values:
                    rows.append({"dataset": dataset, "rate": rate, "method": method, "metric": metric, "value": value, "completed_query_count": len(part), "query_count": total})
    return pd.DataFrame(rows)


def certain_quality() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        frame = pd.read_csv(CERTAIN_SOURCE / f"{dataset}.csv")
        frame = frame[frame["method"].eq("Certain Answers")].copy()
        frame["rate"] = pd.to_numeric(frame["rate"], errors="raise").astype(int)
        frame["metric"] = pd.to_numeric(frame["metric"], errors="raise")
        frame["uniform_set_tvd"] = pd.to_numeric(frame["uniform_set_tvd"], errors="raise")
        frame["precision"] = pd.to_numeric(frame["precision"], errors="raise")
        if not frame["status"].eq("ok").all():
            raise RuntimeError(f"Certain contains an incomplete query for {dataset}")
        if not np.allclose(frame["metric"], frame["uniform_set_tvd"], rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Certain does not use uniform-set TVD for {dataset}")
        if not np.allclose(frame["precision"], 1.0, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Certain is not a subset of the ground-truth answers for {dataset}")
        for rate in RATES:
            part = frame[frame["rate"].eq(rate)]
            if len(part) != 10:
                raise RuntimeError(f"Certain for {dataset} at {rate}% does not contain ten queries")
            rows.append({"dataset": dataset, "rate": rate, "method": "Certain", "metric": "Non-aggregation (TVD)", "value": part["metric"].mean(), "completed_query_count": len(part), "query_count": len(part)})
    return pd.DataFrame(rows)


def baseline_quality() -> pd.DataFrame:
    rows = []
    for method, mechanisms in BASELINE_QUALITY.items():
        for dataset in DATASETS:
            for rate in RATES:
                completed = [mechanisms[mechanism][dataset][rate] for mechanism in ("MCAR", "MAR")]
                completed = [value for value in completed if value is not None]
                percentage = float(np.mean([value[0] for value in completed])) if completed else np.nan
                width_values = [value[1] for value in completed if value[1] is not None]
                width = float(np.mean(width_values)) if width_values else np.nan
                percentage_metric = "Aggregation (Coverage \\%)" if method == "Interval-Based" else "Aggregation (Accuracy \\%)"
                rows.append({"dataset": dataset, "rate": rate, "method": method, "metric": percentage_metric, "value": percentage, "completed_query_count": 5 * len(completed), "query_count": 15})
                rows.append({"dataset": dataset, "rate": rate, "method": method, "metric": "Aggregation ($\\Delta w$)", "value": width, "completed_query_count": 5 * len(width_values), "query_count": 15})
    return pd.DataFrame(rows)


def build_quality(caex: pd.DataFrame, qe: pd.DataFrame) -> pd.DataFrame:
    result = pd.concat([estimator_quality(caex, "CAEX"), cade_quality(), qe_quality(qe), certain_quality(), baseline_quality()], ignore_index=True, sort=False)
    result["dataset_label"] = result["dataset"].map(DATASET_LABELS)
    result.to_csv(OUT / "table3_quality.csv", index=False)
    return result


def plot_runtime(runtime: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 8.5})
    positions = {"Aggregation": np.array([0.0, 1.0, 2.0]), "Non-aggregation": np.array([4.0, 5.0, 6.0])}
    colors = {"CAEX": "#D62728", "CADE": "#009E73", "QE": "#E69F00", "Certain": "#7B61A8", "Interval-Based": "#0057B8", "Distribution-Centric": "#B2185B"}
    hatches = {"CAEX": "xx", "CADE": "..", "QE": "//", "Certain": "\\\\", "Interval-Based": "", "Distribution-Centric": ""}
    width = 0.15
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=False)
    for axis_index, (axis, dataset) in enumerate(zip(axes, DATASETS)):
        for query_type, centers in positions.items():
            methods = AGGREGATION_METHODS if query_type == "Aggregation" else NONAGGREGATION_METHODS
            offsets = {method: (index - (len(methods) - 1) / 2.0) * width for index, method in enumerate(methods)}
            for method in methods:
                values = []
                for rate in RATES:
                    row = runtime[runtime["dataset"].eq(dataset) & runtime["rate"].eq(rate) & runtime["query_type"].eq(query_type) & runtime["method"].eq(method)]
                    if len(row) != 1:
                        raise RuntimeError(f"Missing runtime for {dataset}, {rate}, {query_type}, {method}")
                    values.append(float(row.iloc[0]["mean_time_s"]))
                axis.bar(centers + offsets[method], values, width, color=colors[method], hatch=hatches[method], edgecolor="white", linewidth=0.5)
        axis.axvline(3.0, color="#777777", linestyle=(0, (2, 2)), linewidth=0.9)
        axis.set_yscale("log")
        axis.set_ylim(1e-3, 5e2)
        axis.set_yticks([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
        axis.set_title(DATASET_LABELS[dataset], fontweight="bold")
        axis.set_xticks(np.concatenate([positions["Aggregation"], positions["Non-aggregation"]]))
        axis.set_xticklabels(["5%", "10%", "20%", "5%", "10%", "20%"])
        axis.text(1.0, -0.30, "Aggregation", ha="center", transform=axis.get_xaxis_transform())
        axis.text(5.0, -0.30, "Non-aggregation", ha="center", transform=axis.get_xaxis_transform())
        axis.grid(axis="y", alpha=0.35, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Mean execution time (s)")
    figure.supxlabel("Missingness rate", y=0.005)
    handles = [Patch(facecolor=colors[method], hatch=hatches[method], edgecolor="white", label=method) for method in METHODS]
    figure.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.02), columnspacing=0.9, handlelength=1.6)
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.36, top=0.76, wspace=0.27)
    figure.savefig(OUT / "figure3_all_methods.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def formatted(metric: str, value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.4f}"


def write_table(quality: pd.DataFrame) -> None:
    metrics = ("Non-aggregation (TVD)", "Non-aggregation ($\\Delta w$)", "Aggregation (Accuracy \\%)", "Aggregation ($\\Delta w$)")
    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{Mean query quality on injected MCAR, MAR, and factorizable MNAR data. The baseline values are computed from completed MCAR and MAR aggregation queries.}", r"\label{tab:injected_query_quality}", r"\resizebox{\linewidth}{!}{%", r"\setlength{\tabcolsep}{1.5pt}", r"\begin{tabular}{llcccccccccccccccccc}", r"\toprule", r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Query type and metric} & \multicolumn{3}{c}{CAEX} & \multicolumn{3}{c}{CADE} & \multicolumn{3}{c}{QE} & \multicolumn{3}{c}{Certain} & \multicolumn{3}{c}{Interval-Based} & \multicolumn{3}{c}{Distribution-Centric} \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}\cmidrule(lr){15-17}\cmidrule(lr){18-20}", r" & & 5\% & 10\% & 20\% & 5\% & 10\% & 20\% & 5\% & 10\% & 20\% & 5\% & 10\% & 20\% & 5\% & 10\% & 20\% & 5\% & 10\% & 20\% \\", r"\midrule"]
    for dataset_index, dataset in enumerate(DATASETS):
        for metric_index, metric in enumerate(metrics):
            values = []
            for method in METHODS:
                for rate in RATES:
                    source_metric = "Aggregation (Coverage \\%)" if method == "Interval-Based" and metric == "Aggregation (Accuracy \\%)" else metric
                    row = quality[quality["dataset"].eq(dataset) & quality["rate"].eq(rate) & quality["method"].eq(method) & quality["metric"].eq(source_metric)]
                    if method == "Certain" and metric != "Non-aggregation (TVD)":
                        values.append("N/A")
                    elif method in ("Interval-Based", "Distribution-Centric") and metric.startswith("Non-aggregation"):
                        values.append("N/A")
                    elif method == "Distribution-Centric" and metric == "Aggregation ($\\Delta w$)":
                        values.append("N/A")
                    elif row.empty or int(row.iloc[0]["completed_query_count"]) == 0:
                        values.append("--")
                    elif method == "Interval-Based" and metric == "Aggregation (Accuracy \\%)":
                        values.append(r"100\%")
                    else:
                        values.append(formatted(source_metric, float(row.iloc[0]["value"])))
            prefix = rf"\multirow{{4}}{{*}}{{{DATASET_LABELS[dataset]}}}" if metric_index == 0 else ""
            lines.append(prefix + " & " + metric + " & " + " & ".join(values) + r" \\")
        if dataset_index != len(DATASETS) - 1:
            lines.append(r"\midrule")
    lines.extend((r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"))
    (OUT / "table3_quality.tex").write_text("\n".join(lines) + "\n")


def write_audit(runtime: pd.DataFrame, quality: pd.DataFrame, caex: pd.DataFrame, qe: pd.DataFrame) -> None:
    fallback = pd.to_numeric(qe["factor_fallback_symbols"], errors="coerce").dropna()
    lines = ["Figure 3 comparison", "", "CAEX and QE use the same four MCAR and four MAR non-aggregation queries represented by the appendix results.", "CAEX and QE use the five MCAR and five MAR aggregation queries in all_queries.json.", "CADE uses the existing appendix measurements.", "The MNAR entries use the same five selected queries per query type.", "Therefore, each non-aggregation estimator mean contains 13 queries and each aggregation estimator mean contains 15 queries.", "Certain uses the same ten MNAR non-aggregation measurements as the current Figure 3.", "The Interval-Based and Distribution-Centric values use the MCAR and MAR results in Appendix Tables tab:injected_mcar_agg and tab:injected_mar_agg.", "Their runtime means include the five-minute MNAR timeout.", "Their quality means cover completed MCAR and MAR aggregation queries only.", "Runtime means include timed-out queries. Quality means include completed queries only.", "", "Runtime counts:", runtime.groupby(["method", "workload"])[["query_count", "completed_query_count"]].agg(lambda values: sorted(set(values))).to_string(), "", "Quality completion counts:", quality.groupby(["method", "metric"])["completed_query_count"].agg(lambda values: sorted(set(values))).to_string(), "", "CAEX mechanisms: " + str(sorted(set(caex["mechanism"]))), "QE factor-distribution fallback counts: " + str(sorted(set(fallback)))]
    (OUT / "SOURCE_AUDIT.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    caex = read_caex()
    qe = read_qe()
    runtime = build_runtime(caex, qe)
    quality = build_quality(caex, qe)
    plot_runtime(runtime)
    write_table(quality)
    write_audit(runtime, quality, caex, qe)


if __name__ == "__main__":
    main()
