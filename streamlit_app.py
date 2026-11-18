"""Cloud-ready ThermoGuard AI entry point.

Uses the repository's existing validated data, injection, and detector modules.
Ground-truth labels are only used after prediction, for evaluation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from thermoguard.data.injection import inject_anomaly
from thermoguard.data.splitting import chronological_split
from thermoguard.data.validation import validate_dataset
from thermoguard.detectors.control_limits import StatisticalControlLimits
from thermoguard.detectors.isolation_forest import TemperatureIsolationForest
from thermoguard.evaluation.metrics import calculate_metrics

EXPERIMENTS = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text(encoding="utf-8"))
DETECTORS = yaml.safe_load((ROOT / "configs/detectors.yaml").read_text(encoding="utf-8"))["detectors"]

st.set_page_config(page_title="ThermoGuard AI", page_icon="🌡️", layout="wide")
st.title("🌡️ ThermoGuard AI")
st.caption("Temperature anomaly detection · Statistical baseline vs. unsupervised machine learning")
st.warning(
    "Evaluation uses synthetic injected anomalies. These are controlled test cases, "
    "not confirmed real-world equipment faults."
)

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload ETTh1.csv", type=["csv"])
    st.caption("The CSV is processed in memory for this session; it is not committed to GitHub.")
    st.divider()
    st.header("Evaluation notes")
    st.write("• Chronological 70/15/15 split")
    st.write("• Models fit on clean training OT only")
    st.write("• Labels used only after prediction")
    st.write("• Validation and test scenarios remain separate")

if uploaded is None:
    local_csv = ROOT / "data/raw/ETTh1.csv"
    if local_csv.exists():
        raw = pd.read_csv(local_csv)
        st.info("Using data/raw/ETTh1.csv from the repository workspace.")
    else:
        st.info("Upload the ETTh1.csv dataset in the sidebar to start the analysis.")
        st.stop()
else:
    try:
        raw = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()

issues = validate_dataset(raw)
if issues:
    st.error("Dataset validation failed. Correct the source file; the app will not silently repair it.")
    st.dataframe(pd.DataFrame({"Issue": issues}), hide_index=True, use_container_width=True)
    st.stop()

try:
    split_cfg = EXPERIMENTS["split"]
    split = chronological_split(
        raw,
        train_fraction=split_cfg["train_fraction"],
        validation_fraction=split_cfg["validation_fraction"],
        test_fraction=split_cfg["test_fraction"],
    )
except Exception as exc:
    st.error(f"Could not create chronological splits: {exc}")
    st.stop()

if len(split.train) < 2:
    st.error("Not enough clean training observations.")
    st.stop()

# Fit both detectors only on the clean chronological training segment.
try:
    control_cfg = DETECTORS["statistical_control_limits"]
    control = StatisticalControlLimits(
        k=control_cfg["k"], feature_column="OT"
    ).fit(split.train["OT"])

    forest_cfg = DETECTORS["isolation_forest"]
    forest = TemperatureIsolationForest(
        random_state=forest_cfg.get("random_state", 20261023),
        contamination=forest_cfg.get("contamination", 0.01),
        n_estimators=forest_cfg.get("n_estimators", 200),
        max_samples=forest_cfg.get("max_samples", "auto"),
        max_features=forest_cfg.get("max_features", 1.0),
        bootstrap=forest_cfg.get("bootstrap", False),
        n_jobs=forest_cfg.get("n_jobs", 1),
        fit_feature="OT",
    ).fit(split.train["OT"])
except Exception as exc:
    st.error(f"Could not fit the detectors: {exc}")
    st.stop()

# Generate scenarios and score each detector. Labels never enter fit() or predict().
scenario_frames: dict[str, pd.DataFrame] = {}
rows: list[dict] = []
scenario_errors: list[str] = []

for cfg in EXPERIMENTS["scenarios"]:
    try:
        segment_name = cfg["segment"]
        clean_segment = getattr(split, segment_name)
        kwargs = {
            "anomaly_type": cfg["anomaly_type"],
            "segment": segment_name,
            "start_position": cfg["start_position"],
            "duration": cfg["duration"],
            "seed": cfg.get("seed", EXPERIMENTS["seed"]),
            "affected_feature": cfg.get("affected_feature", "OT"),
        }
        if "magnitude" in cfg:
            kwargs["magnitude"] = cfg["magnitude"]
        if "variance_multiplier" in cfg:
            kwargs["variance_multiplier"] = cfg["variance_multiplier"]
            kwargs["reference_values"] = split.train["OT"]

        injected = inject_anomaly(clean_segment, **kwargs).data
        injected = injected.copy()
        injected["scenario_name"] = cfg["name"]
        injected["segment"] = segment_name
        scenario_frames[cfg["name"]] = injected

        truth = injected["anomaly_label"].copy()
        sensor_values = injected["OT_injected"].copy()
        detector_predictions = {
            "Statistical control limits": control.predict(sensor_values),
            "Isolation Forest": forest.predict(sensor_values),
        }

        for detector_label, predicted in detector_predictions.items():
            if not predicted.index.equals(truth.index):
                raise ValueError(f"Prediction alignment failed for {cfg['name']}.")
            metric = calculate_metrics(truth, predicted)
            metric["balanced_accuracy"] = float(
                balanced_accuracy_score(truth, predicted)
            )
            metric["mcc"] = float(matthews_corrcoef(truth, predicted))
            metric.update({
                "scenario": cfg["name"],
                "pattern": cfg["anomaly_type"],
                "split": segment_name,
                "detector": detector_label,
            })
            rows.append(metric)
    except Exception as exc:
        scenario_errors.append(f"{cfg.get('name', 'unnamed scenario')}: {exc}")

if scenario_errors:
    st.error("One or more configured scenarios could not be evaluated.")
    st.write("\n".join(scenario_errors))
    st.stop()

results = pd.DataFrame(rows)
if results.empty:
    st.error("No evaluation results were generated.")
    st.stop()

# Overview
dates = pd.to_datetime(raw["date"])
a, b, c, d = st.columns(4)
a.metric("Observations", f"{len(raw):,}")
b.metric("Clean training rows", f"{len(split.train):,}")
c.metric("Sampling interval", "1 hour")
d.metric("Sensor signals", "7")
st.caption(f"Dataset range: {dates.min():%Y-%m-%d %H:%M} to {dates.max():%Y-%m-%d %H:%M}")

st.subheader("Scenario and detector")
scenario_name = st.selectbox("Choose a synthetic scenario", list(scenario_frames))
scenario_cfg = next(x for x in EXPERIMENTS["scenarios"] if x["name"] == scenario_name)
detector_label = st.radio(
    "Choose detector",
    ["Statistical control limits", "Isolation Forest"],
    horizontal=True,
)
selected = scenario_frames[scenario_name]
selected_detector = control if detector_label == "Statistical control limits" else forest
predicted = selected_detector.predict(selected["OT_injected"])
scores = (
    selected_detector.score(selected["OT_injected"])
    if detector_label == "Statistical control limits"
    else selected_detector.score(selected["OT_injected"])
)

chart = go.Figure()
chart.add_trace(go.Scatter(
    x=selected["date"], y=selected["OT"], mode="lines", name="Original OT"
))
chart.add_trace(go.Scatter(
    x=selected["date"], y=selected["OT_injected"], mode="lines",
    name="Injected scenario OT", line={"dash": "dash"}
))
flagged = selected.loc[predicted == 1]
chart.add_trace(go.Scatter(
    x=flagged["date"], y=flagged["OT_injected"], mode="markers",
    name="Predicted anomalies", marker={"size": 9, "symbol": "x", "color": "#ef4444"}
))
chart.update_layout(
    height=440, xaxis_title="Timestamp", yaxis_title="Oil temperature (OT)",
    hovermode="x unified", legend={"orientation": "h"}
)
st.plotly_chart(chart, use_container_width=True)

pred_count = int(predicted.sum())
x1, x2, x3 = st.columns(3)
x1.metric("Rows analyzed", f"{len(selected):,}")
x2.metric("Predicted anomalies", f"{pred_count:,}")
x3.metric("Predicted anomaly rate", f"{100 * pred_count / len(selected):.2f}%")

metric_row = results.loc[
    (results["scenario"] == scenario_name) & (results["detector"] == detector_label)
].iloc[0]

st.subheader("Evaluation metrics")
st.caption(
    "These scores compare predictions with synthetic labels after prediction. "
    "They are point-level metrics, not proof of real-world fault detection."
)
metric_cols = st.columns(4)
for col, key, label in zip(
    metric_cols,
    ["precision", "recall", "f1", "accuracy"],
    ["Precision", "Recall", "F1 score", "Accuracy"],
    strict=True,
):
    col.metric(label, f"{metric_row[key]:.3f}")
metric_cols2 = st.columns(4)
for col, key, label in zip(
    metric_cols2,
    ["balanced_accuracy", "specificity", "false_positive_rate", "mcc"],
    ["Balanced accuracy", "Specificity", "False positive rate", "Matthews corr. (MCC)"],
    strict=True,
):
    col.metric(label, f"{metric_row[key]:.3f}")

st.write({
    "True positives": int(metric_row["true_positives"]),
    "False positives": int(metric_row["false_positives"]),
    "True negatives": int(metric_row["true_negatives"]),
    "False negatives": int(metric_row["false_negatives"]),
    "Actual synthetic anomaly points": int(metric_row["n_true_anomaly_points"]),
    "Predicted anomaly points": int(metric_row["n_predicted_anomaly_points"]),
})

st.subheader("Compare all scenarios")
display_cols = [
    "scenario", "pattern", "split", "detector", "precision", "recall", "f1",
    "accuracy", "balanced_accuracy", "specificity", "false_positive_rate", "mcc",
    "true_positives", "false_positives", "true_negatives", "false_negatives",
]
table = results[display_cols].copy()
for col in [
    "precision", "recall", "f1", "accuracy", "balanced_accuracy",
    "specificity", "false_positive_rate", "mcc",
]:
    table[col] = table[col].map(lambda v: round(float(v), 4))
st.dataframe(table, hide_index=True, use_container_width=True)
st.download_button(
    "Download evaluation results (CSV)",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="thermoguard_evaluation_results.csv",
    mime="text/csv",
)

with st.expander("How to interpret the high accuracy"):
    st.markdown(
        """
        Accuracy can look very high when normal observations greatly outnumber
        anomalies. For example, a detector that predicts every point as normal
        can achieve high accuracy while having **zero recall** on a rare anomaly.

        - **Precision:** among flagged points, how many were injected anomaly points?
        - **Recall:** among injected anomaly points, how many were flagged?
        - **F1:** harmonic mean of precision and recall.
        - **Balanced accuracy:** average of sensitivity (recall) and specificity.
        - **MCC:** correlation between binary predictions and labels; 1 is perfect,
          0 is no better than chance, and -1 is total disagreement.
        """
    )

with st.expander("Model design and limitations"):
    st.markdown(
        """
        **Statistical control limits** use the clean training mean ± k sample
        standard deviations. **Isolation Forest** is an unsupervised ML detector
        fit on the same clean training OT values. They are independent baselines,
        not an ensemble.

        Both are univariate and do not model seasonality, time order, load,
        operating regimes, or corroborating sensors. Synthetic injections are
        controlled test cases, not naturally occurring confirmed faults.
        """
    )

st.success("Dataset validation passed. Source measurements were not modified.")
