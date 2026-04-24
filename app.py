from __future__ import annotations

import base64
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
MANUAL_FEATURE_COLUMNS = [
    "Atomosphere of build",
    "Same Layer Scanned?",
    "Build Direction [degrees]",
    "Power [W]",
    "Velocity [m/s]",
    "Hatch Spacing [μm]",
    "Beam Diameter [μm]",
    "d90 [um]",
    "Nucleants",
    "Powder Shape [0-1]",
    "Powder Layer Thickness [μm]",
    "Atomization Atomosphere",
    "Powder Recycled How Many Times?",
    "Initial Powder Bed Temperature [K]",
    "Substrate/Platform Temperature [K]",
    "Surface Energy Density [J/mm2]",
    "Linear Energy Density [J/mm]",
    "Volume Energy Density [J/mm3]",
]
FALLBACK_NUMERIC_FEATURES = [
    "Power [W]",
    "Velocity [m/s]",
    "Hatch Spacing [μm]",
    "Beam Diameter [μm]",
    "Powder Layer Thickness [μm]",
    "Initial Powder Bed Temperature [K]",
    "Substrate/Platform Temperature [K]",
    "Surface Energy Density [J/mm2]",
    "Linear Energy Density [J/mm]",
    "Volume Energy Density [J/mm3]",
]
PATHS = {
    "model": ROOT_DIR / "models" / "best_model_smote.pkl",
    "video": ROOT_DIR / "static" / "Video.mp4",
    "confusion_matrix": ROOT_DIR / "outputs" / "figures" / "confusion_matrix.png",
    "roc_curve": ROOT_DIR / "outputs" / "figures" / "roc_curve.png",
    "processed_data": ROOT_DIR / "data" / "processed" / "DataClean_FINAL.csv",
    "best_combinations": ROOT_DIR / "data" / "processed" / "best_success_combinations.csv",
    "metrics": ROOT_DIR / "outputs" / "metrics" / "model_metrics.json",
    "logo": ROOT_DIR / "assets" / "logo.png",
}


def inject_theme() -> None:
    st.markdown(
        """
<style>
body, .stApp {
    background-color: #0E0E0E;
    color: #E0E0E0;
}
h1, h2, h3, h4 {
    color: #FFFFFF;
}
label {
    color: #CCCCCC !important;
}
.success-badge {
    background-color: #0f5132;
    color: #d1e7dd;
    padding: 6px 12px;
    border-radius: 6px;
    font-weight: bold;
}
.fail-badge {
    background-color: #842029;
    color: #f8d7da;
    padding: 6px 12px;
    border-radius: 6px;
    font-weight: bold;
}
.video-box {
    display: flex;
    justify-content: center;
    margin-top: 15px;
    margin-bottom: 10px;
}
</style>
""",
        unsafe_allow_html=True,
    )


def explain(text: str) -> None:
    st.markdown(f"<p style='color:#888; font-size:13px;'>{text}</p>", unsafe_allow_html=True)


def load_video_base64(path: Path) -> str | None:
    if not path.exists():
        st.warning(f"Optional asset missing: `{path.as_posix()}`")
        return None
    return base64.b64encode(path.read_bytes()).decode()


@st.cache_resource(show_spinner=False)
def load_pipeline(path: Path):
    if not path.exists():
        st.error(f"Required model file not found: `{path.as_posix()}`")
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        st.error(f"Failed to load model pipeline: {exc}")
        return None


def load_optional_dataframe(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        st.warning(f"Optional dataset missing ({label}): `{path.as_posix()}`")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {label}: {exc}")
        return None


def clean_feature_label(raw_name: str) -> str:
    name = str(raw_name).strip()
    if "__" in name:
        name = name.split("__", 1)[1]

    lower_name = name.lower()
    if "_" in name and ("atomosphere" in lower_name or "nucleants" in lower_name or "scanned" in lower_name):
        prefix, suffix = name.rsplit("_", 1)
        pretty_prefix = (
            prefix.replace("Atomosphere of build", "Atmosphere")
            .replace("Atomization Atomosphere", "Atomization Atmosphere")
            .replace("Same Layer Scanned?", "Same Layer Scanned")
            .replace("Nucleants", "Nucleants")
        )
        if suffix:
            return f"{pretty_prefix}: {suffix}"

    name = name.replace("Power [W]", "Laser Power [W]")
    name = name.replace("Velocity [m/s]", "Scan Speed [m/s]")
    name = name.replace("Hatch Spacing [μm]", "Hatch Spacing [μm]")
    name = name.replace("Powder Layer Thickness [μm]", "Layer Thickness [μm]")
    name = name.replace("Beam Diameter [μm]", "Spot Size [μm]")
    name = name.replace("Atomosphere of build", "Atmosphere")
    name = name.replace("Atomization Atomosphere", "Atomization Atmosphere")
    name = name.replace("Initial Powder Bed Temperature [K]", "Preheat Temperature [K]")
    name = name.replace("Build Direction [degrees]", "Build Orientation [degrees]")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def get_final_estimator(pipeline):
    if hasattr(pipeline, "named_steps"):
        return list(pipeline.named_steps.values())[-1]
    return pipeline


def load_processed_data_for_importance() -> pd.DataFrame | None:
    primary = load_optional_dataframe(PATHS["processed_data"], "processed dataset")
    if primary is not None:
        return primary
    fallback_path = ROOT_DIR / "DataClean_FINAL.csv"
    return load_optional_dataframe(fallback_path, "processed dataset (root fallback)")


def build_feature_importance_table(pipeline) -> tuple[pd.DataFrame, str]:
    model = get_final_estimator(pipeline)
    importances = np.asarray(getattr(model, "feature_importances_", []), dtype=float).reshape(-1)
    if importances.size == len(MANUAL_FEATURE_COLUMNS):
        df = pd.DataFrame(
            {
                "feature": [clean_feature_label(x) for x in MANUAL_FEATURE_COLUMNS],
                "importance": importances,
            }
        )
        return df.sort_values("importance", ascending=False), "model"

    data = load_processed_data_for_importance()
    if data is None or "Printability Evaluation" not in data.columns:
        raise ValueError("Processed dataset unavailable for exploratory fallback chart.")

    y = data["Printability Evaluation"].astype(str).str.strip().str.lower()
    y_numeric = y.map({"success": 1, "fail": 0, "1": 1, "0": 0})
    valid = y_numeric.notna()
    if valid.sum() == 0:
        raise ValueError("Could not map target labels for exploratory fallback chart.")

    rows = []
    for col in FALLBACK_NUMERIC_FEATURES:
        if col not in data.columns:
            continue
        series = pd.to_numeric(data[col], errors="coerce")
        pair = pd.concat([series, y_numeric], axis=1).dropna()
        if pair.empty:
            continue
        corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
        if pd.notna(corr):
            rows.append({"feature": clean_feature_label(col), "importance": abs(float(corr))})

    if not rows:
        raise ValueError("No numeric fallback features available for exploratory chart.")
    df = pd.DataFrame(rows).sort_values("importance", ascending=False)
    return df, "exploratory"


def render_importance_bar(feature_names: list[str], importances: np.ndarray, top_n: int = 12) -> None:
    values = np.asarray(importances).reshape(-1)
    names = np.asarray(feature_names, dtype=str)
    count = min(top_n, len(values), len(names))
    if count == 0:
        st.warning("Feature importance values are unavailable.")
        return

    idx = np.argsort(values)[-count:]
    sorted_names = names[idx]
    sorted_values = values[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_names, sorted_values, color="#4c78a8")
    ax.set_xlabel("Relative importance")
    ax.set_ylabel("Feature")
    ax.set_title("Model Feature Importance")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def render_header() -> None:
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        if PATHS["logo"].exists():
            st.image(str(PATHS["logo"]), width=110)
        else:
            st.warning("Optional branding asset missing: `assets/logo.png`")
    with col_title:
        st.markdown("# NorthStar - Process Advisor")
        st.markdown("### Laser Powder Bed Fusion (L-PBF) Optimization")
        st.markdown(
            "AI-driven guidance for predicting and optimizing print success in L-PBF systems."
        )

    video_base64 = load_video_base64(PATHS["video"])
    if video_base64:
        st.markdown(
            f"""
<div class="video-box">
    <video autoplay loop muted playsinline style="width:350px; border-radius: 12px;">
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
    </video>
</div>
""",
            unsafe_allow_html=True,
        )


def render_project_overview() -> None:
    st.markdown("## Project Overview")
    st.markdown(
        """
L-PBF is an additive manufacturing process that fuses metal powder layer-by-layer using a high-energy laser.
The process is sensitive to parameter interactions (for example power, scan velocity, and hatch spacing),
which makes trial-and-error expensive.

NorthStar predicts process success probability from selected machine and material settings.
Predictions are **decision-support only** and are **not a replacement for laboratory validation**.
"""
    )
    df = load_optional_dataframe(PATHS["processed_data"], "processed dataset")
    if df is not None:
        st.markdown("### Processed Dataset Snapshot")
        st.dataframe(df.head(10), use_container_width=True)


def render_process_advisor(pipeline) -> None:
    st.markdown("## Process Advisor")
    st.markdown(
        """
Use the controls below to define an L-PBF parameter set. The model predicts whether the setup is likely to
achieve a successful process outcome and reports a success probability.
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        power = st.number_input("Power [W]", 50, 500, 180)
        explain("Laser energy delivered to the powder bed.")
    with col2:
        velocity = st.number_input("Velocity [m/s]", 0.1, 3.0, 0.9)
        explain("Laser scan speed across the powder bed.")
    with col1:
        hatch = st.number_input("Hatch Spacing [um]", 10, 200, 90)
        explain("Distance between scan tracks.")
    with col2:
        beam = st.number_input("Beam Diameter [um]", 20, 200, 78)
        explain("Effective beam spot size.")
    with col1:
        layer = st.number_input("Powder Layer Thickness [um]", 10, 200, 50)
        explain("Thickness of each powder layer.")
    with col2:
        d90 = st.number_input("d90 [um]", 1, 60, 15)
        explain("Powder particle coarse fraction (90th percentile).")

    st.markdown("### Material and Process Settings")
    col1, col2 = st.columns(2)
    with col1:
        atomosphere = st.selectbox("Atmosphere of Build", ["Argon", "Nitrogen"])
        explain("Protective gas used during printing.")
    with col2:
        nucleants = st.selectbox("Nucleants", ["N11", "N12", "N13"])
        explain("Powder additive to promote solidification.")
    with col1:
        atom_atm = st.selectbox("Atomization Atmosphere", ["Gas", "Water"])
        explain("Powder production method.")
    with col2:
        same_layer = st.selectbox("Same Layer Scanned?", ["0", "1"])
        explain("Laser rescans same layer.")

    X = pd.DataFrame(
        [
            {
                "Power [W]": power,
                "Velocity [m/s]": velocity,
                "Hatch Spacing [μm]": hatch,
                "Beam Diameter [μm]": beam,
                "Powder Layer Thickness [μm]": layer,
                "d90 [um]": d90,
                "Atomosphere of build": atomosphere,
                "Nucleants": nucleants,
                "Atomization Atomosphere": atom_atm,
                "Same Layer Scanned?": same_layer,
                "Initial Powder Bed Temperature [K]": 300,
                "Substrate/Platform Temperature [K]": 100,
                "Surface Energy Density [J/mm2]": 2.5,
                "Linear Energy Density [J/mm]": 3.5,
                "Volume Energy Density [J/mm3]": 30,
                "Powder Recycled How Many Times?": 0,
                "Powder Shape [0-1]": 1.0,
                "Build Direction [degrees]": "90",
            }
        ]
    )

    if st.button("Predict"):
        pred = pipeline.predict(X)[0]
        prob = float(pipeline.predict_proba(X)[0][1])

        st.markdown("### Prediction Result")
        if pred == 1:
            st.markdown("<span class='success-badge'>SUCCESS</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='fail-badge'>FAIL</span>", unsafe_allow_html=True)

        st.markdown(f"**Probability of SUCCESS:** `{prob:.5f}`")
        st.caption(
            "Interpretation: this probability ranks expected success likelihood for decision support."
        )
        st.caption("Always validate recommendations with controlled experiments and lab testing.")

        st.write("---")
        st.markdown("### Model Feature Importance")
        st.caption("Top process parameters influencing predicted print success")
        try:
            importance_df, source = build_feature_importance_table(pipeline)
            top_df = importance_df.head(10)
            render_importance_bar(
                top_df["feature"].tolist(),
                top_df["importance"].to_numpy(),
                top_n=10,
            )
            if source == "exploratory":
                st.info(
                    "Exploratory fallback: this chart uses absolute correlation with printability "
                    "for key numeric process parameters."
                )
            st.caption(
                "Higher values indicate process parameters with greater influence on the model or "
                "stronger exploratory relationship with printability in the available dataset."
            )
        except Exception as exc:
            st.warning(f"Feature importance chart is unavailable: {exc}")

        st.write("---")
        st.markdown("### Power x Velocity Success Map")
        power_range = np.linspace(50, 500, 30)
        vel_range = np.linspace(0.2, 3.0, 30)
        grid = []
        for p in power_range:
            for v in vel_range:
                row = X.copy()
                row["Power [W]"] = p
                row["Velocity [m/s]"] = v
                grid.append(row.iloc[0])

        df_grid = pd.DataFrame(grid)
        probs = pipeline.predict_proba(df_grid)[:, 1]
        df_grid["prob"] = probs
        pivot = df_grid.pivot_table(index="Velocity [m/s]", columns="Power [W]", values="prob")

        fig, ax = plt.subplots(figsize=(9, 5))
        image = ax.imshow(pivot, cmap="viridis", origin="lower", aspect="auto")
        ax.set_title("Success Probability Across Power and Velocity")
        ax.set_xlabel("Power [W]")
        ax.set_ylabel("Velocity [m/s]")
        cbar = plt.colorbar(image, ax=ax)
        cbar.set_label("Success Probability")
        st.pyplot(fig)


def render_model_performance() -> None:
    st.markdown("## Model Performance")
    st.markdown(
        "If available, the confusion matrix and ROC curve generated during model evaluation are shown below."
    )
    if PATHS["confusion_matrix"].exists():
        st.image(str(PATHS["confusion_matrix"]), caption="Confusion Matrix", use_container_width=True)
    else:
        st.warning("Optional figure missing: `outputs/figures/confusion_matrix.png`")

    if PATHS["roc_curve"].exists():
        st.image(str(PATHS["roc_curve"]), caption="ROC Curve", use_container_width=True)
    else:
        st.warning("Optional figure missing: `outputs/figures/roc_curve.png`")

    st.info(
        "Validated scalar metrics are not embedded here. Export notebook metrics to "
        "`outputs/metrics/model_metrics.json` to display them in-app."
    )


def render_data_overview() -> None:
    st.markdown("## Data Overview")
    processed = load_optional_dataframe(PATHS["processed_data"], "processed dataset")
    best = load_optional_dataframe(PATHS["best_combinations"], "best success combinations")
    if processed is not None:
        st.markdown("### Processed Data")
        st.write(f"Rows: {processed.shape[0]} | Columns: {processed.shape[1]}")
        st.dataframe(processed.head(20), use_container_width=True)
    if best is not None:
        st.markdown("### Best Success Combinations")
        st.dataframe(best.head(20), use_container_width=True)


def render_optimization_insights() -> None:
    st.markdown("## Optimization Insights")
    st.markdown(
        """
Key process parameters typically include:
- **Power [W]**: higher values increase melt pool energy but may increase defects when excessive.
- **Velocity [m/s]**: controls interaction time and energy deposition.
- **Hatch Spacing [um]**: affects overlap and fusion consistency between scan tracks.
- **Layer Thickness and Beam Diameter**: influence volumetric energy density and part quality.

Use this app to prioritize candidate process windows, then validate shortlisted settings experimentally.
"""
    )


def main() -> None:
    st.set_page_config(page_title="NorthStar - Process Advisor", layout="wide")
    inject_theme()
    render_header()
    st.write("---")

    pipeline = load_pipeline(PATHS["model"])
    page = st.sidebar.radio(
        "Navigate",
        [
            "Project Overview",
            "Process Advisor",
            "Model Performance",
            "Data Overview",
            "Optimization Insights",
        ],
    )

    if page == "Project Overview":
        render_project_overview()
    elif page == "Process Advisor":
        if pipeline is None:
            st.warning(
                "Process Advisor requires the trained model at `models/best_model_smote.pkl`."
            )
        else:
            render_process_advisor(pipeline)
    elif page == "Model Performance":
        render_model_performance()
    elif page == "Data Overview":
        render_data_overview()
    elif page == "Optimization Insights":
        render_optimization_insights()


if __name__ == "__main__":
    main()
