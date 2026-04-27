from __future__ import annotations

import base64
import json
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
SHOW_DEBUG = False


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
div.stButton > button {
    width: 100%;
    min-height: 3rem;
    font-size: 1.05rem;
    font-weight: 700;
    background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
    color: #ffffff;
    border: 1px solid #1e40af;
    border-radius: 10px;
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.35);
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8 0%, #1e3a8a 100%);
    border-color: #1e3a8a;
}
div[data-testid="stExpander"] {
    border: 1px solid #2b2f36;
    border-radius: 8px;
    background-color: #151a21;
}
div[data-testid="stExpander"] details {
    background-color: #151a21;
    border-radius: 8px;
}
div[data-testid="stExpander"] details summary {
    background-color: #1b2230 !important;
    color: #e8eefc !important;
    border-radius: 8px;
}
div[data-testid="stExpander"] details summary p,
div[data-testid="stExpander"] details summary span,
div[data-testid="stExpander"] details summary div {
    color: #e8eefc !important;
}
div[data-testid="stExpander"] details summary:hover,
div[data-testid="stExpander"] details summary:focus,
div[data-testid="stExpander"] details summary:focus-visible,
div[data-testid="stExpander"] details[open] summary,
div[data-testid="stExpander"] details[open] summary:hover {
    background-color: #243047 !important;
    color: #ffffff !important;
    outline: none !important;
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

    fig, ax = plt.subplots(figsize=(8.2, 3.8), dpi=120)
    ax.barh(sorted_names, sorted_values, color="#4c78a8")
    ax.set_xlabel("Relative importance")
    ax.set_ylabel("Feature")
    ax.set_title("Model Feature Importance")
    fig.tight_layout(pad=1.0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_header() -> None:
    with st.container():
        col_left, col_right = st.columns([3, 2], gap="large")
        with col_left:
            st.markdown("# NorthStar - Process Advisor")
            st.markdown("### Laser Powder Bed Fusion (L-PBF) Optimization")
            st.markdown(
                "A portfolio demonstration app that uses machine learning to support process parameter "
                "selection and print success prediction in additive manufacturing."
            )
        with col_right:
            video_base64 = load_video_base64(PATHS["video"])
            media_col1, media_col2 = st.columns(2, gap="small")
            with media_col1:
                if video_base64:
                    st.markdown(
                        f"""
<div class="video-box">
    <video autoplay loop muted playsinline style="width:100%; max-width:260px; border-radius: 12px;">
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
    </video>
</div>
""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("Video preview unavailable.")
            with media_col2:
                if PATHS["logo"].exists():
                    st.image(str(PATHS["logo"]), width=220)
                else:
                    st.caption("Freefuse logo unavailable.")

        card1, card2, card3 = st.columns(3, gap="medium")
        with card1:
            st.info("**AI Process Guidance**\n\nExplore viable parameter windows for L-PBF decision support.")
        with card2:
            st.info("**Print Success Prediction**\n\nEstimate likelihood of build success from selected inputs.")
        with card3:
            st.info("**Optimization Insights**\n\nReview practical trends for improving print outcomes.")


def render_dataset_summary() -> None:
    st.markdown("## Dataset Summary")
    st.markdown(
        """
This project uses an L-PBF process dataset designed to simulate realistic additive manufacturing conditions.
It includes variables such as laser power, scan speed, hatch spacing, layer thickness, energy density,
material descriptors, and quality indicators.
The model focuses on predicting build success and supporting process optimization insights.
"""
    )
    processed = load_optional_dataframe(PATHS["processed_data"], "processed dataset")
    if processed is not None:
        with st.expander("View compact dataset preview", expanded=False):
            st.caption(f"Rows: {processed.shape[0]} | Columns: {processed.shape[1]}")
            st.dataframe(processed.head(8), use_container_width=True)


def calculate_energy_density(power: float, velocity: float, hatch_um: float, layer_um: float) -> float:
    hatch_mm = hatch_um / 1000.0
    layer_mm = layer_um / 1000.0
    velocity_mm_s = velocity * 1000.0
    denominator = velocity_mm_s * hatch_mm * layer_mm
    if denominator <= 0:
        return 0.0
    return float(power / denominator)


def build_dynamic_advisor_result(
    power: float,
    velocity: float,
    hatch: float,
    layer: float,
    pred: int,
    prob: float,
) -> dict:
    energy_density = calculate_energy_density(power, velocity, hatch, layer)
    warnings = []
    suggestions = []
    outcome = "Success"
    # Tuned for current UI input ranges and converted units used in calculate_energy_density.
    lower_threshold = 35.0
    upper_threshold = 110.0

    if energy_density < lower_threshold:
        outcome = "Risk of Failure"
        warnings.append("The selected parameters may not provide enough energy for stable fusion.")
        suggestions.append("Increase laser power, reduce scan speed, or reduce hatch spacing.")
    elif energy_density > upper_threshold:
        outcome = "Risk of Failure"
        warnings.append(
            "The selected parameters may introduce excessive energy, increasing overheating or keyholing risk."
        )
        suggestions.append("Reduce laser power, increase scan speed, or increase hatch spacing.")
    else:
        warnings.append("The selected parameters are within a more stable process window.")
        suggestions.append("Keep parameters close to this range and validate with controlled test prints.")

    if power >= 280 and velocity <= 0.7:
        warnings.append("High power combined with low scan speed may increase overheating or keyholing risk.")
        suggestions.append("Reduce laser power slightly or increase scan speed to avoid excessive heat input.")

    if power <= 140 and velocity >= 1.6:
        warnings.append("Low power combined with high scan speed may increase under-fusion risk.")
        suggestions.append("Increase laser power or reduce scan speed to improve melt pool stability.")

    if hatch >= 120:
        warnings.append("Large hatch spacing may reduce overlap between scan tracks and increase lack-of-fusion risk.")
        suggestions.append("Reduce hatch spacing to improve scan-track overlap and bonding consistency.")

    if layer >= 80:
        warnings.append("Higher layer thickness may require more energy input to maintain stable fusion.")
        suggestions.append("Lower layer thickness or raise effective energy input for improved fusion stability.")

    if outcome == "Risk of Failure":
        risk_level = "High"
    elif prob >= 0.78:
        risk_level = "Low"
    elif prob >= 0.55:
        risk_level = "Medium"
    else:
        risk_level = "High"

    main_reason = warnings[0] if warnings else "No major rule-based risk was detected from the selected inputs."
    practical_suggestion = (
        suggestions[0]
        if suggestions
        else "Run a small validation build and adjust one parameter at a time to improve repeatability."
    )
    interpretation = (
        "The selected setup is predicted to have a stronger chance of successful printability."
        if outcome == "Success"
        else "The selected setup may require adjustment before physical validation."
    )

    low_energy_condition = energy_density < lower_threshold or (power <= 150 and velocity >= 1.2)
    high_energy_condition = energy_density > upper_threshold or (power >= 280 and velocity <= 0.7)

    if low_energy_condition:
        power_speed_note = (
            f"Low power combined with high scan speed may reduce fusion stability; current risk is **{risk_level}**."
        )
        practical_suggestion = "Increase laser power, reduce scan speed, or reduce hatch spacing."
    elif high_energy_condition:
        power_speed_note = (
            f"High power combined with low scan speed may increase overheating/keyholing risk; current risk is **{risk_level}**."
        )
        practical_suggestion = "Reduce laser power, increase scan speed, or increase hatch spacing."
    else:
        power_speed_note = (
            f"The selected power and scan speed are within a more balanced process range; current risk is **{risk_level}**."
        )

    if hatch >= 120 and layer >= 80:
        hatch_layer_note = (
            "Large hatch spacing and high layer thickness can reduce track overlap and increase lack-of-fusion sensitivity."
        )
    elif hatch >= 120 or layer >= 80:
        hatch_layer_note = (
            f"Hatch `{hatch:.0f} um` and layer `{layer:.0f} um` are relatively aggressive and may increase fusion sensitivity."
        )
    else:
        hatch_layer_note = (
            f"Hatch `{hatch:.0f} um` and layer `{layer:.0f} um` support stronger overlap and stable fusion consistency."
        )

    return {
        "pred": int(pred),
        "prob": float(prob),
        "outcome": outcome,
        "interpretation": interpretation,
        "energy_density": energy_density,
        "risk_level": risk_level,
        "main_reason": main_reason,
        "practical_adjustment": practical_suggestion,
        "rule_based_details": warnings,
        "optimization_insights": {
            "power_speed_balance": power_speed_note,
            "hatch_layer_balance": hatch_layer_note,
            "practical_recommendation": practical_suggestion,
        },
        "inputs": {
            "power": float(power),
            "velocity": float(velocity),
            "hatch": float(hatch),
            "layer": float(layer),
        },
    }


def get_current_input_signature(inputs: dict[str, float | str]) -> str:
    return json.dumps(inputs, sort_keys=True, ensure_ascii=True)


def render_process_advisor(pipeline) -> None:
    st.markdown("## Process Advisor")
    st.markdown(
        """
Use the controls below to define an L-PBF parameter set. The model predicts whether the setup is likely to
achieve a successful process outcome and reports a success probability.
"""
    )

    with st.container():
        col1, col2 = st.columns(2)
    with col1:
        power = st.number_input("Power [W]", 50, 500, 180, key="power_input")
        explain("Laser energy delivered to the powder bed.")
    with col2:
        velocity = st.number_input("Velocity [m/s]", 0.1, 3.0, 0.9, key="velocity_input")
        explain("Laser scan speed across the powder bed.")
    with col1:
        hatch = st.number_input("Hatch Spacing [um]", 10, 200, 90, key="hatch_input")
        explain("Distance between scan tracks.")
    with col2:
        beam = st.number_input("Beam Diameter [um]", 20, 200, 78, key="beam_input")
        explain("Effective beam spot size.")
    with col1:
        layer = st.number_input("Powder Layer Thickness [um]", 10, 200, 50, key="layer_input")
        explain("Thickness of each powder layer.")
    with col2:
        d90 = st.number_input("d90 [um]", 1, 60, 15, key="d90_input")
        explain("Powder particle coarse fraction (90th percentile).")

    st.markdown("### Material and Process Settings")
    col1, col2 = st.columns(2)
    with col1:
        atomosphere = st.selectbox("Atmosphere of Build", ["Argon", "Nitrogen"], key="atmosphere_input")
        explain("Protective gas used during printing.")
    with col2:
        nucleants = st.selectbox("Nucleants", ["N11", "N12", "N13"], key="nucleants_input")
        explain("Powder additive to promote solidification.")
    with col1:
        atom_atm = st.selectbox("Atomization Atmosphere", ["Gas", "Water"], key="atomization_input")
        explain("Powder production method.")
    with col2:
        same_layer = st.selectbox("Same Layer Scanned?", ["0", "1"], key="same_layer_input")
        explain("Laser rescans same layer.")

    current_inputs = {
        "power": float(power),
        "velocity": float(velocity),
        "hatch": float(hatch),
        "beam": float(beam),
        "layer": float(layer),
        "d90": float(d90),
        "atmosphere": str(atomosphere),
        "nucleants": str(nucleants),
        "atomization": str(atom_atm),
        "same_layer": str(same_layer),
    }
    current_signature = get_current_input_signature(current_inputs)

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

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
    if st.button("Predict / Analyze", key="predict_button", use_container_width=True):
        pred = pipeline.predict(X)[0]
        prob = float(pipeline.predict_proba(X)[0][1])
        st.session_state["last_advisor_result"] = build_dynamic_advisor_result(
            power=power,
            velocity=velocity,
            hatch=hatch,
            layer=layer,
            pred=int(pred),
            prob=prob,
        )
        st.session_state["last_prediction_signature"] = current_signature
    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

    result = st.session_state.get("last_advisor_result")
    last_signature = st.session_state.get("last_prediction_signature")
    is_stale_result = result is not None and last_signature != current_signature
    if is_stale_result:
        st.info("Inputs were changed after the last prediction. Click **Predict / Analyze** to refresh recommendations.")
    st.session_state["prediction_is_fresh"] = bool(result and not is_stale_result)

    if result and not is_stale_result:
        st.markdown("### Process Advisor Result")
        selected_cols = st.columns(4)
        with selected_cols[0]:
            st.metric("Selected Laser Power [W]", f"{result['inputs']['power']:.0f}")
        with selected_cols[1]:
            st.metric("Selected Scan Speed [m/s]", f"{result['inputs']['velocity']:.2f}")
        with selected_cols[2]:
            st.metric("Selected Hatch Spacing [um]", f"{result['inputs']['hatch']:.0f}")
        with selected_cols[3]:
            st.metric("Selected Layer Thickness [um]", f"{result['inputs']['layer']:.0f}")

        result_cols = st.columns(4)
        with result_cols[0]:
            st.metric("Predicted Build Outcome", result["outcome"])
        with result_cols[1]:
            st.metric("Success Probability", f"{result['prob']:.3f}")
        with result_cols[2]:
            st.metric("Energy Density [J/mm3]", f"{result['energy_density']:.2f}")
        with result_cols[3]:
            st.metric("Risk Level", result["risk_level"])

        if result["outcome"] == "Success":
            st.markdown("<span class='success-badge'>SUCCESS</span>", unsafe_allow_html=True)
            st.caption(f"Interpretation: {result['interpretation']}")
        else:
            st.markdown("<span class='fail-badge'>RISK OF FAILURE</span>", unsafe_allow_html=True)
            st.caption(f"Interpretation: {result['interpretation']}")

        st.markdown(f"**Main reason:** {result['main_reason']}")
        st.markdown(f"**Practical parameter adjustment:** {result['practical_adjustment']}")
        if result["rule_based_details"]:
            with st.expander("Rule-based explanation details", expanded=True):
                for item in result["rule_based_details"]:
                    st.markdown(f"- {item}")
        if SHOW_DEBUG:
            st.markdown("**Debug - current input/recommendation payload**")
            st.json(result, expanded=False)
        st.caption("Use this recommendation as decision support and validate with controlled experiments.")


def render_model_level_visual_insights(pipeline) -> None:
    st.markdown("## Model-Level Visual Insights")
    with st.expander("Global Model Evaluation", expanded=False):
        st.caption(
            "These charts summarize the available model/sample evaluation artifacts and are not recalculated "
            "for each individual prediction."
        )

        st.markdown("### Feature Importance")
        try:
            importance_df, _ = build_feature_importance_table(pipeline)
            top_df = importance_df.head(10)
            render_importance_bar(
                top_df["feature"].tolist(),
                top_df["importance"].to_numpy(),
                top_n=10,
            )
        except Exception as exc:
            st.caption(f"Feature importance chart is unavailable: {exc}")

        metrics = None
        if PATHS["metrics"].exists():
            try:
                metrics = json.loads(PATHS["metrics"].read_text(encoding="utf-8"))
            except Exception:
                metrics = None

        if metrics:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 'n/a')}")
            with m2:
                st.metric("Precision", f"{metrics.get('precision', 'n/a')}")
            with m3:
                st.metric("Recall", f"{metrics.get('recall', 'n/a')}")

        charts = st.columns(2)
        with charts[0]:
            if PATHS["confusion_matrix"].exists():
                st.image(
                    str(PATHS["confusion_matrix"]),
                    caption="Confusion Matrix",
                    use_container_width=True,
                    width=460,
                )
            else:
                st.caption("Confusion matrix image is not available.")
        with charts[1]:
            if PATHS["roc_curve"].exists():
                st.image(
                    str(PATHS["roc_curve"]),
                    caption="ROC Curve",
                    use_container_width=True,
                    width=460,
                )
            else:
                st.caption("ROC curve image is not available.")


def render_optimization_insights(advisor_result: dict) -> None:
    st.markdown("## Optimization Insights")
    insights = advisor_result["optimization_insights"]

    insight_1, insight_2, insight_3 = st.columns(3)
    with insight_1:
        st.success(f"**Power & Speed Balance**\n\n{insights['power_speed_balance']}")
    with insight_2:
        st.success(f"**Hatch Spacing & Layer Thickness**\n\n{insights['hatch_layer_balance']}")
    with insight_3:
        st.success(f"**Practical Recommendation**\n\n{insights['practical_recommendation']}")

    with st.expander("Show recommendation logic", expanded=False):
        st.markdown(
            """
- Increase success likelihood by operating within stable energy-density windows.
- Reduce risk by avoiding extreme combinations of power and velocity.
- Prioritize repeatable settings and validate in controlled lab prints.
"""
        )


def main() -> None:
    st.set_page_config(page_title="NorthStar - Process Advisor", layout="wide")
    if "last_advisor_result" not in st.session_state:
        st.session_state["last_advisor_result"] = None
    if "last_prediction_signature" not in st.session_state:
        st.session_state["last_prediction_signature"] = None
    if "prediction_is_fresh" not in st.session_state:
        st.session_state["prediction_is_fresh"] = False

    inject_theme()
    render_header()
    st.divider()

    pipeline = load_pipeline(PATHS["model"])
    render_dataset_summary()
    st.divider()

    if pipeline is None:
        st.warning("Process Advisor requires the trained model at `models/best_model_smote.pkl`.")
    else:
        render_process_advisor(pipeline)
        has_fresh_prediction = bool(st.session_state.get("prediction_is_fresh"))
        if has_fresh_prediction:
            st.divider()
            render_model_level_visual_insights(pipeline)
            st.divider()
            render_optimization_insights(st.session_state["last_advisor_result"])


if __name__ == "__main__":
    main()
