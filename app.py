# app.py
import sys
import numpy as np
import pandas as pd
import streamlit as st

# --- Compatibility shim for some sklearn pickles (safe to keep) ---
try:
    import sklearn.compose._column_transformer as _ctm
    if not hasattr(_ctm, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _ctm._RemainderColsList = _RemainderColsList
except Exception:
    pass

import joblib

st.set_page_config(page_title="Smart Hydration", page_icon="💧")
st.markdown(
    """
    <style>
    .big-success {
        padding: 1.5em; 
        font-size: 1.5em; 
        border-radius: 10px;
        background-color: #1e7e34;
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("💧 Smart Hydration")

@st.cache_resource
def load_model():
    blob = joblib.load("hydration_pipeline.pkl")  # same folder as app.py
    pipe = blob["pipeline"]

    app_numeric = blob.get("app_numeric", [
        "Age","Exercise_Duration_Seconds","Temperature_C","Weight_kg","Height_cm",
        "Distance_m","BMI","Pace_mps","LongDistanceFlag","Effort_Score"
    ])
    app_categ = blob.get("app_categ", ["Gender","Exercise_Type"])

    # Pull categories from fitted OHE so UI matches training exactly
    ohe = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"]
    cat_map = {col: [str(v).upper() for v in cats] for col, cats in zip(app_categ, ohe.categories_)}

    exercise_types_all = sorted(cat_map.get("Exercise_Type", []))
    gender_opts = cat_map.get("Gender", ["MALE","FEMALE"])

    dist_types = set(blob.get("exercise_types_distance_sensitive", []))
    if not dist_types:
        dist_types = {t for t in exercise_types_all if any(k in t for k in ["RUN", "WALK", "TREADMILL"])}

    return blob, pipe, app_numeric, app_categ, exercise_types_all, dist_types, gender_opts

blob, pipe, APP_NUMERIC, APP_CATEG, EX_TYPES_ALL, DIST_TYPES, GENDER_OPTS = load_model()

# ----- Sidebar UI (sliders; names & casing match training) -----
with st.sidebar:
    st.header("Session Inputs")

    age = st.slider("Age", min_value=12, max_value=90, value=27, step=1)
    gender = st.radio("Gender", options=GENDER_OPTS,
                      index=min(1, len(GENDER_OPTS)-1), horizontal=True)
    exercise_type = st.selectbox("Exercise_Type", options=EX_TYPES_ALL)

    duration_min = st.slider("Exercise_Duration (minutes)", min_value=5, max_value=400, value=60, step=5)
    temp = st.slider("Temperature_C (°C)", min_value=-5.0, max_value=50.0, value=18.0, step=0.5)
    weight = st.slider("Weight_kg", min_value=35.0, max_value=160.0, value=60.0, step=0.5)
    height = st.slider("Height_cm", min_value=120.0, max_value=220.0, value=165.0, step=0.5)

    if exercise_type in DIST_TYPES:
        distance_km = st.slider("Distance_km", min_value=0.0, max_value=60.0, value=8.0, step=0.5)
        distance_m = distance_km * 1000.0
    else:
        distance_km, distance_m = 0.0, 0.0

# ----- Derived features (must mirror training) -----
duration_sec = int(duration_min * 60)
bmi = float(weight) / ((float(height)/100.0)**2)
pace_mps = (distance_m / duration_sec) if duration_sec > 0 else 0.0
long_distance_flag = 1 if (exercise_type in DIST_TYPES and distance_km >= 10.0) else 0
effort_score = (duration_sec / 3600.0) * (1.0 + 0.25 * long_distance_flag)

# Build exact model input row
row = {
    "Age": int(age),
    "Exercise_Duration_Seconds": duration_sec,
    "Temperature_C": float(temp),
    "Weight_kg": float(weight),
    "Height_cm": float(height),
    "Distance_m": float(distance_m),
    "BMI": float(bmi),
    "Pace_mps": float(pace_mps),
    "LongDistanceFlag": int(long_distance_flag),
    "Effort_Score": float(effort_score),
    "Gender": str(gender).upper(),
    "Exercise_Type": str(exercise_type).upper(),
}
X_user = pd.DataFrame([row])

st.subheader("Inputs (as fed to the model)")
st.dataframe(X_user)

if st.button("Predict hydration (L)"):
    pred = float(pipe.predict(X_user)[0])
    pred = float(np.clip(pred, 0.2, 4.0))
    #st.success(f"Recommended intake: **{pred:.2f} L** for this session.")
    st.markdown(
    f"<div class='big-success'>💧 Recommended intake: {pred:.2f} L for this session.</div>",
    unsafe_allow_html=True
)
    if row["Exercise_Type"] in DIST_TYPES and distance_km >= 10.0:
        st.caption("Long-distance session detected (≥10 km); effort adjusted accordingly.")
    st.caption("Model estimate — adjust with your experience and any medical advice.")
