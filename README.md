# 💧 Smart Hydration — Stacked Model

A lightweight Streamlit app that recommends **in-session water intake (L)** for a workout based on simple user inputs.  
The model is a **Stacking Regressor** (Random Forest + XGBoost + Ridge) wrapped in a single scikit-learn `Pipeline`.

## ✨ What it does
- Collects **Age, Gender, Exercise_Type (UPPERCASE), Duration (min), Temperature (°C), Weight (kg), Height (cm)**.
- For distance-based types (e.g., `RUNNING`, `WALKING`, `TREADMILL_RUNNING`), it also asks for **Distance (km)**.
- Derives features consistently with training: **BMI, Pace_mps, LongDistanceFlag (≥10 km), Effort_Score**.
- Loads one artifact: **`hydration_pipeline.pkl`**, predicts liters, and shows a big green result box.

## 🧠 How the label was built (training-time)
Training used a distance-aware label function that extends heart-rate & calorie logic with:
- **Long distance boost** for `RUNNING`/`WALKING`/`TREADMILL_RUNNING` (≥10km, ≥21.1km, ≥30km bands).
- **Pace factor** (higher pace → higher sweat rate).
- Temperature, gender, and fat% modifiers.
> At inference time, the app only uses the simple inputs + derived features. No HR or calories required.

## 🗂️ Repo structure
