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
- **Long distance boost** for `RUNNING` / `WALKING` / `TREADMILL_RUNNING` (≥10km, ≥21.1km, ≥30km bands).
- **Pace factor** (higher pace → higher sweat rate).
- Temperature, gender, and fat% modifiers.

> At inference time, the app only uses the simple inputs + derived features. No HR or calories required.

## 📂 Repo structure
```
.
├─ app.py                         # Streamlit UI (loads hydration_pipeline.pkl)
├─ requirements.txt               # Pinned versions (match training)
├─ train_hydration_pipeline.py    # Stacked model training (distance-aware label)
├─ combined_data1-2.csv           # (optional) training data location
└─ hydration_pipeline.pkl         # Saved artifact created by the training script
```

## 🚀 Quick start

### 1) Create & activate a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate           # macOS/Linux
# .venv\Scripts\activate          # Windows
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Train (optional if you already have the artifact)
```bash
python train_hydration_pipeline.py
```
This will create `hydration_pipeline.pkl` in the project folder.

### 4) Run the app
```bash
streamlit run app.py
```

## 🧩 Model artifact (`hydration_pipeline.pkl`)
A `joblib`-saved dictionary:
```python
{
  "pipeline": sklearn.Pipeline(...),        # preprocessor (ColumnTransformer) → StackingRegressor
  "app_numeric": [...],                     # numeric feature names used at inference
  "app_categ": ["Gender","Exercise_Type"],  # categorical feature names
  "exercise_types_distance_sensitive": ["WALKING","TREADMILL_RUNNING","RUNNING"]
}
```
The app introspects the fitted OneHotEncoder to populate **all** exercise types seen in training (UPPERCASE).

## 🧮 Inputs & engineered features
**Inputs:**  
`Age`, `Gender` (UPPERCASE), `Exercise_Type` (UPPERCASE), `Exercise_Duration (min)`, `Temperature_C`, `Weight_kg`, `Height_cm`, `[Distance_km if distance type]`  

**Derived:**  
- `Exercise_Duration_Seconds = minutes * 60`  
- `BMI = weight / (height_m^2)`  
- `Pace_mps = Distance_m / Duration_sec`  
- `LongDistanceFlag = 1 if distance type and distance_km ≥ 10 else 0`  
- `Effort_Score = hours * (1 + 0.25 * LongDistanceFlag)`

## 🛠️ Training notes
- The training script normalizes column names from the CSV and **uppercases** `Gender` and `Exercise_Type` to match the app.
- `ColumnTransformer(..., remainder="drop")` is used to avoid serializing private remainder internals.
- The stacked model uses:
  - `RandomForestRegressor`
  - `XGBRegressor`
  - `Ridge` as base + final estimator

## 🧷 Version pinning (important)
`requirements.txt` pins versions that match the training environment to avoid pickle incompatibilities:
- scikit-learn **1.6.1**
- joblib **1.5.1**
- xgboost **3.0.3**
- numpy **2.0.2**
- pandas **2.2.2**
- streamlit (latest)

If you see an error like:
```
AttributeError: Can't get attribute '_RemainderColsList' ...
```
you’re likely loading the model with a different scikit-learn build. Use the pinned versions, or retrain and re-save under your local environment. The app also includes a tiny compatibility shim as an extra safeguard.

## 🧪 Running in Google Colab
```python
from google.colab import drive
drive.mount('/content/drive')

# Create project dir
import os
PROJECT_DIR = "/content/drive/MyDrive/smart_hydration/hydration_app"
os.makedirs(PROJECT_DIR, exist_ok=True)

# Train & save artifact into Drive
# (edit paths in train_hydration_pipeline.py to point to your CSV in Drive)
```
Then download `hydration_pipeline.pkl` and `app.py` from Drive and run locally with Streamlit.

## ⚠️ Disclaimer
This tool provides **estimates**. It is **not medical advice**. Adjust based on your own experience and any guidance from healthcare professionals.
