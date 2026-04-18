# Databricks notebook source
# DBTITLE 1,Cell 1
import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score

# ── 1. LOAD PROFILES ─────────────────────────────────────────
with open("/Volumes/workspace/default/crops/20crops/final_crops.json") as f:
    raw = json.load(f)

def parse_profile(c):
    stor  = c["storage"]
    ci    = c["chilling_injury"]
    sl    = c["shelf_life"]
    t_min = stor["temp_min_c"]
    t_max = stor["temp_max_c"]
    return {
        "t_opt":      (t_min + t_max) / 2 if t_max != t_min else t_min,
        "t_safe_min": t_min,
        "t_safe_max": t_max,
        "rh_min":     stor["humidity_min_pct"] or 50,
        "rh_max":     stor["humidity_max_pct"] or 70,
    }

PROFILES = {c["crop"].lower(): parse_profile(c) for c in raw}
print(f"Loaded {len(PROFILES)} crop profiles\n")

# ── 2. DELTA FEATURE COMPUTATION ─────────────────────────────
# delta_max_t : degrees max temp exceeded crop's safe ceiling  (0 if safe)
# dist_norm   : distance normalised 0–1 over 2500 km
# delta_rh    : humidity deviation outside crop's safe band    (0 if inside)
# delta_min_t : degrees min temp dropped below crop's safe floor (0 if safe)
# delta_avg_t : avg temp minus crop optimum                    (signed °C)

FEATURES = ["delta_max_t", "dist_norm", "delta_rh", "delta_min_t", "delta_avg_t"]

def compute_deltas(row, profiles):
    crop = str(row.get("crop", "")).lower().strip()
    p    = profiles.get(crop)
    if p is None:
        return {k: np.nan for k in FEATURES}

    avg_t = row["avg_temp_c"]
    min_t = row["min_temp_c"]
    max_t = row["max_temp_c"]
    avg_h = row["avg_humidity_%"]
    km    = row["distance_km"]

    return {
        "delta_max_t": round(max(0.0, max_t - p["t_safe_max"]),        3),
        "dist_norm":   round(min(km / 2500.0, 1.0),                    4),
        "delta_rh":    round(max(0.0, avg_h - p["rh_max"])
                           + max(0.0, p["rh_min"] - avg_h),            3),
        "delta_min_t": round(max(0.0, p["t_safe_min"] - min_t),        3),
        "delta_avg_t": round(avg_t - p["t_opt"],                       3),
    }

# ── 3. LOAD & TRANSFORM ──────────────────────────────────────
df = pd.read_csv("spoilage_dataset_v3.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print(f"Raw dataset: {df.shape[0]} rows | Columns: {list(df.columns)}\n")

deltas_df = df.apply(lambda r: pd.Series(compute_deltas(r, PROFILES)), axis=1)
df        = pd.concat([df, deltas_df], axis=1)

before = len(df)
df.dropna(subset=FEATURES, inplace=True)
print(f"After profile match: {len(df)} rows (dropped {before - len(df)} unmatched)\n")

# ── 4. FEATURES & TARGET ─────────────────────────────────────
TARGET = "spoilage_score"   # adjust to your actual column name

X = df[FEATURES].values
y = df[TARGET].values

# ── 5. TRAIN / TEST ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

# ── 6. TRAIN BOTH, PICK BEST ─────────────────────────────────
candidates = {
    "RandomForest":     RandomForestRegressor(
                            n_estimators=200, max_depth=12,
                            min_samples_leaf=3, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(
                            n_estimators=200, max_depth=5,
                            learning_rate=0.05, random_state=42),
}

results = {}
for name, m in candidates.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    results[name] = {
        "model": m,
        "r2":   r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae":  mean_absolute_error(y_test, y_pred),
    }

print("=" * 55)
print("  MODEL COMPARISON")
print("=" * 55)
for name, r in results.items():
    print(f"\n  {name}")
    print(f"    R2   : {r['r2']:.4f}")
    print(f"    RMSE : {r['rmse']:.4f}")
    print(f"    MAE  : {r['mae']:.4f}")

best_name  = max(results, key=lambda k: results[k]["r2"])
best_model = results[best_name]["model"]
print(f"\n  >> Using: {best_name} (R2={results[best_name]['r2']:.4f})\n")

# ── 7. FEATURE IMPORTANCE ────────────────────────────────────
print("=" * 55)
print("  WHAT DRIVES SPOILAGE")
print("=" * 55)
for fname, imp in sorted(zip(FEATURES, best_model.feature_importances_),
                         key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {fname:<16s}  {imp:.4f}  {bar}")

# ── 8. PREDICT FUNCTION ──────────────────────────────────────
def predict_spoilage(crop, avg_temp, min_temp, max_temp, humidity, km):
    p = PROFILES.get(crop.lower())
    if p is None:
        raise ValueError(f"'{crop}' not in profiles. Available: {sorted(PROFILES)}")

    x = np.array([[
        max(0.0, max_temp - p["t_safe_max"]),
        min(km / 2500.0, 1.0),
        max(0.0, humidity - p["rh_max"]) + max(0.0, p["rh_min"] - humidity),
        max(0.0, p["t_safe_min"] - min_temp),
        avg_temp - p["t_opt"],
    ]])

    score = float(np.clip(best_model.predict(x)[0], 0.0, 1.0))
    risk  = "Low" if score < 0.25 else ("Medium" if score < 0.55 else "High")
    return score, risk

# ── 9. DEMO ──────────────────────────────────────────────────
print(f"\n{'='*55}")
print("  DEMO — hot summer route (34°C avg, 780 km)")
print(f"{'='*55}\n")
print(f"  {'Crop':<16} {'Score':>6}  Risk")
print("  " + "-" * 40)
for crop in sorted(PROFILES):
    try:
        score, risk = predict_spoilage(
            crop, avg_temp=34, min_temp=29, max_temp=40, humidity=72, km=780
        )
        bar = "█" * int(score * 25)
        print(f"  {crop:<16} {score:.3f}   {risk:<6}  {bar}")
    except Exception as e:
        print(f"  {crop:<16} ERROR: {e}")