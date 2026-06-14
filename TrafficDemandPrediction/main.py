import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

import lightgbm as lgb

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

test_index = test["Index"].copy()

train.drop(columns=["Index"], inplace=True)
test.drop(columns=["Index"], inplace=True)

# ─────────────────────────────────────────────
# TIME PARSING
# ─────────────────────────────────────────────

def parse_time(df):
    parts = df["timestamp"].astype(str).str.split(":", expand=True)

    df["hour"]   = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype(int)
    df["minute"] = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype(int)

    df["time_slot"] = df["hour"] * 4 + df["minute"] // 15
    return df

train = parse_time(train)
test  = parse_time(test)

# ─────────────────────────────────────────────
# BASIC IMPUTATION
# ─────────────────────────────────────────────

temp_by_geo = train.groupby("geohash")["Temperature"].median()

weather_by_geo = train.groupby("geohash")["Weather"].agg(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown"
)

road_by_geo = train.groupby("geohash")["RoadType"].agg(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown"
)

for df in [train, test]:
    df["Temperature"] = df["Temperature"].fillna(df["geohash"].map(temp_by_geo)).fillna(train["Temperature"].median())
    df["Weather"]     = df["Weather"].fillna(df["geohash"].map(weather_by_geo)).fillna("Unknown")
    df["RoadType"]    = df["RoadType"].fillna(df["geohash"].map(road_by_geo)).fillna("Unknown")

# ─────────────────────────────────────────────
# GEO FREQUENCY FEATURE
# ─────────────────────────────────────────────

geo_freq = train["geohash"].value_counts()

for df in [train, test]:
    df["geo_freq"] = df["geohash"].map(geo_freq).fillna(0).astype(int)

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer(df):

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["slot_sin"] = np.sin(2 * np.pi * df["time_slot"] / 96)
    df["slot_cos"] = np.cos(2 * np.pi * df["time_slot"] / 96)

    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 7)

    df["is_rush"] = df["hour"].isin([7,8,9,17,18,19]).astype(int)

    df["temp_x_slot"] = df["Temperature"] * df["time_slot"]
    df["temp_x_day"]  = df["Temperature"] * df["day"]

    df["weather_road"] = df["Weather"].astype(str) + "_" + df["RoadType"].astype(str)

    df["temp_x_hour"]  = df["Temperature"] * df["hour"]
    df["lanes_x_rush"] = df["NumberofLanes"] * df["is_rush"]
    df["day_x_slot"]   = df["day"] * df["time_slot"]

    df["temp_x_lanes"] = df["Temperature"] * df["NumberofLanes"]
    df["lanes_x_slot"] = df["NumberofLanes"] * df["time_slot"]

    # ✅ UPGRADE 3: NEW FEATURE
    df["rush_temp"] = df["is_rush"] * df["Temperature"]

    return df

train = engineer(train)
test  = engineer(test)

# ─────────────────────────────────────────────
# KFOLD SETUP
# ─────────────────────────────────────────────

kf = KFold(n_splits=5, shuffle=True, random_state=42)

global_mean = train["demand"].mean()
fold_splits = list(kf.split(train))

# ─────────────────────────────────────────────
# SAFE OOF TARGET ENCODING FUNCTION (UPDATED)
# ─────────────────────────────────────────────

def oof_te(train_df, test_df, group_col, out_col):

    train_df[out_col] = np.nan
    test_acc = np.zeros(len(test_df))

    for tr_idx, val_idx in fold_splits:

        grp_mean = (
            train_df.iloc[tr_idx]
            .groupby(group_col)["demand"]
            .mean()
        )

        train_df.loc[
            train_df.index[val_idx],
            out_col
        ] = (
            train_df.iloc[val_idx][group_col]
            .map(grp_mean)
            .fillna(global_mean)
            .clip(global_mean * 0.5, global_mean * 1.5)   # ✅ UPGRADE 2
            .values
        )

        test_acc += (
            test_df[group_col]
            .map(grp_mean)
            .fillna(global_mean)
            .clip(global_mean * 0.5, global_mean * 1.5)   # ✅ UPGRADE 2
            .values
        )

    train_df[out_col] = train_df[out_col].fillna(global_mean)
    test_df[out_col]  = test_acc / len(fold_splits)

    return train_df, test_df

# ─────────────────────────────────────────────
# TARGET ENCODINGS
# ─────────────────────────────────────────────

train["gh_te"] = np.nan
test["gh_te"] = 0

train["slot_te"] = np.nan
test_slot_acc = np.zeros(len(test))

# NOTE: unchanged TE loop
for tr_idx, val_idx in fold_splits:

    fold_tr = train.iloc[tr_idx]

    geo_mean = fold_tr.groupby("geohash")["demand"].mean()
    slot_mean = fold_tr.groupby("time_slot")["demand"].mean()

    train.loc[train.index[val_idx], "gh_te"] = (
        train.iloc[val_idx]["geohash"].map(geo_mean).fillna(global_mean).values
    )

    train.loc[train.index[val_idx], "slot_te"] = (
        train.iloc[val_idx]["time_slot"].map(slot_mean).fillna(global_mean).values
    )

    test["gh_te"] += test["geohash"].map(geo_mean).fillna(global_mean)
    test_slot_acc += test["time_slot"].map(slot_mean).fillna(global_mean)

train["gh_te"] = train["gh_te"].fillna(global_mean)
train["slot_te"] = train["slot_te"].fillna(global_mean)

test["gh_te"] = test["gh_te"] / 5
test["slot_te"] = test_slot_acc / 5

# NEW TE FEATURES
train, test = oof_te(train, test, "day", "day_te")
train, test = oof_te(train, test, "Weather", "weather_te")
train, test = oof_te(train, test, "RoadType", "road_te")

# ─────────────────────────────────────────────
# LABEL ENCODING
# ─────────────────────────────────────────────

cat_cols = ["geohash", "RoadType", "LargeVehicles", "Landmarks", "Weather", "weather_road"]

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col].astype(str), test[col].astype(str)])
    le.fit(combined)

    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))

# ─────────────────────────────────────────────
# FINAL FEATURES
# ─────────────────────────────────────────────

X = train.drop(["demand", "timestamp"], axis=1)

# ✅ UPGRADE 1: LOG TARGET
y = np.log1p(train["demand"])

X_test = test.drop(["timestamp"], axis=1)

# ─────────────────────────────────────────────
# LIGHTGBM MODEL
# ─────────────────────────────────────────────

params = dict(
    objective="regression",
    n_estimators=5000,
    learning_rate=0.01,
    num_leaves=127,
    min_child_samples=30,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
scores = []

for fold, (tr_idx, val_idx) in enumerate(fold_splits):

    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(200, verbose=False)]
    )

    val_preds = model.predict(X_val)

    oof_preds[val_idx] = val_preds
    test_preds += model.predict(X_test)

    score = r2_score(y_val, val_preds)
    scores.append(score)

    print(f"Fold {fold+1} R²: {score:.5f}")

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

print("\nMean CV R²:", np.mean(scores))
print("OOF R²:", r2_score(y, oof_preds))

# ─────────────────────────────────────────────
# SUBMISSION (UPDATED)
# ─────────────────────────────────────────────

final_preds = np.expm1(test_preds / 5)   # ✅ UPGRADE 1 FIX
final_preds = np.clip(final_preds, 0, 1)

submission = pd.DataFrame({
    "Index": test_index,
    "demand": final_preds
})

submission.to_csv("submission_u.csv", index=False)

print("Saved submission_u.csv")

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────

fi = pd.Series(model.feature_importances_, index=X.columns)
fi = fi.sort_values(ascending=False)

print("\nTop 15 features:")
print(fi.head(15))