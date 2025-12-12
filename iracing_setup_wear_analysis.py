"""
Richard Murray
COMSC 230 Final Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


CSV_PATH = "C:/COMSC/230/Combined_Data.csv"

# =========================================================
# Constants

TARGETS = ["Fastest Lap", "Best 5 Laps Avg", "Best 10 Laps Avg"]

SETUP_COLS = [
    "Left Air Pressure",
    "Right Air Pressure",
    "Rear Stagger",
    "Front Stagger",
    "Front Spring Rate",
    "Rear Spring Rate",
    "Ride Height",
    "Left Front Camber",
    "Right Front Camber",
    "Front ARB Preload",
]

WEAR_COLS = [
    "LFwearL", "LFwearM", "LFwearR",
    "LRwearL", "LRwearM", "LRwearR",
    "RFwearL", "RFwearM", "RFwearR",
    "RRwearL", "RRwearM", "RRwearR",
]


# =========================================================
# Calculated Values

def add_derived_metrics(df):

    # Worst wear band per tire (%remaining) 
    df["LF_remaining"] = df[["LFwearL", "LFwearM", "LFwearR"]].min(axis=1)
    df["LR_remaining"] = df[["LRwearL", "LRwearM", "LRwearR"]].min(axis=1)
    df["RF_remaining"] = df[["RFwearL", "RFwearM", "RFwearR"]].min(axis=1)
    df["RR_remaining"] = df[["RRwearL", "RRwearM", "RRwearR"]].min(axis=1)
  
    # Tire wear spread (%difference) 
    df["LF_band_spread"] = df[["LFwearL", "LFwearM", "LFwearR"]].max(axis=1) - df[["LFwearL", "LFwearM", "LFwearR"]].min(axis=1)
    df["LR_band_spread"] = df[["LRwearL", "LRwearM", "LRwearR"]].max(axis=1) - df[["LRwearL", "LRwearM", "LRwearR"]].min(axis=1)
    df["RF_band_spread"] = df[["RFwearL", "RFwearM", "RFwearR"]].max(axis=1) - df[["RFwearL", "RFwearM", "RFwearR"]].min(axis=1)
    df["RR_band_spread"] = df[["RRwearL", "RRwearM", "RRwearR"]].max(axis=1) - df[["RRwearL", "RRwearM", "RRwearR"]].min(axis=1)

    # Std dev of all four corners worst wear (Lower is more balanced) 
    df["corner_remaining_std"] = df[["LF_remaining", "LR_remaining", "RF_remaining", "RR_remaining"]].std(axis=1)

    # Avg tire remaining (%)
    df["overall_remaining"] = df[["LF_remaining", "LR_remaining", "RF_remaining", "RR_remaining"]].mean(axis=1)

    # All four tire band spread avg (%difference)
    df["overall_band_spread"] = df[["LF_band_spread", "LR_band_spread", "RF_band_spread", "RR_band_spread"]].mean(axis=1)

    # Side / axle totals using worst band wear
    df["left_remaining"] = df["LF_remaining"] + df["LR_remaining"]
    df["right_remaining"] = df["RF_remaining"] + df["RR_remaining"]
    df["front_remaining"] = df["LF_remaining"] + df["RF_remaining"]
    df["rear_remaining"] = df["LR_remaining"] + df["RR_remaining"]

    # Imbalances (lower is better)
    df["LR_remaining_balance_abs"] = (df["left_remaining"] - df["right_remaining"]).abs()
    df["FR_remaining_balance_abs"] = (df["front_remaining"] - df["rear_remaining"]).abs()

    # Lap time falloff
    df["falloff_5"] = df["Best 5 Laps Avg"] - df["Fastest Lap"]
    df["falloff_10"] = df["Best 10 Laps Avg"] - df["Fastest Lap"]

    return df

# =========================================================
# Sweet Spot Score

def zscore(series):
    s = series.astype(float)
    std = s.std(ddof=0)
    if std == 0:
        return s * 0.0
    return (s - s.mean()) / std

def add_sweet_spot_score(
    df,
    w_time=1.0,
    w_corner_balance=0.6,
    w_within_tire_balance=0.6,
    w_remaining=0.5,
    w_falloff=0.4
):

    # Weighted z-score contributions from:
    # - Best 10 lap pace (lower better)
    # - corner-to-corner balance of remaining (lower better)
    # - within-tire band imbalance (lower better)
    # - overall remaining (higher better, invert z-score)
    # - falloff_10 (lower better)

    z_time = zscore(df["Best 10 Laps Avg"])
    z_corner_bal = zscore(df["corner_remaining_std"])
    z_within_bal = zscore(df["overall_band_spread"])
    z_fall = zscore(df["falloff_10"])

    z_remaining = -zscore(df["overall_remaining"])

    df["sweet_spot_score"] = (
        w_time * z_time +
        w_corner_balance * z_corner_bal +
        w_within_tire_balance * z_within_bal +
        w_remaining * z_remaining +
        w_falloff * z_fall
    )
    return df


# =========================================================
# Pareto Fronts

def pareto_front(df, metrics):

    # Identifies which stints are not worse in all three categories than any other stint
    data = df[metrics].to_numpy(dtype=float)
    n = data.shape[0]
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        diffs = data - data[i]
        dominates = np.all(diffs <= 0, axis=1) & np.any(diffs < 0, axis=1)
        dominates[i] = False
        if np.any(dominates):
            is_pareto[i] = False

    return is_pareto


# =========================================================
# Model Training

def cv_metrics(model, X, y, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    maes, rmses, r2s = [], [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        maes.append(mean_absolute_error(y_test, pred))
        mse = mean_squared_error(y_test, pred)
        rmses.append(np.sqrt(mse))
        r2s.append(r2_score(y_test, pred))

    return float(np.mean(maes)), float(np.mean(rmses)), float(np.mean(r2s))


def run_modeling(df):
    setup_only = [c for c in SETUP_COLS if c in df.columns]
    wear_only = [c for c in WEAR_COLS if c in df.columns]

    derived = [
        "LF_remaining", "LR_remaining", "RF_remaining", "RR_remaining",
        "LF_band_spread", "LR_band_spread", "RF_band_spread", "RR_band_spread",
        "corner_remaining_std", "overall_remaining", "overall_band_spread",
        "LR_remaining_balance_abs", "FR_remaining_balance_abs",
        "falloff_5", "falloff_10"
    ]
    derived = [c for c in derived if c in df.columns]

    setup_plus = setup_only + derived
    wear_plus = wear_only + derived
    all_plus = setup_only + wear_only + derived

    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=42))
    ])

    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        min_samples_leaf=2
    )

    print("\n--- Cross-validated performance (5-fold) ---")
    for tgt in TARGETS:
        y = df[tgt]

        for label, cols, model in [
            ("Ridge | Setup only", setup_only, ridge),
            ("Ridge | Wear only", wear_only, ridge),
            ("Ridge | Setup + Derived", setup_plus, ridge),
            ("Ridge | Wear + Derived", wear_plus, ridge),
            ("Ridge | Setup + Wear + Derived", all_plus, ridge)
        ]:
            X = df[cols]
            mae, rmse, r2 = cv_metrics(model, X, y)
            print(f"{tgt:>16} | {label:<32} MAE={mae:.4f}s  RMSE={rmse:.4f}s  R2={r2:.3f}")


# =========================================================
# Visuals
def make_plots(df):

    # ---------------------------------------------------------
    # Lap time distributions
    plt.figure()
    plt.hist(df["Fastest Lap"], bins=15, alpha=0.6, label="Fastest")
    plt.hist(df["Best 5 Laps Avg"], bins=15, alpha=0.6, label="Best 5 Avg")
    plt.hist(df["Best 10 Laps Avg"], bins=15, alpha=0.6, label="Best 10 Avg")
    plt.title("Lap Time Distributions")
    plt.xlabel("Seconds")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # Pareto tradeoffs

    # number of top sweet spot scores to highlight
    TOP_N = 5  

    # Ensure masks exist
    is_pareto = df["is_pareto"].astype(bool)

    top_idx = df.nsmallest(TOP_N, "sweet_spot_score").index
    is_top = df.index.isin(top_idx)

    both = is_top & is_pareto
    pareto_only = is_pareto & ~is_top
    top_only = is_top & ~is_pareto
    neither = ~is_pareto & ~is_top

    # ---------------------------------------------------------
    # Best10 vs within-tire imbalance
    plt.figure()

    plt.scatter(
        df.loc[neither, "overall_band_spread"],
        df.loc[neither, "Best 10 Laps Avg"],
        alpha=0.25,
        label="Other"
    )
    plt.scatter(
        df.loc[pareto_only, "overall_band_spread"],
        df.loc[pareto_only, "Best 10 Laps Avg"],
        alpha=0.8,
        label="Pareto only"
    )
    plt.scatter(
        df.loc[top_only, "overall_band_spread"],
        df.loc[top_only, "Best 10 Laps Avg"],
        alpha=0.9,
        label=f"Top {TOP_N} only"
    )
    plt.scatter(
        df.loc[both, "overall_band_spread"],
        df.loc[both, "Best 10 Laps Avg"],
        color="red",
        edgecolors="black",
        s=90,
        label=f"Top {TOP_N} + Pareto"
    )

    plt.title("Pareto View 1: Best 10 Avg vs Within-Tire Imbalance")
    plt.xlabel("overall_band_spread  [lower = more even across tire width]")
    plt.ylabel("Best 10 Laps Avg (s)  [lower = better pace]")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # Best10 vs corner-to-corner remaining balance
    plt.figure()

    plt.scatter(
        df.loc[neither, "corner_remaining_std"],
        df.loc[neither, "Best 10 Laps Avg"],
        alpha=0.25,
        label="Other"
    )
    plt.scatter(
        df.loc[pareto_only, "corner_remaining_std"],
        df.loc[pareto_only, "Best 10 Laps Avg"],
        alpha=0.8,
        label="Pareto only"
    )
    plt.scatter(
        df.loc[top_only, "corner_remaining_std"],
        df.loc[top_only, "Best 10 Laps Avg"],
        alpha=0.9,
        label=f"Top {TOP_N} only"
    )
    plt.scatter(
        df.loc[both, "corner_remaining_std"],
        df.loc[both, "Best 10 Laps Avg"],
        color="red",
        edgecolors="black",
        s=90,
        label=f"Top {TOP_N} + Pareto"
    )

    plt.title("Pareto View 2: Best 10 Avg vs Corner Remaining Balance")
    plt.xlabel("corner_remaining_std  [lower = more even corner health]")
    plt.ylabel("Best 10 Laps Avg (s)  [lower = better pace]")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # Balance vs balance
    plt.figure()

    plt.scatter(
        df.loc[neither, "overall_band_spread"],
        df.loc[neither, "corner_remaining_std"],
        alpha=0.25,
        label="Other"
    )
    plt.scatter(
        df.loc[pareto_only, "overall_band_spread"],
        df.loc[pareto_only, "corner_remaining_std"],
        alpha=0.8,
        label="Pareto only"
    )
    plt.scatter(
        df.loc[top_only, "overall_band_spread"],
        df.loc[top_only, "corner_remaining_std"],
        alpha=0.9,
        label=f"Top {TOP_N} only"
    )
    plt.scatter(
        df.loc[both, "overall_band_spread"],
        df.loc[both, "corner_remaining_std"],
        color="red",
        edgecolors="black",
        s=90,
        label=f"Top {TOP_N} + Pareto"
    )

    plt.title("Pareto View 3: Within-Tire Balance vs Corner Balance")
    plt.xlabel("overall_band_spread  [lower = better within-tire balance]")
    plt.ylabel("corner_remaining_std  [lower = better corner balance]")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # Boxplot: Best10 by Rear Stagger
    if "Rear Stagger" in df.columns:
        groups = sorted(df["Rear Stagger"].dropna().unique())
        if len(groups) > 1:
            data = [df.loc[df["Rear Stagger"] == g, "Best 10 Laps Avg"] for g in groups]

            plt.figure()
            plt.boxplot(data, labels=[str(g) for g in groups])
            plt.title("Best 10 Avg by Rear Stagger")
            plt.xlabel("Rear Stagger")
            plt.ylabel("Best 10 Laps Avg (s)  [lower is better]")
            plt.tight_layout()
            plt.show()

    # ---------------------------------------------------------
    # Boxplot: within-tire imbalance by Rear Stagger
    if "Rear Stagger" in df.columns and "overall_band_spread" in df.columns:
        groups = sorted(df["Rear Stagger"].dropna().unique())
        if len(groups) > 1:
            data = [df.loc[df["Rear Stagger"] == g, "overall_band_spread"] for g in groups]

            plt.figure()
            plt.boxplot(data, labels=[str(g) for g in groups])
            plt.title("Within-Tire Imbalance by Rear Stagger")
            plt.xlabel("Rear Stagger")
            plt.ylabel("overall_band_spread  [lower is better]")
            plt.tight_layout()
            plt.show()

    # ---------------------------------------------------------
    # Ridge model RMSE comparison

    # Feature sets
    setup_only = [c for c in SETUP_COLS if c in df.columns]
    wear_only = [c for c in WEAR_COLS if c in df.columns]

    derived = [
        "LF_remaining", "LR_remaining", "RF_remaining", "RR_remaining",
        "LF_band_spread", "LR_band_spread", "RF_band_spread", "RR_band_spread",
        "corner_remaining_std", "overall_remaining", "overall_band_spread",
        "LR_remaining_balance_abs", "FR_remaining_balance_abs",
        "falloff_5", "falloff_10"
    ]
    derived = [c for c in derived if c in df.columns]

    setup_plus = setup_only + derived
    wear_plus = wear_only + derived
    all_plus = setup_only + wear_only + derived

    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=42))
    ])

    feature_sets = [
        ("Setup only", setup_only),
        ("Wear only", wear_only),
        ("Setup + Derived", setup_plus),
        ("Wear + Derived", wear_plus),
        ("Setup + Wear + Derived", all_plus),
    ]

    # Compute RMSEs for bar chart
    rmse_map = {tgt: [] for tgt in TARGETS}
    labels = []

    for name, cols in feature_sets:
        if len(cols) == 0:
            continue
        labels.append(name)
        X = df[cols]
        for tgt in TARGETS:
            y = df[tgt]
            _, rmse, _ = cv_metrics(ridge, X, y)
            rmse_map[tgt].append(rmse)

    # Plot grouped bars (one group per feature set)
    x = np.arange(len(labels))
    width = 0.25

    plt.figure()
    for i, tgt in enumerate(TARGETS):
        vals = rmse_map[tgt]
        plt.bar(x + i * width, vals, width, label=tgt)

    plt.title("Ridge Model RMSE Comparison")
    plt.xlabel("Feature Set")
    plt.ylabel("RMSE (s)  [lower is better]")
    plt.xticks(x + width, labels, rotation=15, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.show()

    
# =========================================================

def main():
    df = pd.read_csv(CSV_PATH)

    # Print the best stints by lap time
    print("\n--- Top 15 stints by Fastest Lap ---")
    show_cols = (
        (["Stint ID"] if "Stint ID" in df.columns else [])
        #+ SETUP_COLS
        + TARGETS
    )
    show_cols = [c for c in show_cols if c in df.columns]
    print(df.sort_values("Fastest Lap")[show_cols].head(15).to_string(index=False))

    # Print the best stints by Best 10 Laps Avg
    print("\n--- Top 15 stints by Best 10 Laps Avg ---")
    show_cols = (
        (["Stint ID"] if "Stint ID" in df.columns else [])
        #+ SETUP_COLS
        + TARGETS
    )
    show_cols = [c for c in show_cols if c in df.columns]
    print(df.sort_values("Best 10 Laps Avg")[show_cols].head(15).to_string(index=False))

    # Derived metrics + sweet spot score
    df = add_derived_metrics(df)
    df = add_sweet_spot_score(df)

    # Pareto: stints that are not beaten in all three categories by any other stint
    # - Best10 (lower is better)
    # - Std dev of all four corners worst wear (Lower is more balanced)
    # - All four tire band spread avg (%difference)
    pareto_metrics = ["Best 10 Laps Avg", "corner_remaining_std", "overall_band_spread"]
    df["is_pareto"] = pareto_front(df, pareto_metrics)

    # Print best sweet spot candidates
    print("\n--- Top 20 stints by sweet spot score ---")
    show_cols = (
        (["Stint ID"] if "Stint ID" in df.columns else [])
        #+ SETUP_COLS
        + [
            "sweet_spot_score", "is_pareto",
            "Fastest Lap", "Best 5 Laps Avg", "Best 10 Laps Avg",
            "LF_remaining", "LR_remaining", "RF_remaining", "RR_remaining",
            "corner_remaining_std", "overall_remaining",
            "LF_band_spread", "LR_band_spread", "RF_band_spread", "RR_band_spread",
            "overall_band_spread",
            "falloff_10"
        ]
    )
    show_cols = [c for c in show_cols if c in df.columns]
    print(df.sort_values("sweet_spot_score")[show_cols].head(20).to_string(index=False))

    # Print Pareto front
    print("\n--- Pareto front count ---")
    print(int(df["is_pareto"].sum()),
          "stints on Pareto front (Best10 + corner remaining balance + within-tire balance).")
    show_cols = (
        (["Stint ID"] if "Stint ID" in df.columns else [])
        #+ SETUP_COLS
        + [
            "sweet_spot_score", "is_pareto",
            "Fastest Lap", "Best 5 Laps Avg", "Best 10 Laps Avg",
            "LF_remaining", "LR_remaining", "RF_remaining", "RR_remaining",
            "corner_remaining_std", "overall_remaining",
            "LF_band_spread", "LR_band_spread", "RF_band_spread", "RR_band_spread",
            "overall_band_spread",
            "falloff_10"
        ]
    )
    show_cols = [c for c in show_cols if c in df.columns]
    print("\n--- Top 10 Pareto front stints by sweet spot score ---")
    print(df.sort_values(["is_pareto", "sweet_spot_score"], ascending=[False, True])[show_cols].head(10).to_string(index=False))

    run_modeling(df)
    make_plots(df)


if __name__ == "__main__":
    main()

