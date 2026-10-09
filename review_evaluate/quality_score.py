import pandas as pd
import numpy as np


# ============================================================
# Config
# ============================================================

INPUT_CSV = "all_reviews_probability.csv"

OUTPUT_CSV = "all_reviews_quality_critic_final.csv"

WEIGHT_OUTPUT_CSV = "critic_weights_final.csv"

CORRELATION_OUTPUT_CSV = "critic_dimension_correlation.csv"

CONSTRUCTIVE_COL = "constructive_score_binary"


HEDGE_IS_UNCERTAINTY = True

EPS = 1e-12


# ============================================================
# CRITIC Weight Method
# ============================================================

def critic_weight_method(data, eps=1e-12):
    """
    CRITIC:
    Criteria Importance Through Intercriteria Correlation

    """

    X_raw = data.copy().astype(float)

    # ========================================================
    # 1. Min-Max normalization
    # ========================================================

    X = pd.DataFrame(
        index=X_raw.index,
        columns=X_raw.columns,
        dtype=float
    )

    for col in X_raw.columns:

        col_min = X_raw[col].min()
        col_max = X_raw[col].max()

        denominator = col_max - col_min

        if abs(denominator) < eps:

            X[col] = 0.0

        else:

            X[col] = (
                X_raw[col] - col_min
            ) / denominator

    # ========================================================
    # 2. Standard deviation
    # ========================================================

    # ddof=0:
    normalized_std = X.std(
        axis=0,
        ddof=0
    )

    # ========================================================
    # 3. Correlation matrix
    # ========================================================

    correlation = X.corr(
        method="pearson"
    )

    correlation = correlation.fillna(0.0)

    np.fill_diagonal(
        correlation.values,
        1.0
    )

    # ========================================================
    # 4. Conflict
    # ========================================================

    # conflict_j = sum_k(1 - r_jk)
    conflict = (
        1.0 - correlation
    ).sum(axis=1)

    # ========================================================
    # 5. CRITIC information quantity
    # ========================================================

    information = (
        normalized_std
        * conflict
    )

    # ========================================================
    # 6. CRITIC weights
    # ========================================================

    if information.sum() < eps:

        weights = pd.Series(
            np.ones(len(X.columns))
            / len(X.columns),
            index=X.columns
        )

    else:

        weights = (
            information
            / information.sum()
        )

    return (
        weights,
        X,
        normalized_std,
        conflict,
        information,
        correlation
    )


# ============================================================
# Load data
# ============================================================

print("=" * 75)
print("Loading dataset")
print("=" * 75)

df = pd.read_csv(INPUT_CSV)

print(f"Number of reviews: {len(df):,}")
print(f"Number of columns: {len(df.columns)}")


# ============================================================
# Required columns
# ============================================================

required_columns = [
    "hedge_score",
    CONSTRUCTIVE_COL,
    "substan_score",
    "politeness_score",
    "aspect_score"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# Convert to numeric
# ============================================================

for col in required_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# Missing value check
# ============================================================

print("\nMissing values:")

missing = df[
    required_columns
].isna().sum()

print(missing)

if missing.sum() > 0:

    raise ValueError(
        "Quality dimensions contain missing values. "
        "Please check the input CSV."
    )


# ============================================================
# Range check
# ============================================================

print("\n" + "=" * 75)
print("ORIGINAL SCORE RANGE CHECK")
print("=" * 75)

for col in required_columns:

    print(
        f"{col:<30} "
        f"min={df[col].min():.6f}, "
        f"max={df[col].max():.6f}"
    )


# ============================================================
# Confidence score
# ============================================================

if HEDGE_IS_UNCERTAINTY:



    df["confidence_score"] = (
        1.0 - df["hedge_score"]
    )

else:

    df["confidence_score"] = (
        df["hedge_score"]
    )


# ============================================================
# Final 5 dimensions
# ============================================================

critic_data = pd.DataFrame({

    "Confidence":
        df["confidence_score"],

    "Constructiveness":
        df[CONSTRUCTIVE_COL],

    "Substantiation":
        df["substan_score"],

    "Kindness":
        df["politeness_score"],

    "Comprehensiveness":
        df["aspect_score"]
})


# ============================================================
# Check final range
# ============================================================

print("\n")
print("=" * 75)
print("FINAL FIVE-DIMENSION RANGE")
print("=" * 75)

for col in critic_data.columns:

    print(
        f"{col:<20} "
        f"min={critic_data[col].min():.6f}, "
        f"max={critic_data[col].max():.6f}"
    )


# ============================================================
# Descriptive statistics
# ============================================================

dimension_statistics = pd.DataFrame({

    "Mean":
        critic_data.mean(),

    "Std":
        critic_data.std(),

    "Variance":
        critic_data.var(),

    "Min":
        critic_data.min(),

    "Median":
        critic_data.median(),

    "Max":
        critic_data.max()
})

dimension_statistics["CV"] = (
    dimension_statistics["Std"]
    / dimension_statistics["Mean"]
)


print("\n")
print("=" * 75)
print("DIMENSION STATISTICS")
print("=" * 75)

print(
    dimension_statistics.round(6)
)


# ============================================================
# Calculate CRITIC weights
# ============================================================

(
    weights,
    normalized_data,
    normalized_std,
    conflict,
    information,
    correlation
) = critic_weight_method(
    critic_data,
    eps=EPS
)


# ============================================================
# Correlation matrix
# ============================================================

print("\n")
print("=" * 75)
print("DIMENSION CORRELATION MATRIX")
print("=" * 75)

print(
    correlation.round(6)
)


# ============================================================
# Weight table
# ============================================================

weight_table = pd.DataFrame({

    # 原始维度统计
    "Mean":
        critic_data.mean(),

    "Std":
        critic_data.std(),

    "Variance":
        critic_data.var(),

    "CV":
        critic_data.std()
        / critic_data.mean(),

    # CRITIC 中真正使用的统计量
    "Normalized_Std":
        normalized_std,

    "Conflict":
        conflict,

    "Information":
        information,

    "CRITIC_Weight":
        weights
})


print("\n")
print("=" * 75)
print("FINAL CRITIC WEIGHTS")
print("=" * 75)

print(
    weight_table.round(6)
)


print("\nCRITIC weights:")

for dimension, weight in weights.items():

    print(
        f"{dimension:<20}: "
        f"{weight:.9f}"
    )


print(
    f"\nWeight sum: "
    f"{weights.sum():.12f}"
)


# ============================================================
# Weight order
# ============================================================

print("\n")
print("=" * 75)
print("WEIGHT ORDER")
print("=" * 75)

print(
    weights.sort_values(
        ascending=False
    )
)


# ============================================================
# Save normalized dimension scores
# ============================================================



df["confidence_score_norm"] = (
    normalized_data["Confidence"]
)

df["constructive_score_norm"] = (
    normalized_data["Constructiveness"]
)

df["substan_score_norm"] = (
    normalized_data["Substantiation"]
)

df["politeness_score_norm"] = (
    normalized_data["Kindness"]
)

df["aspect_score_norm"] = (
    normalized_data["Comprehensiveness"]
)


# ============================================================
# Calculate final CRITIC quality score
# ============================================================



df["quality_score"] = (
    normalized_data
    .mul(
        weights,
        axis=1
    )
    .sum(axis=1)
)


# ============================================================
# Quality score distribution
# ============================================================

quality = df["quality_score"]

quality_mean = quality.mean()

quality_std = quality.std()

quality_median = quality.median()

quality_q1 = quality.quantile(0.25)

quality_q3 = quality.quantile(0.75)

quality_iqr = (
    quality_q3
    - quality_q1
)


print("\n")
print("=" * 75)
print("FINAL CRITIC QUALITY SCORE DISTRIBUTION")
print("=" * 75)

print(
    quality.describe(
        percentiles=[
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)


print("\nKey statistics:")

print(
    f"Mean:                "
    f"{quality_mean:.6f}"
)

print(
    f"Std:                 "
    f"{quality_std:.6f}"
)

print(
    f"Median:              "
    f"{quality_median:.6f}"
)

print(
    f"Q1:                  "
    f"{quality_q1:.6f}"
)

print(
    f"Q3:                  "
    f"{quality_q3:.6f}"
)

print(
    f"IQR:                 "
    f"{quality_iqr:.6f}"
)

print(
    f"Min:                 "
    f"{quality.min():.6f}"
)

print(
    f"Max:                 "
    f"{quality.max():.6f}"
)


# ============================================================
# Save CRITIC weights
# ============================================================

weight_table.to_csv(
    WEIGHT_OUTPUT_CSV,
    encoding="utf-8-sig"
)


# ============================================================
# Save dimension correlation matrix
# ============================================================

correlation.to_csv(
    CORRELATION_OUTPUT_CSV,
    encoding="utf-8-sig"
)


# ============================================================
# Save final master CSV
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Final summary
# ============================================================

print("\n")
print("=" * 75)
print("FILES SAVED")
print("=" * 75)

print(
    f"CRITIC weights: "
    f"{WEIGHT_OUTPUT_CSV}"
)

print(
    f"Dimension correlation: "
    f"{CORRELATION_OUTPUT_CSV}"
)

print(
    f"Final master CSV: "
    f"{OUTPUT_CSV}"
)

print("\nFinished.")
