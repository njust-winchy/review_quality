import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr


# ============================================================
# Config
# ============================================================

INPUT_CSV = "all_reviews_quality_critic_final.csv"

OUTPUT_CSV = "critic_leave_one_dimension_out.csv"

WEIGHTS_OUTPUT_CSV = "critic_leave_one_dimension_out_weights.csv"

EPS = 1e-12


# ============================================================
# CRITIC Weight Method
# ============================================================

def critic_weight_method(data, eps=1e-12):
    """
    CRITIC weighting.

    Input:
        rows    = reviews
        columns = quality dimensions

    All dimensions must be positive indicators:
        higher = better

    Returns
    -------
    weights : pd.Series
    normalized_data : pd.DataFrame
    """

    X_raw = data.copy().astype(float)

    # --------------------------------------------------------
    # 1. Min-Max normalization
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 2. Standard deviation
    # --------------------------------------------------------

    std = X.std(
        axis=0,
        ddof=0
    )

    # --------------------------------------------------------
    # 3. Pearson correlation
    # --------------------------------------------------------

    corr = X.corr(
        method="pearson"
    ).fillna(0.0)

    np.fill_diagonal(
        corr.values,
        1.0
    )

    # --------------------------------------------------------
    # 4. Conflict
    # --------------------------------------------------------

    conflict = (
        1.0 - corr
    ).sum(axis=1)

    # --------------------------------------------------------
    # 5. Information quantity
    # --------------------------------------------------------

    information = (
        std * conflict
    )

    # --------------------------------------------------------
    # 6. Weights
    # --------------------------------------------------------

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

    return weights, X


# ============================================================
# Load data
# ============================================================

print("=" * 75)
print("Loading dataset")
print("=" * 75)

df = pd.read_csv(INPUT_CSV)

print(f"Number of reviews: {len(df):,}")


# ============================================================
# Five dimensions
# ============================================================

dimension_data = pd.DataFrame({

    "Confidence":
        df["confidence_score"],

    "Constructiveness":
        df["constructive_score_binary"],

    "Substantiation":
        df["substan_score"],

    "Kindness":
        df["politeness_score"],

    "Comprehensiveness":
        df["aspect_score"]
})


# ============================================================
# Missing value check
# ============================================================

if dimension_data.isna().sum().sum() > 0:

    print(dimension_data.isna().sum())

    raise ValueError(
        "Missing values found in quality dimensions."
    )


# ============================================================
# Full five-dimensional CRITIC score
# ============================================================

full_weights, full_normalized = critic_weight_method(
    dimension_data
)

full_score = (
    full_normalized
    .mul(
        full_weights,
        axis=1
    )
    .sum(axis=1)
)


print("\n")
print("=" * 75)
print("FULL FIVE-DIMENSION CRITIC WEIGHTS")
print("=" * 75)

for dimension, weight in full_weights.items():

    print(
        f"{dimension:<20}: "
        f"{weight:.9f}"
    )


# ============================================================
# Sanity check against saved quality_score
# ============================================================

if "quality_score" in df.columns:

    saved_score = pd.to_numeric(
        df["quality_score"],
        errors="coerce"
    )

    max_diff = (
        full_score - saved_score
    ).abs().max()

    print(
        "\nMax difference between recalculated "
        "and saved quality_score:"
    )

    print(
        f"{max_diff:.12f}"
    )


# ============================================================
# Leave-One-Dimension-Out analysis
# ============================================================

loo_results = []

loo_weights = []


for removed_dimension in dimension_data.columns:

    print("\n" + "=" * 75)
    print(
        f"REMOVING: {removed_dimension}"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Remove one dimension
    # --------------------------------------------------------

    remaining_data = dimension_data.drop(
        columns=[removed_dimension]
    )

    # --------------------------------------------------------
    # Re-estimate CRITIC weights
    # --------------------------------------------------------

    weights_loo, normalized_loo = (
        critic_weight_method(
            remaining_data
        )
    )

    # --------------------------------------------------------
    # New aggregate score
    # --------------------------------------------------------

    score_loo = (
        normalized_loo
        .mul(
            weights_loo,
            axis=1
        )
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # Pearson correlation
    # --------------------------------------------------------

    pearson_r, pearson_p = pearsonr(
        full_score,
        score_loo
    )

    # --------------------------------------------------------
    # Spearman correlation
    # --------------------------------------------------------

    spearman_rho, spearman_p = spearmanr(
        full_score,
        score_loo
    )

    # --------------------------------------------------------
    # Largest remaining weight
    # --------------------------------------------------------

    largest_dimension = (
        weights_loo.idxmax()
    )

    largest_weight = (
        weights_loo.max()
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    loo_results.append({

        "Removed":
            removed_dimension,

        "Pearson_r":
            pearson_r,

        "Pearson_p":
            pearson_p,

        "Spearman_rho":
            spearman_rho,

        "Spearman_p":
            spearman_p,

        "Largest_Remaining_Dimension":
            largest_dimension,

        "Largest_Remaining_Weight":
            largest_weight
    })

    # --------------------------------------------------------
    # Save all re-estimated weights
    # --------------------------------------------------------

    weight_row = {
        "Removed": removed_dimension
    }

    for dim in dimension_data.columns:

        if dim == removed_dimension:
            weight_row[dim] = np.nan
        else:
            weight_row[dim] = weights_loo[dim]

    loo_weights.append(
        weight_row
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\nRe-estimated weights:")

    print(
        weights_loo
        .sort_values(
            ascending=False
        )
        .round(6)
    )

    print(
        f"\nPearson r     = "
        f"{pearson_r:.6f}"
    )

    print(
        f"Spearman rho  = "
        f"{spearman_rho:.6f}"
    )

    print(
        f"Largest weight = "
        f"{largest_dimension} "
        f"({largest_weight:.6f})"
    )


# ============================================================
# Result tables
# ============================================================

loo_results_df = pd.DataFrame(
    loo_results
)

loo_weights_df = pd.DataFrame(
    loo_weights
)


# ============================================================
# Arrange in paper-friendly order
# ============================================================

paper_order = [
    "Confidence",
    "Constructiveness",
    "Substantiation",
    "Kindness",
    "Comprehensiveness"
]

loo_results_df["Removed"] = pd.Categorical(
    loo_results_df["Removed"],
    categories=paper_order,
    ordered=True
)

loo_results_df = (
    loo_results_df
    .sort_values("Removed")
    .reset_index(drop=True)
)


# ============================================================
# Print final table
# ============================================================

print("\n")
print("=" * 75)
print("FINAL LEAVE-ONE-DIMENSION-OUT RESULTS")
print("=" * 75)

print(
    loo_results_df[
        [
            "Removed",
            "Pearson_r",
            "Spearman_rho",
            "Largest_Remaining_Dimension",
            "Largest_Remaining_Weight"
        ]
    ].round(6)
)


# ============================================================
# Save
# ============================================================

loo_results_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

loo_weights_df.to_csv(
    WEIGHTS_OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


print("\n")
print("=" * 75)
print("FILES SAVED")
print("=" * 75)

print(
    f"LOO results: "
    f"{OUTPUT_CSV}"
)

print(
    f"LOO weights: "
    f"{WEIGHTS_OUTPUT_CSV}"
)
print(
    loo_results_df[
        [
            "Removed",
            "Pearson_r",
            "Spearman_rho",
            "Largest_Remaining_Dimension",
            "Largest_Remaining_Weight"
        ]
    ].round(6).to_string(index=False)
)
print("\nFinished.")