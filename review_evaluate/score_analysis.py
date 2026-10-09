import pandas as pd
import numpy as np

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


# ============================================================
# Config
# ============================================================

INPUT_CSV = "all_reviews_quality_critic_final_with_venue_year.csv"

OUTPUT_CSV = "iclr_critic_consecutive_year_comparisons.csv"

QUALITY_COL = "quality_score"


# ============================================================
# 1. Load data
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("=" * 90)
print("LOAD DATA")
print("=" * 90)

print(f"Total reviews: {len(df):,}")


# ============================================================
# 2. Ensure decision_binary exists
# ============================================================

if "decision_binary" not in df.columns:

    df["decision_binary"] = np.where(
        df["decision"]
        .astype(str)
        .str.lower()
        .str.contains(
            "accept",
            na=False
        ),
        "Accept",
        "Reject"
    )


# ============================================================
# 3. Ensure numeric
# ============================================================

df["year"] = pd.to_numeric(
    df["year"],
    errors="coerce"
)

df[QUALITY_COL] = pd.to_numeric(
    df[QUALITY_COL],
    errors="coerce"
)


# ============================================================
# 4. ICLR 2017-2025 only
# ============================================================

iclr = df[
    (df["venue"] == "ICLR")
    &
    (df["year"].between(
        2017,
        2025
    ))
].copy()


iclr["year"] = (
    iclr["year"]
    .astype(int)
)


years = list(
    range(
        2017,
        2026
    )
)


# ============================================================
# 5. Consecutive-year comparisons
# ============================================================

results = []


for decision in [
    "Accept",
    "Reject"
]:

    temp = iclr[
        iclr["decision_binary"]
        == decision
    ].copy()


    for year_1, year_2 in zip(
        years[:-1],
        years[1:]
    ):

        x = (
            temp.loc[
                temp["year"] == year_1,
                QUALITY_COL
            ]
            .dropna()
            .values
        )

        y = (
            temp.loc[
                temp["year"] == year_2,
                QUALITY_COL
            ]
            .dropna()
            .values
        )


        # ----------------------------------------------------
        # Mann-Whitney U
        #
        # y = later year
        # x = earlier year
        #
        # positive r_rb:
        # later year > earlier year
        # ----------------------------------------------------

        U, p = mannwhitneyu(
            y,
            x,
            alternative="two-sided"
        )


        rank_biserial = (
            2 * U
            /
            (
                len(y)
                *
                len(x)
            )
            - 1
        )


        mean_1 = np.mean(x)
        mean_2 = np.mean(y)

        median_1 = np.median(x)
        median_2 = np.median(y)


        mean_difference = (
            mean_2
            -
            mean_1
        )


        if mean_difference > 0:

            direction = "Later > Earlier"

        elif mean_difference < 0:

            direction = "Earlier > Later"

        else:

            direction = "Equal"


        results.append({

            "decision":
                decision,

            "year_1":
                year_1,

            "year_2":
                year_2,

            "n_year_1":
                len(x),

            "n_year_2":
                len(y),

            "mean_year_1":
                mean_1,

            "mean_year_2":
                mean_2,

            "difference_y2_minus_y1":
                mean_difference,

            "median_year_1":
                median_1,

            "median_year_2":
                median_2,

            "mannwhitney_U":
                U,

            "p_value":
                p,

            "rank_biserial":
                rank_biserial,

            "direction":
                direction
        })


results_df = pd.DataFrame(
    results
)


# ============================================================
# 6. BH-FDR correction
#
# Accept and Reject separately
# Each family = 8 consecutive-year tests
# ============================================================

results_df["p_fdr"] = np.nan


for decision in [
    "Accept",
    "Reject"
]:

    mask = (
        results_df["decision"]
        == decision
    )


    pvals = (
        results_df.loc[
            mask,
            "p_value"
        ]
        .values
    )


    _, p_adj, _, _ = multipletests(
        pvals,
        alpha=0.05,
        method="fdr_bh"
    )


    results_df.loc[
        mask,
        "p_fdr"
    ] = p_adj


results_df["significant_fdr"] = (
    results_df["p_fdr"]
    < 0.05
)


# ============================================================
# 7. Effect-size labels
# ============================================================

def effect_label(r):

    if pd.isna(r):

        return "NA"

    a = abs(r)

    if a < 0.10:

        return "negligible"

    elif a < 0.30:

        return "small"

    elif a < 0.50:

        return "moderate"

    else:

        return "large"


results_df["effect_size"] = (
    results_df[
        "rank_biserial"
    ]
    .apply(
        effect_label
    )
)


# ============================================================
# 8. Sort
# ============================================================

results_df = (
    results_df
    .sort_values(
        [
            "decision",
            "year_1"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 9. Full results
# ============================================================

display_cols = [

    "decision",

    "year_1",
    "year_2",

    "n_year_1",
    "n_year_2",

    "mean_year_1",
    "mean_year_2",

    "difference_y2_minus_y1",

    "p_value",
    "p_fdr",
    "significant_fdr",

    "rank_biserial",
    "effect_size",

    "direction"
]


print("\n")
print("=" * 120)
print("ICLR CRITIC QUALITY: CONSECUTIVE-YEAR COMPARISONS")
print("=" * 120)


print(
    results_df[
        display_cols
    ]
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 10. Accepted summary
# ============================================================

accepted = results_df[
    results_df["decision"]
    == "Accept"
]


print("\n")
print("=" * 90)
print("ACCEPTED REVIEWS SUMMARY")
print("=" * 90)


print(
    "Later year higher:",
    (
        accepted[
            "difference_y2_minus_y1"
        ] > 0
    ).sum(),
    "/ 8"
)


print(
    "Earlier year higher:",
    (
        accepted[
            "difference_y2_minus_y1"
        ] < 0
    ).sum(),
    "/ 8"
)


print(
    "FDR significant:",
    accepted[
        "significant_fdr"
    ].sum(),
    "/ 8"
)


print(
    "Median |rank-biserial|:",
    accepted[
        "rank_biserial"
    ].abs().median()
)


print(
    "Max |rank-biserial|:",
    accepted[
        "rank_biserial"
    ].abs().max()
)


# ============================================================
# 11. Rejected summary
# ============================================================

rejected = results_df[
    results_df["decision"]
    == "Reject"
]


print("\n")
print("=" * 90)
print("REJECTED REVIEWS SUMMARY")
print("=" * 90)


print(
    "Later year higher:",
    (
        rejected[
            "difference_y2_minus_y1"
        ] > 0
    ).sum(),
    "/ 8"
)


print(
    "Earlier year higher:",
    (
        rejected[
            "difference_y2_minus_y1"
        ] < 0
    ).sum(),
    "/ 8"
)


print(
    "FDR significant:",
    rejected[
        "significant_fdr"
    ].sum(),
    "/ 8"
)


print(
    "Median |rank-biserial|:",
    rejected[
        "rank_biserial"
    ].abs().median()
)


print(
    "Max |rank-biserial|:",
    rejected[
        "rank_biserial"
    ].abs().max()
)


# ============================================================
# 12. Only significant comparisons
# ============================================================

print("\n")
print("=" * 100)
print("FDR-SIGNIFICANT CONSECUTIVE-YEAR COMPARISONS")
print("=" * 100)


significant = results_df[
    results_df[
        "significant_fdr"
    ]
].copy()


print(
    significant[
        [
            "decision",
            "year_1",
            "year_2",
            "mean_year_1",
            "mean_year_2",
            "difference_y2_minus_y1",
            "rank_biserial",
            "p_fdr",
            "effect_size",
            "direction"
        ]
    ]
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 13. Save
# ============================================================

results_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


print("\n")
print("=" * 90)
print("SAVED")
print("=" * 90)

print(OUTPUT_CSV)

print("\nFinished.")
