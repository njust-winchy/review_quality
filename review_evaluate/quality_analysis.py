# import pandas as pd
# import numpy as np
#
# from scipy.stats import mannwhitneyu
# from statsmodels.stats.multitest import multipletests
#
#
# # ============================================================
# # Config
# # ============================================================
#
# INPUT_CSV = "all_reviews_quality_critic_final.csv"
#
# OUTPUT_CSV = "iclr_accept_reject_critic_quality.csv"
#
# MIN_GROUP_N = 20
#
#
# # ============================================================
# # Load data
# # ============================================================
#
# df = pd.read_csv(INPUT_CSV)
#
# print("=" * 80)
# print("Loaded dataset")
# print("=" * 80)
#
# print(f"Reviews: {len(df):,}")
#
#
# # ============================================================
# # Parse venue and year
# # ============================================================
#
# def parse_source_file(source_file):
#
#     name = (
#         str(source_file)
#         .lower()
#         .replace(".csv", "")
#     )
#
#     # --------------------------------------------------------
#     # ICLR
#     # --------------------------------------------------------
#
#     if name.startswith("iclr_"):
#
#         year = int(
#             name.split("_")[1]
#         )
#
#         return "ICLR", year
#
#
#     # --------------------------------------------------------
#     # NeurIPS / NIPS
#     # --------------------------------------------------------
#
#     if name.startswith("nips_"):
#
#         year = int(
#             name.split("_")[1]
#         )
#
#         return "NeurIPS", year
#
#
#     # --------------------------------------------------------
#     # Other NLP venues
#     # --------------------------------------------------------
#
#     mapping = {
#
#         "acl_17":
#             ("ACL", 2017),
#
#         "arr_22":
#             ("ARR", 2022),
#
#         "coling_20":
#             ("COLING", 2020),
#
#         "conll_16":
#             ("CoNLL", 2016),
#
#         "emnlp_2023":
#             ("EMNLP", 2023),
#     }
#
#     if name in mapping:
#
#         return mapping[name]
#
#     return "Unknown", np.nan
#
#
# parsed = df[
#     "source_file"
# ].apply(
#     parse_source_file
# )
#
# df["venue"] = [
#     x[0]
#     for x in parsed
# ]
#
# df["year"] = [
#     x[1]
#     for x in parsed
# ]
#
#
# # ============================================================
# # Check venue / year
# # ============================================================
#
# print("\nVenue/year check:")
#
# print(
#     df[
#         [
#             "source_file",
#             "venue",
#             "year"
#         ]
#     ]
#     .drop_duplicates()
#     .sort_values(
#         [
#             "venue",
#             "year"
#         ]
#     )
#     .to_string(
#         index=False
#     )
# )
#
#
# # ============================================================
# # Decision binary
# # ============================================================
#
# # 当前 CRITIC master CSV 没有 decision_binary，
# # 所以重新生成
#
# df["decision_binary"] = np.where(
#     df["decision"]
#     .astype(str)
#     .str.lower()
#     .str.contains(
#         "accept",
#         na=False
#     ),
#     "Accept",
#     "Reject"
# )
#
#
# # ============================================================
# # Ensure numeric
# # ============================================================
#
# df["quality_score"] = pd.to_numeric(
#     df["quality_score"],
#     errors="coerce"
# )
#
#
# # ============================================================
# # ICLR only: 2017-2025
# # ============================================================
#
# iclr = df[
#     (df["venue"] == "ICLR")
#     &
#     (df["year"].between(
#         2017,
#         2025
#     ))
# ].copy()
#
#
# print("\n")
# print("=" * 80)
# print("ICLR SAMPLE")
# print("=" * 80)
#
# print(
#     pd.crosstab(
#         iclr["year"],
#         iclr["decision_binary"]
#     )
# )
#
#
# # ============================================================
# # Accept vs Reject by year
# # ============================================================
#
# results = []
#
#
# for year in sorted(
#     iclr["year"]
#     .dropna()
#     .unique()
# ):
#
#     sub = iclr[
#         iclr["year"] == year
#     ].copy()
#
#
#     accept = (
#         sub.loc[
#             sub["decision_binary"]
#             == "Accept",
#             "quality_score"
#         ]
#         .dropna()
#         .values
#     )
#
#
#     reject = (
#         sub.loc[
#             sub["decision_binary"]
#             == "Reject",
#             "quality_score"
#         ]
#         .dropna()
#         .values
#     )
#
#
#     n_accept = len(accept)
#     n_reject = len(reject)
#
#
#     # --------------------------------------------------------
#     # Descriptive statistics
#     # --------------------------------------------------------
#
#     accept_mean = (
#         np.mean(accept)
#         if n_accept > 0
#         else np.nan
#     )
#
#     reject_mean = (
#         np.mean(reject)
#         if n_reject > 0
#         else np.nan
#     )
#
#
#     accept_median = (
#         np.median(accept)
#         if n_accept > 0
#         else np.nan
#     )
#
#     reject_median = (
#         np.median(reject)
#         if n_reject > 0
#         else np.nan
#     )
#
#
#     accept_std = (
#         np.std(
#             accept,
#             ddof=1
#         )
#         if n_accept > 1
#         else np.nan
#     )
#
#     reject_std = (
#         np.std(
#             reject,
#             ddof=1
#         )
#         if n_reject > 1
#         else np.nan
#     )
#
#
#     mean_difference = (
#         reject_mean
#         -
#         accept_mean
#     )
#
#
#     # --------------------------------------------------------
#     # Mann-Whitney U
#     #
#     # Reject = x
#     # Accept = y
#     #
#     # positive rank-biserial:
#     # Reject > Accept
#     # --------------------------------------------------------
#
#     if (
#         n_accept >= MIN_GROUP_N
#         and
#         n_reject >= MIN_GROUP_N
#     ):
#
#         U, p = mannwhitneyu(
#             reject,
#             accept,
#             alternative="two-sided"
#         )
#
#
#         rank_biserial = (
#             2 * U
#             /
#             (
#                 n_reject
#                 *
#                 n_accept
#             )
#             - 1
#         )
#
#     else:
#
#         U = np.nan
#         p = np.nan
#         rank_biserial = np.nan
#
#
#     # --------------------------------------------------------
#     # Direction
#     # --------------------------------------------------------
#
#     if mean_difference > 0:
#
#         direction = "Reject > Accept"
#
#     elif mean_difference < 0:
#
#         direction = "Accept > Reject"
#
#     else:
#
#         direction = "Equal"
#
#
#     results.append({
#
#         "year":
#             int(year),
#
#         "n_accept":
#             n_accept,
#
#         "n_reject":
#             n_reject,
#
#         "accept_mean":
#             accept_mean,
#
#         "reject_mean":
#             reject_mean,
#
#         "mean_difference_R_minus_A":
#             mean_difference,
#
#         "accept_median":
#             accept_median,
#
#         "reject_median":
#             reject_median,
#
#         "accept_std":
#             accept_std,
#
#         "reject_std":
#             reject_std,
#
#         "mannwhitney_U":
#             U,
#
#         "p_value":
#             p,
#
#         "rank_biserial":
#             rank_biserial,
#
#         "direction":
#             direction
#     })
#
#
# results_df = pd.DataFrame(
#     results
# )
#
#
# # ============================================================
# # FDR correction
# #
# # IMPORTANT:
# # only the 9 ICLR yearly comparisons
# # ============================================================
#
# mask = (
#     results_df[
#         "p_value"
#     ].notna()
# )
#
# p_values = (
#     results_df.loc[
#         mask,
#         "p_value"
#     ].values
# )
#
#
# _, p_adjusted, _, _ = multipletests(
#     p_values,
#     alpha=0.05,
#     method="fdr_bh"
# )
#
#
# results_df.loc[
#     mask,
#     "p_fdr"
# ] = p_adjusted
#
#
# results_df["significant_fdr"] = (
#     results_df["p_fdr"]
#     < 0.05
# )
#
#
# # ============================================================
# # Effect size label
# # ============================================================
#
# def interpret_effect_size(r):
#
#     if pd.isna(r):
#
#         return "NA"
#
#     r_abs = abs(r)
#
#     if r_abs < 0.10:
#
#         return "negligible"
#
#     elif r_abs < 0.30:
#
#         return "small"
#
#     elif r_abs < 0.50:
#
#         return "moderate"
#
#     else:
#
#         return "large"
#
#
# results_df["effect_size"] = (
#     results_df[
#         "rank_biserial"
#     ].apply(
#         interpret_effect_size
#     )
# )
#
#
# # ============================================================
# # Display
# # ============================================================
#
# display_cols = [
#
#     "year",
#
#     "n_accept",
#     "n_reject",
#
#     "accept_mean",
#     "reject_mean",
#
#     "mean_difference_R_minus_A",
#
#     "accept_median",
#     "reject_median",
#
#     "p_value",
#     "p_fdr",
#
#     "significant_fdr",
#
#     "rank_biserial",
#     "effect_size",
#
#     "direction"
# ]
#
#
# print("\n")
# print("=" * 110)
# print(
#     "ICLR CRITIC QUALITY SCORE: "
#     "ACCEPT vs REJECT"
# )
# print("=" * 110)
#
#
# print(
#     results_df[
#         display_cols
#     ]
#     .round(6)
#     .to_string(
#         index=False
#     )
# )
#
#
# # ============================================================
# # Summary
# # ============================================================
#
# print("\n")
# print("=" * 80)
# print("SUMMARY")
# print("=" * 80)
#
#
# print(
#     "Reject > Accept:",
#     (
#         results_df[
#             "mean_difference_R_minus_A"
#         ] > 0
#     ).sum(),
#     "/",
#     len(results_df)
# )
#
#
# print(
#     "Accept > Reject:",
#     (
#         results_df[
#             "mean_difference_R_minus_A"
#         ] < 0
#     ).sum(),
#     "/",
#     len(results_df)
# )
#
#
# print(
#     "Significant after FDR:",
#     results_df[
#         "significant_fdr"
#     ].sum(),
#     "/",
#     len(results_df)
# )
#
#
# print(
#     "Significant Reject > Accept:",
#     (
#         results_df[
#             "significant_fdr"
#         ]
#         &
#         (
#             results_df[
#                 "mean_difference_R_minus_A"
#             ] > 0
#         )
#     ).sum(),
#     "/",
#     len(results_df)
# )
#
#
# print(
#     "Median rank-biserial:",
#     results_df[
#         "rank_biserial"
#     ].median()
# )
#
#
# # ============================================================
# # Save
# # ============================================================
#
# results_df.to_csv(
#     OUTPUT_CSV,
#     index=False,
#     encoding="utf-8-sig"
# )
#
#
# # 顺便保存以后一直使用的 master
# df.to_csv(
#     "all_reviews_quality_critic_final_with_venue_year.csv",
#     index=False,
#     encoding="utf-8-sig"
# )
#
#
# print("\nSaved:")
# print(
#     OUTPUT_CSV
# )
#
# print(
#     "all_reviews_quality_critic_final_with_venue_year.csv"
# )


#
# # step 2
#
# import pandas as pd
# import numpy as np
#
# from scipy.stats import kruskal
#
#
# # ============================================================
# # Config
# # ============================================================
#
# INPUT_CSV = "all_reviews_quality_critic_final_with_venue_year.csv"
#
# OUTPUT_YEARLY = "iclr_critic_quality_yearly_by_decision.csv"
# OUTPUT_KW = "iclr_critic_quality_kruskal_by_decision.csv"
#
# QUALITY_COL = "quality_score"
#
#
# # ============================================================
# # 1. Load data
# # ============================================================
#
# df = pd.read_csv(INPUT_CSV)
#
# print("=" * 90)
# print("LOAD DATA")
# print("=" * 90)
#
# print(f"Total reviews: {len(df):,}")
#
#
# # ============================================================
# # 2. Make sure decision_binary exists
# # ============================================================
#
# if "decision_binary" not in df.columns:
#
#     df["decision_binary"] = np.where(
#         df["decision"]
#         .astype(str)
#         .str.lower()
#         .str.contains(
#             "accept",
#             na=False
#         ),
#         "Accept",
#         "Reject"
#     )
#
#
# # ============================================================
# # 3. Ensure numeric
# # ============================================================
#
# df["year"] = pd.to_numeric(
#     df["year"],
#     errors="coerce"
# )
#
# df[QUALITY_COL] = pd.to_numeric(
#     df[QUALITY_COL],
#     errors="coerce"
# )
#
#
# # ============================================================
# # 4. ICLR 2017-2025 only
# # ============================================================
#
# iclr = df[
#     (df["venue"] == "ICLR")
#     &
#     (df["year"].between(
#         2017,
#         2025
#     ))
# ].copy()
#
#
# iclr["year"] = (
#     iclr["year"]
#     .astype(int)
# )
#
#
# print("\n")
# print("=" * 90)
# print("ICLR SAMPLE SIZE")
# print("=" * 90)
#
# print(
#     pd.crosstab(
#         iclr["year"],
#         iclr["decision_binary"]
#     )
# )
#
#
# # ============================================================
# # 5. Yearly descriptive statistics
# # ============================================================
#
# yearly = (
#     iclr
#     .groupby(
#         [
#             "year",
#             "decision_binary"
#         ]
#     )[QUALITY_COL]
#     .agg(
#         n="count",
#         mean="mean",
#         std="std",
#         median="median",
#         q1=lambda x: x.quantile(0.25),
#         q3=lambda x: x.quantile(0.75)
#     )
#     .reset_index()
# )
#
#
# yearly["iqr"] = (
#     yearly["q3"]
#     -
#     yearly["q1"]
# )
#
#
# print("\n")
# print("=" * 90)
# print("ICLR YEARLY CRITIC QUALITY BY DECISION")
# print("=" * 90)
#
# print(
#     yearly
#     .round(6)
#     .to_string(
#         index=False
#     )
# )
#
#
# # ============================================================
# # 6. Pivot table for yearly means
# # ============================================================
#
# mean_table = yearly.pivot(
#     index="year",
#     columns="decision_binary",
#     values="mean"
# )
#
#
# print("\n")
# print("=" * 90)
# print("MEAN CRITIC QUALITY SCORE")
# print("=" * 90)
#
# print(
#     mean_table
#     .round(6)
#     .to_string()
# )
#
#
# # ============================================================
# # 7. Kruskal-Wallis across years
# #
# # Accept and Reject tested separately
# # ============================================================
#
# kw_results = []
#
#
# print("\n")
# print("=" * 90)
# print("KRUSKAL-WALLIS ACROSS YEARS")
# print("=" * 90)
#
#
# for decision in [
#     "Accept",
#     "Reject"
# ]:
#
#     temp = iclr[
#         iclr["decision_binary"]
#         == decision
#     ].copy()
#
#
#     years = sorted(
#         temp["year"]
#         .dropna()
#         .unique()
#     )
#
#
#     groups = []
#
#     group_sizes = []
#
#
#     for year in years:
#
#         values = (
#             temp.loc[
#                 temp["year"] == year,
#                 QUALITY_COL
#             ]
#             .dropna()
#             .values
#         )
#
#         groups.append(
#             values
#         )
#
#         group_sizes.append(
#             len(values)
#         )
#
#
#     # --------------------------------------------------------
#     # Kruskal-Wallis
#     # --------------------------------------------------------
#
#     H, p = kruskal(
#         *groups
#     )
#
#
#     kw_results.append({
#
#         "decision":
#             decision,
#
#         "n_years":
#             len(years),
#
#         "total_n":
#             sum(group_sizes),
#
#         "H":
#             H,
#
#         "p_value":
#             p
#     })
#
#
#     print(
#         f"{decision:<10} "
#         f"H = {H:.6f}, "
#         f"p = {p:.10g}, "
#         f"N = {sum(group_sizes):,}"
#     )
#
#
# # ============================================================
# # 8. Save Kruskal results
# # ============================================================
#
# kw_df = pd.DataFrame(
#     kw_results
# )
#
#
# print("\n")
# print("=" * 90)
# print("KRUSKAL-WALLIS SUMMARY")
# print("=" * 90)
#
# print(
#     kw_df
#     .round(6)
#     .to_string(
#         index=False
#     )
# )
#
#
# # ============================================================
# # 9. Save files
# # ============================================================
#
# yearly.to_csv(
#     OUTPUT_YEARLY,
#     index=False,
#     encoding="utf-8-sig"
# )
#
# kw_df.to_csv(
#     OUTPUT_KW,
#     index=False,
#     encoding="utf-8-sig"
# )
#
#
# print("\n")
# print("=" * 90)
# print("FILES SAVED")
# print("=" * 90)
#
# print(
#     OUTPUT_YEARLY
# )
#
# print(
#     OUTPUT_KW
# )
#
# print("\nFinished.")

# #step 3
#
# import pandas as pd
# import numpy as np
#
# from scipy.stats import mannwhitneyu, kruskal
# from statsmodels.stats.multitest import multipletests
#
#
# # ============================================================
# # Config
# # ============================================================
#
# INPUT_CSV = "all_reviews_quality_critic_final_with_venue_year.csv"
#
# OUTPUT_MASTER = "all_reviews_quality_critic_wo_substan.csv"
# OUTPUT_ACCEPT = "iclr_accept_reject_critic_wo_substan.csv"
# OUTPUT_KW = "iclr_year_critic_wo_substan_kruskal.csv"
#
# EPS = 1e-12
#
#
# # ============================================================
# # CRITIC
# # ============================================================
#
# def critic_weight_method(data, eps=1e-12):
#
#     X_raw = data.copy().astype(float)
#
#     # --------------------------------------------------------
#     # Min-Max normalization
#     # --------------------------------------------------------
#
#     X = pd.DataFrame(
#         index=X_raw.index,
#         columns=X_raw.columns,
#         dtype=float
#     )
#
#     for col in X_raw.columns:
#
#         col_min = X_raw[col].min()
#         col_max = X_raw[col].max()
#
#         denominator = col_max - col_min
#
#         if abs(denominator) < eps:
#             X[col] = 0.0
#         else:
#             X[col] = (
#                 X_raw[col] - col_min
#             ) / denominator
#
#     # --------------------------------------------------------
#     # Standard deviation
#     # --------------------------------------------------------
#
#     std = X.std(
#         axis=0,
#         ddof=0
#     )
#
#     # --------------------------------------------------------
#     # Correlation
#     # --------------------------------------------------------
#
#     corr = X.corr(
#         method="pearson"
#     ).fillna(0.0)
#
#     np.fill_diagonal(
#         corr.values,
#         1.0
#     )
#
#     # --------------------------------------------------------
#     # Conflict
#     # --------------------------------------------------------
#
#     conflict = (
#         1.0 - corr
#     ).sum(axis=1)
#
#     # --------------------------------------------------------
#     # Information
#     # --------------------------------------------------------
#
#     information = (
#         std * conflict
#     )
#
#     # --------------------------------------------------------
#     # Weights
#     # --------------------------------------------------------
#
#     weights = (
#         information
#         / information.sum()
#     )
#
#     return (
#         weights,
#         X,
#         std,
#         conflict,
#         information,
#         corr
#     )
#
#
# # ============================================================
# # Load
# # ============================================================
#
# df = pd.read_csv(
#     INPUT_CSV
# )
#
# print("=" * 90)
# print("LOAD DATA")
# print("=" * 90)
#
# print(
#     f"Reviews: {len(df):,}"
# )
#
#
# # ============================================================
# # Ensure decision_binary
# # ============================================================
#
# if "decision_binary" not in df.columns:
#
#     df["decision_binary"] = np.where(
#         df["decision"]
#         .astype(str)
#         .str.lower()
#         .str.contains(
#             "accept",
#             na=False
#         ),
#         "Accept",
#         "Reject"
#     )
#
#
# # ============================================================
# # Four dimensions excluding Substantiation
# # ============================================================
#
# four_dim = pd.DataFrame({
#
#     "Confidence":
#         df["confidence_score"],
#
#     "Constructiveness":
#         df["constructive_score_binary"],
#
#     "Kindness":
#         df["politeness_score"],
#
#     "Comprehensiveness":
#         df["aspect_score"]
# })
#
#
# # ============================================================
# # Re-estimate CRITIC weights
# # ============================================================
#
# (
#     weights,
#     normalized,
#     std,
#     conflict,
#     information,
#     corr
# ) = critic_weight_method(
#     four_dim
# )
#
#
# print("\n")
# print("=" * 90)
# print("CRITIC WEIGHTS WITHOUT SUBSTANTIATION")
# print("=" * 90)
#
# for dim, w in weights.items():
#
#     print(
#         f"{dim:<20}: "
#         f"{w:.9f}"
#     )
#
#
# print(
#     f"\nWeight sum: "
#     f"{weights.sum():.12f}"
# )
#
#
# # ============================================================
# # New aggregate score without Substantiation
# # ============================================================
#
# df["quality_score_wo_substan"] = (
#     normalized
#     .mul(
#         weights,
#         axis=1
#     )
#     .sum(axis=1)
# )
#
#
# print("\n")
# print("=" * 90)
# print("QUALITY SCORE WITHOUT SUBSTANTIATION")
# print("=" * 90)
#
# print(
#     df[
#         "quality_score_wo_substan"
#     ]
#     .describe()
# )
#
#
# # ============================================================
# # ICLR 2017-2025
# # ============================================================
#
# iclr = df[
#     (df["venue"] == "ICLR")
#     &
#     (
#         pd.to_numeric(
#             df["year"],
#             errors="coerce"
#         ).between(
#             2017,
#             2025
#         )
#     )
# ].copy()
#
#
# iclr["year"] = pd.to_numeric(
#     iclr["year"],
#     errors="coerce"
# ).astype(int)
#
#
# # ============================================================
# # Accept vs Reject by year
# # ============================================================
#
# results = []
#
#
# for year in sorted(
#     iclr["year"].unique()
# ):
#
#     temp = iclr[
#         iclr["year"] == year
#     ]
#
#     accept = (
#         temp.loc[
#             temp["decision_binary"] == "Accept",
#             "quality_score_wo_substan"
#         ]
#         .dropna()
#         .values
#     )
#
#     reject = (
#         temp.loc[
#             temp["decision_binary"] == "Reject",
#             "quality_score_wo_substan"
#         ]
#         .dropna()
#         .values
#     )
#
#
#     U, p = mannwhitneyu(
#         reject,
#         accept,
#         alternative="two-sided"
#     )
#
#
#     r_rb = (
#         2 * U
#         / (
#             len(reject)
#             * len(accept)
#         )
#         - 1
#     )
#
#
#     results.append({
#
#         "year":
#             year,
#
#         "n_accept":
#             len(accept),
#
#         "n_reject":
#             len(reject),
#
#         "accept_mean":
#             np.mean(accept),
#
#         "reject_mean":
#             np.mean(reject),
#
#         "difference_R_minus_A":
#             np.mean(reject)
#             -
#             np.mean(accept),
#
#         "accept_median":
#             np.median(accept),
#
#         "reject_median":
#             np.median(reject),
#
#         "p_value":
#             p,
#
#         "rank_biserial":
#             r_rb
#     })
#
#
# results_df = pd.DataFrame(
#     results
# )
#
#
# # ============================================================
# # FDR across 9 ICLR yearly comparisons
# # ============================================================
#
# _, p_fdr, _, _ = multipletests(
#     results_df["p_value"],
#     alpha=0.05,
#     method="fdr_bh"
# )
#
# results_df["p_fdr"] = p_fdr
#
# results_df["significant_fdr"] = (
#     results_df["p_fdr"]
#     < 0.05
# )
#
#
# results_df["direction"] = np.where(
#     results_df[
#         "difference_R_minus_A"
#     ] > 0,
#     "Reject > Accept",
#     "Accept > Reject"
# )
#
#
# # ============================================================
# # Print Accept vs Reject
# # ============================================================
#
# print("\n")
# print("=" * 110)
# print("ICLR ACCEPT vs REJECT WITHOUT SUBSTANTIATION")
# print("=" * 110)
#
# print(
#     results_df[
#         [
#             "year",
#             "n_accept",
#             "n_reject",
#             "accept_mean",
#             "reject_mean",
#             "difference_R_minus_A",
#             "p_value",
#             "p_fdr",
#             "significant_fdr",
#             "rank_biserial",
#             "direction"
#         ]
#     ]
#     .round(6)
#     .to_string(
#         index=False
#     )
# )
#
#
# # ============================================================
# # Summary
# # ============================================================
#
# print("\n")
# print("=" * 90)
# print("ACCEPTANCE SUMMARY WITHOUT SUBSTANTIATION")
# print("=" * 90)
#
# print(
#     "Reject > Accept:",
#     (
#         results_df[
#             "difference_R_minus_A"
#         ] > 0
#     ).sum(),
#     "/",
#     len(results_df)
# )
#
# print(
#     "FDR significant:",
#     results_df[
#         "significant_fdr"
#     ].sum(),
#     "/",
#     len(results_df)
# )
#
# print(
#     "Significant Reject > Accept:",
#     (
#         results_df[
#             "significant_fdr"
#         ]
#         &
#         (
#             results_df[
#                 "difference_R_minus_A"
#             ] > 0
#         )
#     ).sum(),
#     "/",
#     len(results_df)
# )
#
# print(
#     "Median rank-biserial:",
#     results_df[
#         "rank_biserial"
#     ].median()
# )
#
# print(
#     "Max |rank-biserial|:",
#     results_df[
#         "rank_biserial"
#     ].abs().max()
# )
#
#
# # ============================================================
# # Kruskal-Wallis across years
# # ============================================================
#
# kw_results = []
#
#
# print("\n")
# print("=" * 90)
# print("KRUSKAL-WALLIS WITHOUT SUBSTANTIATION")
# print("=" * 90)
#
#
# for decision in [
#     "Accept",
#     "Reject"
# ]:
#
#     temp = iclr[
#         iclr[
#             "decision_binary"
#         ] == decision
#     ]
#
#     groups = []
#
#     years = sorted(
#         temp[
#             "year"
#         ].unique()
#     )
#
#     for year in years:
#
#         values = (
#             temp.loc[
#                 temp["year"] == year,
#                 "quality_score_wo_substan"
#             ]
#             .dropna()
#             .values
#         )
#
#         groups.append(
#             values
#         )
#
#
#     H, p = kruskal(
#         *groups
#     )
#
#
#     kw_results.append({
#
#         "decision":
#             decision,
#
#         "H":
#             H,
#
#         "p_value":
#             p
#     })
#
#
#     print(
#         f"{decision:<10} "
#         f"H = {H:.6f}, "
#         f"p = {p:.10g}"
#     )
#
#
# kw_df = pd.DataFrame(
#     kw_results
# )
#
#
# # ============================================================
# # Save
# # ============================================================
#
# df.to_csv(
#     OUTPUT_MASTER,
#     index=False,
#     encoding="utf-8-sig"
# )
#
# results_df.to_csv(
#     OUTPUT_ACCEPT,
#     index=False,
#     encoding="utf-8-sig"
# )
#
# kw_df.to_csv(
#     OUTPUT_KW,
#     index=False,
#     encoding="utf-8-sig"
# )
#
#
# print("\n")
# print("=" * 90)
# print("FILES SAVED")
# print("=" * 90)
#
# print(
#     OUTPUT_MASTER
# )
#
# print(
#     OUTPUT_ACCEPT
# )
#
# print(
#     OUTPUT_KW
# )
#
# print("\nFinished.")

#step 4
import pandas as pd
import numpy as np

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


# ============================================================
# Config
# ============================================================

INPUT_CSV = "all_reviews_quality_critic_final_with_venue_year.csv"

OUTPUT_CSV = "iclr_vs_neurips_critic_stratified_2021_2024.csv"

QUALITY_COL = "quality_score"

COMMON_YEARS = [
    2021,
    2022,
    2023,
    2024
]

MIN_GROUP_N = 20


# ============================================================
# 1. Load data
# ============================================================

df = pd.read_csv(
    INPUT_CSV
)

print("=" * 100)
print("LOAD DATA")
print("=" * 100)

print(
    f"Total reviews: {len(df):,}"
)


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
# 4. Keep ICLR / NeurIPS, common years only
# ============================================================

data = df[
    (
        df["venue"]
        .isin(
            [
                "ICLR",
                "NeurIPS"
            ]
        )
    )
    &
    (
        df["year"]
        .isin(
            COMMON_YEARS
        )
    )
].copy()


print("\n")
print("=" * 100)
print("SAMPLE SIZE CHECK")
print("=" * 100)

sample_table = pd.crosstab(
    [
        data["year"],
        data["decision_binary"]
    ],
    data["venue"]
)

print(
    sample_table
    .to_string()
)


# ============================================================
# 5. Cross-venue comparison
#
# Positive rank-biserial:
# NeurIPS > ICLR
#
# Negative rank-biserial:
# ICLR > NeurIPS
# ============================================================

results = []


for year in COMMON_YEARS:

    for decision in [
        "Accept",
        "Reject"
    ]:

        temp = data[
            (
                data["year"]
                == year
            )
            &
            (
                data["decision_binary"]
                == decision
            )
        ]


        # ----------------------------------------------------
        # ICLR
        # ----------------------------------------------------

        iclr = (
            temp.loc[
                temp["venue"]
                == "ICLR",
                QUALITY_COL
            ]
            .dropna()
            .values
        )


        # ----------------------------------------------------
        # NeurIPS
        # ----------------------------------------------------

        neurips = (
            temp.loc[
                temp["venue"]
                == "NeurIPS",
                QUALITY_COL
            ]
            .dropna()
            .values
        )


        n_iclr = len(iclr)
        n_neurips = len(neurips)


        if (
            n_iclr == 0
            or
            n_neurips == 0
        ):

            continue


        # ----------------------------------------------------
        # Descriptive statistics
        # ----------------------------------------------------

        iclr_mean = np.mean(
            iclr
        )

        neurips_mean = np.mean(
            neurips
        )


        iclr_median = np.median(
            iclr
        )

        neurips_median = np.median(
            neurips
        )


        iclr_std = (
            np.std(
                iclr,
                ddof=1
            )
            if n_iclr > 1
            else np.nan
        )

        neurips_std = (
            np.std(
                neurips,
                ddof=1
            )
            if n_neurips > 1
            else np.nan
        )


        difference = (
            neurips_mean
            -
            iclr_mean
        )


        # ----------------------------------------------------
        # Mann-Whitney U
        # ----------------------------------------------------

        if (
            n_iclr >= MIN_GROUP_N
            and
            n_neurips >= MIN_GROUP_N
        ):

            U, p = mannwhitneyu(
                neurips,
                iclr,
                alternative="two-sided"
            )


            # -----------------------------------------------
            # Rank-biserial correlation
            #
            # positive:
            # NeurIPS > ICLR
            #
            # negative:
            # ICLR > NeurIPS
            # -----------------------------------------------

            rank_biserial = (
                2 * U
                /
                (
                    n_neurips
                    *
                    n_iclr
                )
                - 1
            )


            tested = True

        else:

            U = np.nan
            p = np.nan
            rank_biserial = np.nan
            tested = False


        # ----------------------------------------------------
        # Direction
        # ----------------------------------------------------

        if difference > 0:

            direction = (
                "NeurIPS > ICLR"
            )

        elif difference < 0:

            direction = (
                "ICLR > NeurIPS"
            )

        else:

            direction = "Equal"


        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        results.append({

            "year":
                year,

            "decision":
                decision,

            "n_iclr":
                n_iclr,

            "n_neurips":
                n_neurips,

            "iclr_mean":
                iclr_mean,

            "neurips_mean":
                neurips_mean,

            "difference_NeurIPS_minus_ICLR":
                difference,

            "iclr_median":
                iclr_median,

            "neurips_median":
                neurips_median,

            "iclr_std":
                iclr_std,

            "neurips_std":
                neurips_std,

            "mannwhitney_U":
                U,

            "p_value":
                p,

            "rank_biserial":
                rank_biserial,

            "direction":
                direction,

            "tested":
                tested
        })


result_df = pd.DataFrame(
    results
)


# ============================================================
# 6. BH-FDR correction
#
# Accept / Reject separately
#
# Each family contains four yearly comparisons:
# 2021, 2022, 2023, 2024
# ============================================================

result_df["p_fdr"] = np.nan


for decision in [
    "Accept",
    "Reject"
]:

    mask = (
        (result_df["decision"]
         == decision)
        &
        result_df[
            "p_value"
        ].notna()
    )


    pvals = (
        result_df.loc[
            mask,
            "p_value"
        ]
        .values
    )


    if len(pvals) > 0:

        _, p_adj, _, _ = multipletests(
            pvals,
            alpha=0.05,
            method="fdr_bh"
        )


        result_df.loc[
            mask,
            "p_fdr"
        ] = p_adj


result_df[
    "significant_fdr"
] = (
    result_df[
        "p_fdr"
    ] < 0.05
)


# ============================================================
# 7. Effect size labels
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


result_df[
    "effect_size"
] = (
    result_df[
        "rank_biserial"
    ]
    .apply(
        effect_label
    )
)


# ============================================================
# 8. Sort
# ============================================================

result_df = (
    result_df
    .sort_values(
        [
            "decision",
            "year"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 9. Display full results
# ============================================================

display_cols = [

    "year",
    "decision",

    "n_iclr",
    "n_neurips",

    "iclr_mean",
    "neurips_mean",

    "difference_NeurIPS_minus_ICLR",

    "iclr_median",
    "neurips_median",

    "p_value",
    "p_fdr",
    "significant_fdr",

    "rank_biserial",
    "effect_size",

    "direction"
]


print("\n")
print("=" * 120)
print(
    "ICLR vs NeurIPS — CRITIC QUALITY SCORE "
    "(STRATIFIED BY ACCEPTANCE)"
)
print("=" * 120)


print(
    result_df[
        display_cols
    ]
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 10. Summary: Accepted
# ============================================================

print("\n")
print("=" * 100)
print("ACCEPTED REVIEWS SUMMARY")
print("=" * 100)


accepted = result_df[
    result_df["decision"]
    == "Accept"
].copy()


print(
    "NeurIPS > ICLR:",
    (
        accepted[
            "difference_NeurIPS_minus_ICLR"
        ] > 0
    ).sum(),
    "/",
    len(accepted)
)


print(
    "ICLR > NeurIPS:",
    (
        accepted[
            "difference_NeurIPS_minus_ICLR"
        ] < 0
    ).sum(),
    "/",
    len(accepted)
)


print(
    "FDR significant:",
    accepted[
        "significant_fdr"
    ].sum(),
    "/",
    len(accepted)
)


print(
    "Median rank-biserial:",
    accepted[
        "rank_biserial"
    ].median()
)


print(
    "Max |rank-biserial|:",
    accepted[
        "rank_biserial"
    ].abs().max()
)


# ============================================================
# 11. Summary: Rejected
# ============================================================

print("\n")
print("=" * 100)
print("REJECTED REVIEWS SUMMARY")
print("=" * 100)


rejected = result_df[
    result_df["decision"]
    == "Reject"
].copy()


print(
    "NeurIPS > ICLR:",
    (
        rejected[
            "difference_NeurIPS_minus_ICLR"
        ] > 0
    ).sum(),
    "/",
    len(rejected)
)


print(
    "ICLR > NeurIPS:",
    (
        rejected[
            "difference_NeurIPS_minus_ICLR"
        ] < 0
    ).sum(),
    "/",
    len(rejected)
)


print(
    "FDR significant:",
    rejected[
        "significant_fdr"
    ].sum(),
    "/",
    len(rejected)
)


print(
    "Median rank-biserial:",
    rejected[
        "rank_biserial"
    ].median()
)


print(
    "Max |rank-biserial|:",
    rejected[
        "rank_biserial"
    ].abs().max()
)


# ============================================================
# 12. Compact direction/effect summary
# ============================================================

print("\n")
print("=" * 100)
print("COMPACT SUMMARY")
print("=" * 100)


print(
    result_df[
        [
            "year",
            "decision",
            "difference_NeurIPS_minus_ICLR",
            "rank_biserial",
            "p_fdr",
            "significant_fdr",
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

result_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


print("\n")
print("=" * 100)
print("SAVED")
print("=" * 100)

print(
    OUTPUT_CSV
)

print("\nFinished.")
