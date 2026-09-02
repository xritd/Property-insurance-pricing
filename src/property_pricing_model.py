"""
Property Insurance Pricing: Frequency-Severity Model
======================================================
Wisconsin Local Government Property Insurance Fund (LGPIF), 2006-2010.

Builds a pure premium model (expected frequency x expected severity) using
a Negative Binomial GLM for claim frequency and a Gamma GLM for claim
severity, then evaluates the combined model against actual losses.

Data source:
https://raw.githubusercontent.com/OpenActTexts/LDACourse1/main/Data/PropertyFundInsample.csv
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps

DATA_URL = "https://raw.githubusercontent.com/OpenActTexts/LDACourse1/main/Data/PropertyFundInsample.csv"


# Load data

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL)
    return df


# Frequency model (Negative Binomial GLM)
#
# Final predictors: LnCoverage, NoClaimCredit, entity type dummies


FREQ_TYPE_DUMMIES = ["TypeCounty", 
                     "TypeMisc", 
                     "TypeSchool", 
                     "TypeTown", 
                     "TypeVillage"]

FREQ_PREDICTORS = ["LnCoverage", 
                   "NoClaimCredit"] + FREQ_TYPE_DUMMIES


def fit_frequency_model(df: pd.DataFrame):
    """
    Fit the final Negative Binomial frequency model on the full dataset.
    """
    X = sm.add_constant(df[FREQ_PREDICTORS])
    y = df["Freq"]

    # Negative Binomial MLE can converge to a degenerate
    # solution with default starting values when the data is
    # overdispersed. A method-of-moments starting guess for alpha avoids
    # that trap.
    mean_y, var_y = y.mean(), y.var()
    alpha_start = (var_y / mean_y - 1) / mean_y
    start_params = [np.log(mean_y)] + [0.0] * len(FREQ_PREDICTORS) + [alpha_start]

    model = sm.NegativeBinomial(y, X)
    result = model.fit(
        start_params=start_params, method="bfgs", maxiter=300, disp=0,
        cov_type="cluster", cov_kwds={"groups": df["PolicyNum"]},
    )
    return result


# Severity model (Gamma GLM)
#
# Fit only on policy-years with at least one claim (Freq > 0).
#
# Final predictor: Deduct, binned into 5 categories.


DEDUCT_BINS = [0, 1000, 2500, 5000, 10000, np.inf]
DEDUCT_LABELS = ["<=1000", 
                 "1000-2500", 
                 "2500-5000", 
                 "5000-10000", 
                 "10000+"]


def add_deduct_bin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DeductBin"] = pd.cut(df["Deduct"], 
                             bins=DEDUCT_BINS, 
                             labels=DEDUCT_LABELS)
    return df


def fit_severity_model(df: pd.DataFrame):
    """
    Fit the final Gamma GLM severity model on claims-only data.
    """
    claims_df = add_deduct_bin(df[df["Freq"] > 0])
    dummies = pd.get_dummies(claims_df["DeductBin"], 
                             prefix="Ded", 
                             drop_first=True).astype(float)
    X = sm.add_constant(dummies)
    y = claims_df["yAvg"]

    model = sm.GLM(
        y, X, family=sm.families.Gamma(link=sm.families.links.Log()),
        var_weights=claims_df["Freq"],
    )
    result = model.fit()
    return result, X.columns  


def deviance_f_test(deviance_reduced, df_reduced, deviance_full, df_full, dispersion):
    """F-test for comparing nested Gamma GLMs (more reliable than a raw LR test)."""
    f_stat = ((deviance_reduced - deviance_full) / (df_reduced - df_full)) / dispersion
    p_value = sps.f.sf(f_stat, dfn=(df_reduced - df_full), dfd=df_full)
    return f_stat, p_value


# Combine into pure premium


def predict_pure_premium(df: pd.DataFrame, freq_result, sev_result, sev_columns) -> pd.DataFrame:
    df = add_deduct_bin(df).copy()

    Xf = sm.add_constant(df[FREQ_PREDICTORS])
    df["pred_freq"] = freq_result.predict(Xf)

    dummies = pd.get_dummies(df["DeductBin"], prefix="Ded", drop_first=True).astype(float)
    Xs = sm.add_constant(dummies)
    Xs = Xs.reindex(columns=sev_columns, fill_value=0.0)  # align columns exactly
    df["pred_sev"] = sev_result.predict(Xs)

    df["pure_premium"] = df["pred_freq"] * df["pred_sev"]
    return df


# Evaluation: decile lift chart + Gini coefficient


def decile_lift_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["decile"] = pd.qcut(df["pure_premium"], 10, labels=False, duplicates="drop") + 1
    return df.groupby("decile").agg(
        n_policies=("PolicyNum", "count"),
        avg_predicted=("pure_premium", "mean"),
        avg_actual=("y", "mean"),
    ).reset_index()


def gini_coefficient(df: pd.DataFrame) -> float:
    """Gini coefficient measuring rank-ordering ability of predicted pure premium."""
    sorted_df = df.sort_values("pure_premium").reset_index(drop=True)
    cum_policies_pct = (np.arange(len(sorted_df)) + 1) / len(sorted_df)
    cum_actual_pct = sorted_df["y"].cumsum() / sorted_df["y"].sum()
    area_under_lorenz = np.trapezoid(cum_actual_pct, cum_policies_pct)
    return 1 - 2 * area_under_lorenz


# Run the full pipeline


def main():
    df = load_data()
    print(f"Loaded {len(df)} policy-year rows.\n")

    freq_result = fit_frequency_model(df)
    print("=== Frequency model (Negative Binomial) ===")
    print(freq_result.summary())
    print()

    sev_result, sev_columns = fit_severity_model(df)
    print("=== Severity model (Gamma GLM) ===")
    print(sev_result.summary())
    print()

    scored_df = predict_pure_premium(df, freq_result, sev_result, sev_columns)

    print("=== Portfolio totals ===")
    print(f"Total actual losses:    ${scored_df['y'].sum():,.0f}")
    print(f"Total predicted premium: ${scored_df['pure_premium'].sum():,.0f}")
    print()

    print("=== Decile lift table ===")
    print(decile_lift_table(scored_df).to_string(index=False))
    print()

    gini = gini_coefficient(scored_df)
    print(f"Gini coefficient (rank-ordering ability): {gini:.3f}")

    return scored_df, freq_result, sev_result


if __name__ == "__main__":
    main()
