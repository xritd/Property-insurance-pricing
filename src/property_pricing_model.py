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
# Candidate/final predictors:
# LnCoverage, NoClaimCredit, Fire5, and entity type dummies.
#
# Fire5 is tested during model selection but excluded from the final model.



FREQ_TYPE_DUMMIES = ["TypeCounty", 
                     "TypeMisc", 
                     "TypeSchool", 
                     "TypeTown", 
                     "TypeVillage"]

FREQ_PREDICTORS = ["LnCoverage", 
                   "NoClaimCredit"] + FREQ_TYPE_DUMMIES


def fit_frequency_model(
    df: pd.DataFrame,
    predictors=None,
    cluster_se: bool = True,
):
    """
    Fit a Negative Binomial frequency model.

    Cluster-robust standard errors are used for the final model.
    Model-selection comparisons use the ordinary likelihood.
    """
    if predictors is None:
        predictors = FREQ_PREDICTORS

    X = sm.add_constant(df[predictors])
    y = df["Freq"]

    mean_y, var_y = y.mean(), y.var()
    alpha_start = (var_y / mean_y - 1) / mean_y
    start_params = [np.log(mean_y)] + [0.0] * len(predictors) + [alpha_start]

    model = sm.NegativeBinomial(y, X)

    if cluster_se:
        result = model.fit(
            start_params=start_params,
            method="bfgs",
            maxiter=300,
            disp=0,
            cov_type="cluster",
            cov_kwds={"groups": df["PolicyNum"]},
        )
    else:
        result = model.fit(
            start_params=start_params,
            method="bfgs",
            maxiter=300,
            disp=0,
        )

    return result


def frequency_model_selection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the staged frequency-model selection described in the report.

    Each specification is compared with the previous specification using
    a likelihood-ratio test. Cluster-robust SEs are not used here because
    the likelihood is being compared.
    """
    specifications = [
        ("Intercept only", []),
        ("LnCoverage", ["LnCoverage"]),
        (
            "LnCoverage + NoClaimCredit",
            ["LnCoverage", "NoClaimCredit"],
        ),
        (
            "LnCoverage + NoClaimCredit + Fire5",
            ["LnCoverage", "NoClaimCredit", "Fire5"],
        ),
        (
            "LnCoverage + NoClaimCredit + EntityType",
            [
                "LnCoverage",
                "NoClaimCredit",
                "TypeCounty",
                "TypeMisc",
                "TypeSchool",
                "TypeTown",
                "TypeVillage",
            ],
        ),
    ]

    rows = []
    previous_result = None

    for step, (name, predictors) in enumerate(specifications, start=1):
        result = fit_frequency_model(
            df,
            predictors=predictors,
            cluster_se=False,
        )

        if previous_result is None:
            lr_stat = np.nan
            p_value = np.nan
        else:
            lr_stat = 2 * (result.llf - previous_result.llf)
            df_difference = result.df_model - previous_result.df_model
            p_value = sps.chi2.sf(lr_stat, df_difference)

        rows.append({
            "Step": step,
            "Specification": name,
            "Log-likelihood": result.llf,
            "LR statistic": lr_stat,
            "p-value": p_value,
        })

        previous_result = result

    return pd.DataFrame(rows)


# Severity model (Gamma GLM)
#
# Fit only on policy-years with at least one claim (Freq > 0).
#
# Candidate predictors include LnCoverage, DeductBin, Fire5,
# NoClaimCredit, and entity type.
#
# Final predictor: DeductBin, with claim-count weighting.


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


def severity_model_selection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the staged severity-model selection described in the report.

    The original model-selection analysis was unweighted. Claim-count
    weighting is applied only to the final severity model after the
    methodological review.

    LnCoverage and DeductBin are alternative specifications rather than
    nested models, so the move from LnCoverage to DeductBin is treated as a
    specification comparison rather than a formal nested-model test.
    """
    claims_df = add_deduct_bin(df[df["Freq"] > 0])

    def fit(formula):
        return sm.formula.glm(
            formula=formula,
            data=claims_df,
            family=sm.families.Gamma(
                link=sm.families.links.Log()
            ),
        ).fit()

    intercept = fit("yAvg ~ 1")

    coverage = fit("yAvg ~ LnCoverage")

    deductible = fit("yAvg ~ C(DeductBin)")

    deductible_fire = fit(
        "yAvg ~ C(DeductBin) + Fire5"
    )

    deductible_credit = fit(
        "yAvg ~ C(DeductBin) + NoClaimCredit"
    )

    deductible_entity = fit(
        """
        yAvg ~ C(DeductBin)
             + TypeCounty
             + TypeMisc
             + TypeSchool
             + TypeTown
             + TypeVillage
        """
    )

    f_coverage, p_coverage = deviance_f_test(
        intercept.deviance,
        intercept.df_resid,
        coverage.deviance,
        coverage.df_resid,
        coverage.scale,
    )

    f_fire, p_fire = deviance_f_test(
        deductible.deviance,
        deductible.df_resid,
        deductible_fire.deviance,
        deductible_fire.df_resid,
        deductible_fire.scale,
    )

    f_credit, p_credit = deviance_f_test(
        deductible.deviance,
        deductible.df_resid,
        deductible_credit.deviance,
        deductible_credit.df_resid,
        deductible_credit.scale,
    )

    f_entity, p_entity = deviance_f_test(
        deductible.deviance,
        deductible.df_resid,
        deductible_entity.deviance,
        deductible_entity.df_resid,
        deductible_entity.scale,
    )

    return pd.DataFrame([
        {
            "Step": 1,
            "Specification": "Intercept only",
            "Deviance": intercept.deviance,
            "F statistic": np.nan,
            "p-value": np.nan,
        },
        {
            "Step": 2,
            "Specification": "LnCoverage",
            "Deviance": coverage.deviance,
            "F statistic": f_coverage,
            "p-value": p_coverage,
        },
        {
            "Step": 3,
            "Specification": "DeductBin (replaces LnCoverage)",
            "Deviance": deductible.deviance,
            "F statistic": np.nan,
            "p-value": np.nan,
        },
        {
            "Step": 4,
            "Specification": "DeductBin + Fire5",
            "Deviance": deductible_fire.deviance,
            "F statistic": f_fire,
            "p-value": p_fire,
        },
        {
            "Step": 5,
            "Specification": "DeductBin + NoClaimCredit",
            "Deviance": deductible_credit.deviance,
            "F statistic": f_credit,
            "p-value": p_credit,
        },
        {
            "Step": 6,
            "Specification": "DeductBin + EntityType",
            "Deviance": deductible_entity.deviance,
            "F statistic": f_entity,
            "p-value": p_entity,
        },
    ])


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

    print("=== Frequency model selection ===")
    print(
        frequency_model_selection(df).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print()

    print("=== Severity model selection ===")
    print(
        severity_model_selection(df).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print()

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
