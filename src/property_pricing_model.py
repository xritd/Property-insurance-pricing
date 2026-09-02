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


def load_data() -> pd.DataFrame:
    """Load the LGPIF property insurance dataset."""
    df = pd.read_csv(DATA_URL)
    return df


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
    Fit a Negative Binomial model for annual claim frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Policy-year data containing claim frequency and model predictors.
    predictors : list, optional
        Column names to use as predictors. Defaults to the final frequency
        model predictors.
    cluster_se : bool, default=True
        If True, use standard errors clustered by PolicyNum.

    Returns
    -------
    statsmodels result
        Fitted Negative Binomial regression results.
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
    Compare a sequence of Negative Binomial frequency model specifications.

    Each specification is compared with the preceding nested specification
    using a likelihood-ratio test. Standard errors are not clustered during
    model selection because the comparison is based on model likelihoods.

    Parameters
    ----------
    df : pd.DataFrame
        Policy-year data containing claim frequency and candidate predictors.

    Returns
    -------
    pd.DataFrame
        Model specifications, log-likelihoods, likelihood-ratio statistics,
        and p-values.
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


DEDUCT_BINS = [0, 1000, 2500, 5000, 10000, np.inf]
DEDUCT_LABELS = ["<=1000", 
                 "1000-2500", 
                 "2500-5000", 
                 "5000-10000", 
                 "10000+"]


def add_deduct_bin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a categorical deductible variable using predefined deductible bands.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing a Deduct column.

    Returns
    -------
    pd.DataFrame
        Copy of the input data with a DeductBin column.
    """
    df = df.copy()
    df["DeductBin"] = pd.cut(df["Deduct"], 
                             bins=DEDUCT_BINS, 
                             labels=DEDUCT_LABELS)
    return df


def fit_severity_model(df: pd.DataFrame):
    """
    Fit a Gamma GLM for average claim severity.

    The model is fitted only to policy-years with at least one claim.
    Observations are weighted by claim frequency because yAvg represents
    an average over the number of claims reported in that policy-year.

    Parameters
    ----------
    df : pd.DataFrame
        Policy-year data containing claim frequency, average claim severity,
        and deductible information.

    Returns
    -------
    result : statsmodels result
        Fitted Gamma GLM results.
    columns : pandas.Index
        Predictor column names used by the fitted model.
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
    """
    Compare two nested Gamma GLMs using an F-test based on deviance reduction.

    Parameters
    ----------
    deviance_reduced : float
        Deviance of the reduced model.
    df_reduced : float
        Residual degrees of freedom of the reduced model.
    deviance_full : float
        Deviance of the full model.
    df_full : float
        Residual degrees of freedom of the full model.
    dispersion : float
        Estimated dispersion parameter from the full model.

    Returns
    -------
    f_stat : float
        F-test statistic.
    p_value : float
        Upper-tail p-value for the F-test.
    """
    f_stat = ((deviance_reduced - deviance_full) / (df_reduced - df_full)) / dispersion
    p_value = sps.f.sf(f_stat, dfn=(df_reduced - df_full), dfd=df_full)
    return f_stat, p_value


def severity_model_selection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare candidate Gamma GLM specifications for claim severity.

    The analysis uses policy-years with at least one claim. LnCoverage and
    DeductBin are evaluated as alternative baseline specifications. Additional
    predictors are then tested against the DeductBin specification using
    F-tests based on deviance reduction.

    Parameters
    ----------
    df : pd.DataFrame
        Policy-year data containing claim severity and candidate predictors.

    Returns
    -------
    pd.DataFrame
        Model specifications, deviances, F-test statistics, and p-values.
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


def predict_pure_premium(df: pd.DataFrame, freq_result, sev_result, sev_columns) -> pd.DataFrame:
    """
    Calculate predicted pure premium for each policy-year.

    Pure premium is calculated as predicted claim frequency multiplied by
    predicted average claim severity.

    Parameters
    ----------
    df : pd.DataFrame
        Policy-year data to score.
    freq_result : statsmodels result
        Fitted Negative Binomial frequency model.
    sev_result : statsmodels result
        Fitted Gamma severity model.
    sev_columns : pandas.Index
        Predictor columns used by the fitted severity model.

    Returns
    -------
    pd.DataFrame
        Copy of the input data containing predicted frequency, predicted
        severity, and predicted pure premium.
    """
    df = add_deduct_bin(df).copy()

    Xf = sm.add_constant(df[FREQ_PREDICTORS])
    df["pred_freq"] = freq_result.predict(Xf)

    dummies = pd.get_dummies(df["DeductBin"], prefix="Ded", drop_first=True).astype(float)
    Xs = sm.add_constant(dummies)
    Xs = Xs.reindex(columns=sev_columns, fill_value=0.0)
    df["pred_sev"] = sev_result.predict(Xs)

    df["pure_premium"] = df["pred_freq"] * df["pred_sev"]
    return df


def decile_lift_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group policy-years into pure-premium deciles and summarize predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Scored policy-year data containing pure_premium and actual losses.

    Returns
    -------
    pd.DataFrame
        Decile-level policy counts, average predicted pure premiums, and
        average actual losses.
    """
    df = df.copy()
    df["decile"] = pd.qcut(df["pure_premium"], 10, labels=False, duplicates="drop") + 1
    return df.groupby("decile").agg(
        n_policies=("PolicyNum", "count"),
        avg_predicted=("pure_premium", "mean"),
        avg_actual=("y", "mean"),
    ).reset_index()


def gini_coefficient(df: pd.DataFrame) -> float:
    """
    Calculate the Gini coefficient for predicted pure-premium ranking.

    A higher value indicates better separation of policy-years by observed
    losses when ranked by predicted pure premium.

    Parameters
    ----------
    df : pd.DataFrame
        Scored policy-year data containing pure_premium and actual losses.

    Returns
    -------
    float
        Gini coefficient.
    """
    sorted_df = df.sort_values("pure_premium").reset_index(drop=True)
    cum_policies_pct = (np.arange(len(sorted_df)) + 1) / len(sorted_df)
    cum_actual_pct = sorted_df["y"].cumsum() / sorted_df["y"].sum()
    area_under_lorenz = np.trapezoid(cum_actual_pct, cum_policies_pct)
    return 1 - 2 * area_under_lorenz


def main():
    """Run the complete modeling and evaluation pipeline."""
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
