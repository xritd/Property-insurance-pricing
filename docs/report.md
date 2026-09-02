# Property Insurance Pricing: Frequency-Severity Model
### A Walkthrough Using the Wisconsin Local Government Property Insurance Fund (LGPIF)

## 1. Overview

This project builds a **pure premium model** for property insurance claims using
frequency-severity modeling, the standard actuarial approach to insurance pricing.
Rather than predicting total claim cost directly, we model two separate processes:

- **Frequency**: how often a policyholder files a claim
- **Severity**: how large each claim is, given that one occurs

The two are combined as `Pure Premium = Expected Frequency × Expected Severity`,
which represents the expected annual loss cost per policy — the foundation on
which an actual premium (loss cost + expenses + profit margin) would be built.

## 2. Data

**Source**: Wisconsin Local Government Property Insurance Fund (LGPIF), an
insurance pool administered by the Wisconsin Office of the Insurance
Commissioner, covering property losses for local government entities
(cities, counties, schools, towns, villages). Data spans policy-years 2006–2010.

- **5,639** policy-year observations
- **23** columns, including claim counts, claim amounts, coverage, deductible,
  entity type, and alarm/fire-safety indicators
- Data source: [OpenActTexts/LDACourse1 GitHub repository](https://github.com/OpenActTexts/LDACourse1),
  companion dataset to the open-access textbook *Loss Data Analytics* (Frees et al.)
- **Panel structure**: each `PolicyNum` appears in up to 5 consecutive years,
  which matters for how we compute standard errors (see Section 6).

### Key exploratory findings

- **70.2%** of policy-years have zero claims — a large proportion of zero-claim observations
- Claim frequency is strongly **overdispersed**: sample variance (~73) is roughly
  66× the sample mean (~1.11), providing strong evidence that a plain Poisson model is inappropriate
- Claim severity is strongly **right-skewed**: mean claim size ($31,206) is
  6.7× the median ($4,645), driven by a handful of multi-million-dollar claims
  (max: $12.9M)

These results led me to use a Negative Binomial model for frequency and a Gamma model for severity.

## 3. Frequency Model (Negative Binomial GLM)

I built the frequency model incrementally, adding variables in stages and checking whether each addition materially improved the fit (likelihood-ratio tests; cluster-robust SEs are *not* used during selection because the likelihood itself is being compared).

| Step | Variable added | Log-likelihood | Result |
| --- | --- | --- | --- |
| 1 | Intercept only (baseline) | −6,631.7 | — |
| 2 | LnCoverage (log of insured value) | −5,669.6 | Large improvement (ΔLL = 1,924, p < 0.001) |
| 3 | NoClaimCredit (prior claims-free discount) | −5,616.3 | Significant (ΔLL = 107, p < 0.001) |
| 4 | Fire5 (fire class) | −5,616.3 | Dropped — no improvement (p = 0.86) |
| 5 | Entity type (TypeCounty, TypeMisc, TypeSchool, TypeTown, TypeVillage) | −5,519.5 | Significant as a group (ΔLL = 194, p < 0.001), driven mainly by School and Misc |

**Final frequency model**: `LnCoverage + NoClaimCredit + EntityType`, fit with
**cluster-robust standard errors** (clustered on `PolicyNum` — see Section 6).

**Key interpretations** (using clustered SEs):

| Variable | Coefficient | Clustered p-value | Interpretation |
| --- | --- | --- | --- |
| LnCoverage | 0.784 | < 0.001 | Frequency scales as roughly Coverage^0.78 — sub-proportional growth, suggesting economies of scale for larger entities |
| NoClaimCredit | −0.686 | < 0.001 | ~50% fewer expected claims, holding coverage constant |
| TypeSchool | −0.853 | < 0.001 | ~57% fewer claims than City (reference) |
| TypeMisc | −0.619 | 0.050 | ~46% fewer claims than City — right at the edge of significance once properly clustered |
| TypeCounty, TypeTown, TypeVillage | near zero | not significant | Statistically indistinguishable from City |

## 4. Severity Model (Gamma GLM)

Fit only on the 1,679 policy-years with at least one claim. Because the Gamma
GLM estimates a nuisance dispersion parameter, model comparisons used
**F-tests on deviance reduction** rather than raw likelihood-ratio tests.

The original model-selection analysis was **unweighted**. Claim-count weighting
is applied only to the final severity model (see Section 6).

| Step | Variable added | Deviance | Result |
| --- | --- | --- | --- |
| 1 | Intercept only (baseline) | 5,808.0 | — |
| 2 | LnCoverage | 5,260.9 | Marginal (F-test p = 0.022) |
| 3 | DeductBin (replaces LnCoverage) | 4,186.5 | Much better-fitting alternative specification; LnCoverage was dropped |
| 4 | Fire5 | 4,182.2 | Dropped — no improvement (p = 0.67) |
| 5 | NoClaimCredit | 4,180.3 | Dropped — no improvement (p = 0.60) |
| 6 | Entity type | 4,008.5 | Dropped — borderline, not significant (p = 0.079) |

**Final severity model**: `DeductBin` (5 categories), fit with
**claim-count weighting** (`var_weights=Freq` — see Section 6).

**Key interpretation**: The unweighted analysis showed higher deductibles were associated with higher
average reported claim size, largely a **truncation/selection
effect** (policies with high deductibles only ever report claims that clear
a high bar). However, once the model is properly weighted by how many
claims underlie each average, the effect at the highest deductible tier
(`10,000+`) is much weaker and no longer statistically significant — see
Section 6 for why this matters.

## 5. Combined Pure Premium & Model Evaluation

Pure premium was computed per policy-year as `predicted frequency × predicted
severity` and evaluated by ranking policies into deciles.

**Rank-ordering ability**: Gini coefficient = **0.693**. The model shows
strong separation between relatively lower- and higher-risk policy-years,
although absolute dollar predictions are not perfectly calibrated.

**Portfolio-level calibration**:

| --- | Total |
|--- | --- |
| Actual losses | $97,483,101 |
| Predicted pure premium | $112,917,989 |
| Ratio | 1.16× |

**Calibration by decile**:

| Decile | Avg. Predicted | Avg. Actual |
| --- | --- | --- |
| 1 (lowest) | $196 | $2,186 |
| 2 | $541 | $630 |
| 3 | $1,338 | $1,380 |
| 4 | $2,451 | $2,077 |
| 5 | $3,954 | $3,734 |
| 6 | $6,444 | $6,757 |
| 7 | $10,479 | $12,534 |
| 8 | $17,513 | $16,108 |
| 9 | $32,636 | $32,248 |
| 10 (highest) | $124,668 | $95,201 |

Deciles 2–9 track actual losses closely. Decile 10 still overpredicts somewhat (about 1.3×). This is consistent with
the extreme-coverage extrapolation issue discussed below.

### Remaining limitation: extrapolation at the extreme end of coverage

A small number of policies with very high `LnCoverage` (near the top of the
observed range) still produce somewhat inflated predictions. This traces
back to a single high-leverage entity in the training data with both extreme
coverage *and* extreme historical claims, which drives a steep fitted
relationship at that end of the range. Other high-coverage policies with
different (much lower) claims histories partly inherit that same steep
prediction. **In a production setting**, this would be addressed via
credibility weighting, capping/winsorizing extreme covariate values, or
manual underwriting review for the largest accounts.

### Additional limitation: in-sample evaluation only

All results above were evaluated on the same data used to fit the models.
The LGPIF dataset includes a companion out-of-sample test file
(`PropertyFundOutsample.csv`) that was not used in this project and would be
the natural next validation step.

## 6. Methodological Review and Refinement

After building the initial models, I deliberately stepped back and
critiqued the approach before finalizing it — two of the issues found were not nitpicks but genuinely changed
the model’s output.

### 6.1 Panel structure was ignored (fixed)

Each `PolicyNum` appears in the data up to 5 times (2006–2010). Standard GLM
inference assumes independent observations, but a given entity’s claims
experience is likely correlated year to year. Comparing naive vs.
cluster-robust standard errors (clustered on `PolicyNum`) on the frequency
model showed SEs were **1.5× to 4× too small** under the naive approach —
enough to flip `TypeMisc` from “clearly significant” (p ≈ 0.0001) to
“borderline” (p = 0.050). **Fixed** by refitting with
`cov_type="cluster", cov_kwds={"groups": df["PolicyNum"]}`.

### 6.2 Severity model wasn’t weighted by claim count (fixed)

`yAvg` (average claim severity) is computed by averaging over however many
claims (`Freq`) a policy had that year. A policy averaging 50 claims
produces a far more reliable estimate than one averaging a single claim, but
an unweighted Gamma GLM treats every row as equally informative. Fitting
with `var_weights=Freq` corrected this and had a large effect: the
`Deduct 10,000+` coefficient dropped from 2.28 (highly significant) to 0.40
(not significant). This accounts for most of the reduction in portfolio-level over-prediction
seen during the refinement process — **total predicted premium fell
from $284.8M to $112.9M** (vs. $97.5M actual) once this was corrected, while
the Gini coefficient stayed essentially flat (0.694 → 0.693), confirming
the fix improved calibration without hurting the model’s ability to
rank-order risk.

### 6.3 Severity model wasn’t tested against the same variables as frequency (checked, not fixed)

`NoClaimCredit` and entity type mattered for frequency but were not included
in the initial severity-selection workflow — an oversight from momentum in the workflow
rather than a deliberate choice. I subsequently checked both against the selected deductible specification.
`NoClaimCredit` does not improve the model (p = 0.60), while entity type is
borderline but falls short of conventional significance (p = 0.079). The
omission didn’t change the final model, but it should have been checked and
reported regardless — a null result you looked for is different from one you
never checked.

### 6.4 Not yet addressed

- **No residual diagnostics were performed.** All model comparisons relied
  on aggregate deviance/log-likelihood statistics; deviance residuals were
  never plotted, and Gamma vs. lognormal fit was never checked visually.
- **Stepwise variable selection has no correction for selection bias.**
  Testing variables one at a time and keeping what’s significant, evaluated
  on the same data throughout, tends to overstate significance somewhat. A
  more rigorous version would specify candidates upfront and compare via
  AIC/BIC or cross-validated deviance.
- **Extreme-coverage extrapolation** (Section 5) is documented but not
  resolved.

## 7. Summary of Modeling Decisions

| Decision | Rationale |
| --- | -- |
| Negative Binomial (not Poisson) for frequency | Severe overdispersion in claim counts (variance ≫ mean) |
| Gamma GLM (not linear regression) for severity | Strictly positive, heavily right-skewed claim sizes |
| Frequency and severity modeled separately | Different variables were useful for the two components — coverage, NoClaimCredit and entity type for frequency, versus deductible for severity |
| Deductible binned rather than continuous (log) | Continuous form extrapolated unreliably into a sparse high-deductible region |
| F-tests (not LR tests) for Gamma model comparisons | Gamma GLM’s estimated dispersion parameter makes raw likelihood-ratio tests unreliable |
| Cluster-robust SEs (clustered on PolicyNum) for frequency | Policy-years are not independent; naive SEs understated uncertainty by 1.5×–4× |
| Claim-count weighting (var_weights=Freq) for severity | Averages over more claims are more reliable; unweighted fit let a thin, noisy high-deductible bin drive large over-predictions |
| Fire5 excluded from both models | No significant effect on frequency or severity, despite plausible intuition |

## 8. Suggested Next Steps

1. Validate on the out-of-sample test set
2. Address the remaining high-coverage extrapolation issue (capping, credibility weighting, or robust regression)
3. Add residual diagnostics (deviance residual plots, Gamma vs. lognormal comparison)
4. Explore a Tweedie GLM as a single-model alternative to the two-stage frequency-severity approach
5. Compare against a gradient-boosted tree model (e.g., LightGBM with a Tweedie objective) as a modern ML benchmark
