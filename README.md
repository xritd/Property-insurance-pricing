# Property Insurance Pricing: Frequency-Severity Model

A frequency-severity pricing model for property insurance claims, built on
real public claims data from the Wisconsin Local Government Property
Insurance Fund (LGPIF). Predicts expected annual loss cost (pure premium)
per policy using a Negative Binomial GLM for claim frequency and a Gamma
GLM for claim severity.

## Results

- **Rank-ordering ability (Gini coefficient): 0.693** — strong separation
  between relatively lower- and higher-risk policy-years
- **Portfolio-level calibration**: predicted pure premium of $112.9M vs.
  $97.5M actual losses (1.16x), after correcting two methodological issues
  found during a self-review of the initial model (see [full report](docs/report.md))
- **Deciles 2–9 track actual losses relatively closely**, while the lowest
  and highest deciles show the largest calibration differences

| Decile | Avg. Predicted | Avg. Actual |
|---|---|---|
| 1 (lowest risk) | $196 | $2,186 |
| 5 | $3,954 | $3,734 |
| 10 (highest risk) | $124,668 | $95,201 |

## What's here

- 'src/property_pricing_model.py' — full modeling pipeline: data loading,
  frequency model (Negative Binomial GLM, cluster-robust SEs), severity
  model (Gamma GLM, claim-count weighted), pure premium calculation, and
  evaluation (decile lift table, Gini coefficient)
- 'docs/report.md' — full writeup: exploratory data analysis, incremental
  model-building process (including variables tested and rejected), and a
  self-review section documenting two real statistical mistakes found and
  fixed after the initial model was built

## Running it

```bash
pip install -r requirements.txt
python src/property_pricing_model.py
```

No manual data download needed — the script pulls the dataset directly from
its [public source](https://github.com/OpenActTexts/LDACourse1) at runtime.

## Methodology highlights

- **Frequency**: Negative Binomial GLM. Claim counts are strongly
  overdispersed relative to their mean (~66×), providing strong evidence
  that a plain Poisson model is inappropriate. Predictors: log coverage,
  prior no-claims credit, entity type.
- **Severity**: Gamma GLM on claims with 'Freq > 0', weighted by claim
  count ('var_weights=Freq') since each observation is itself an average
  over multiple claims. Predictor: deductible (binned).
- **Standard errors clustered on policyholder** — the data is a 5-year
  panel (2006–2010), and treating repeated observations of the same entity
  as independent understated significance by a meaningful margin.
- Full incremental variable-selection process — including variables that
  were tested and dropped for showing no effect ('Fire5' on both
  frequency and severity) — is documented in the report rather than only
  showing the final model.

## Known limitations

- Evaluated in-sample only; the dataset's companion out-of-sample test
  file was not used
- A handful of policies with extreme, sparsely-represented coverage values
  still produce somewhat unstable predictions (documented in the report)
- No residual diagnostics (deviance residual plots, distributional
  goodness-of-fit checks) were performed

## Data source

[Wisconsin Local Government Property Insurance Fund (LGPIF)](https://github.com/OpenActTexts/LDACourse1),
companion dataset to the open-access textbook *Loss Data Analytics*
(Frees, Derrig, and Meyers, eds.).
