# Loan-risk drift demonstration

This demo uses synthetic data and a locally trained logistic regression model. Drift does not prove performance degradation.

Start DriftWatch and install the demo dependencies, then run:

```bash
docker compose up -d
python -m pip install -r requirements-demo.txt
python -m examples.loan_risk_demo.run_demo
```

Set `DRIFTWATCH_URL` to use an API other than `http://localhost:8000`. The script checks readiness, trains and reloads a model, registers the reference population, sends two weekly batches, and prints API results and measured AUCs. Reports and the model are saved in `artifacts/`. These model artifacts are pickles; never load one from an untrusted source.

Each feature is generated independently. The reference and stable populations use normal distributions for credit score (mean 700, standard deviation 50), annual income (70,000, 15,000), debt-to-income ratio (0.28, 0.08), and employment years (8, 4); open credit accounts follow a Poisson distribution with mean 5. Annual income, debt-to-income ratio, and employment years are floored at 0. The shifted population changes the first three means to 650, 50,000, and 0.42 respectively, while retaining their standard deviations and the other feature distributions.

Labels are Bernoulli samples with probability equal to the logistic function applied to:

```text
-0.5 - 0.9*(credit_score-700)/50 - 0.5*(annual_income-70000)/15000
+ 1.0*(debt_to_income-0.28)/0.08 - 0.2*(employment_years-8)/4
+ 0.1*(open_credit_accounts-5)
```

## Interpreting the measurements

- **Data drift** measures changes in input feature distributions.
- **Prediction drift** measures changes in the distribution of model probabilities.
- **Concept drift** means the relationship between inputs and outcomes changes; this demo keeps its label rule fixed.
- **Measured performance** compares predictions with known labels. AUC is measured separately on held-out reference data and each batch. Drift alone does not establish whether model performance has degraded.
