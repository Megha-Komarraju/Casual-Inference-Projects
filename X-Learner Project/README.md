# X-Learner: Estimating Treatment Effects with EconML

## Overview
This repository demonstrates how to implement the **X-Learner**, a meta-learning approach for **causal inference**. The X-Learner estimates **Individual Treatment Effects (ITEs)** using machine learning models.

We use **Gradient Boosting Regressors** as base learners and **econML’s X-Learner** framework to estimate the treatment effect of an intervention.

## Installation
First, install the required dependencies:
```bash
pip install numpy pandas scikit-learn econml matplotlib
```

## Dataset
We simulate a dataset where the outcome `Y` depends on features `X`, with an assigned treatment `T`. The treatment effect varies based on a specific feature.

## Code Implementation

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from econml.metalearners import XLearner

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
X = np.random.normal(0, 1, (n_samples, 3))
T = np.random.binomial(1, 0.5, size=n_samples)  # Random treatment assignment

# Define potential outcomes (Y0 and Y1)
Y0 = 50 + 5 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 5, size=n_samples)
treatment_effect = 10 + 3 * X[:, 2]  # Treatment effect depends on feature X[:, 2]
Y1 = Y0 + treatment_effect

# Observed outcome
Y = Y1 * T + Y0 * (1 - T)

# Split into train and test sets
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.2, random_state=42)

# Define base learners
model_treated = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
model_control = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
propensity_model = LogisticRegression()

# Initialize and fit the X-Learner
x_learner = XLearner(models=(model_treated, model_control),
                      propensity_model=propensity_model)
x_learner.fit(Y_train, T_train, X=X_train)

# Estimate treatment effects
ite_estimates = x_learner.effect(X_test)

# Print first 5 estimated treatment effects
print("First 5 estimated treatment effects:", ite_estimates[:5])
```

## Explanation
1. **Generate Data**: We create synthetic data where `Y0` (outcome without treatment) is linearly dependent on `X`, and `Y1` (outcome with treatment) adds a treatment effect that varies with `X[:, 2]`.
2. **Train-Test Split**: The dataset is divided into training and testing subsets.
3. **Base Learners**: We use **Gradient Boosting Regressors** for treatment and control groups.
4. **Propensity Model**: A **Logistic Regression** model estimates the probability of treatment.
5. **X-Learner Training**: The X-Learner is trained to estimate the **Individual Treatment Effects (ITEs)**.
6. **Prediction**: We estimate the treatment effects on test data.

## Expected Output
After running the script, you should see something like:
```
First 5 estimated treatment effects: [12.3, 9.8, 11.5, 10.2, 13.1]
```
These values represent the estimated treatment effect for different individuals.

## Applications
- Uplift Modeling in Marketing
- Personalized Medicine
- Policy Evaluation
- A/B Testing

## References
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for Estimating Heterogeneous Treatment Effects using Machine Learning.
- EconML Documentation: [https://econml.azurewebsites.net](https://econml.azurewebsites.net)

---
### Author
This repository was created for showcasing **Causal Inference with Machine Learning**. 🚀

