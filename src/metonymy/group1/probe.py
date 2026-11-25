"""Group 1 frozen-probe helper, retained for upstream trainer compatibility."""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize

def fit_logistic(x, y, lambda_):
    """Logistic regression with an L2 penalty (the bias is not penalised).

    Literal and metonymic mentions get equal total weight, so the model is
    not pushed towards the majority class of its training set.
    """
    positive = y.sum()
    weights = np.where(y == 1, 1 / (2 * positive), 1 / (2 * (len(y) - positive)))

    def loss_and_gradient(w):
        z = x @ w[1:] + w[0]
        loss = np.sum(weights * (np.logaddexp(0, z) - y * z)) + lambda_ * w[1:] @ w[1:]
        error = weights * (expit(z) - y)
        gradient = np.concatenate([[error.sum()], x.T @ error + 2 * lambda_ * w[1:]])
        return loss, gradient

    result = minimize(loss_and_gradient, np.zeros(x.shape[1] + 1), jac=True, method="L-BFGS-B")
    return result.x

