import numpy as np
from scipy import stats
import pandas as pd

def simple_linear_regression_summary(w, b, x_val, y_val, y_pred):
    """
    Generate a summary table for a simple linear regression model.
    
    Parameters:
        w (float): Weight coefficient
        b (float): Bias/intercept
        x_val (array-like): Input features
        y_val (array-like): True target values
        y_pred (array-like): Predicted target values
    
    Returns:
        summary_table (dict): Table of estimates, standard errors, t-stats, and p-values
    """
    x = np.array(x_val)
    y = np.array(y_val)
    y_hat = np.array(y_pred)
    n = len(y)
    
    # Residuals
    residuals = y - y_hat
    
    # Degrees of freedom
    df = n - 2
    
    # Residual variance (sigma^2)
    residual_var = np.sum(residuals**2) / df
    
    # Sum of squares
    SSx = np.sum((x - np.mean(x))**2)
    
    # Standard errors
    SE_w = np.sqrt(residual_var / SSx)
    SE_b = np.sqrt(residual_var * (1/n + (np.mean(x)**2)/SSx))
    
    # t-statistics
    t_w = w / SE_w
    t_b = b / SE_b
    
    # p-values (two-tailed)
    p_w = 2 * (1 - stats.t.cdf(np.abs(t_w), df))
    p_b = 2 * (1 - stats.t.cdf(np.abs(t_b), df))
    
    # Create pandas DataFrame
    summary_df = pd.DataFrame({
        "Parameter": ["Weight", "Bias"],
        "Estimate": [float(w), float(b)],
        "Std. Error": [float(SE_w), float(SE_b)],
        "t-Statistic": [float(t_w), float(t_b)],
        "p-Value": [float(p_w), float(p_b)]
    })
    
    return summary_df