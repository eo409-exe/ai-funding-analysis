"""Reusable modelling helpers for the AI funding analysis project."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def build_design_matrix(
    df: pd.DataFrame,
    features: dict,
    target: str,
    add_constant: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) for a regression by selecting features + dropping rows with NA.

    Parameters
    ----------
    df : DataFrame containing both features and target.
    features : ordered mapping of {display_name: column_name}.
    target : name of the dependent-variable column.
    add_constant : if True, prepend a constant column for statsmodels.
    """
    cols = list(features.values()) + [target]
    sub = df[cols].dropna().copy()
    X = sub[list(features.values())].astype(float)
    y = sub[target].astype(float)
    if add_constant:
        X = sm.add_constant(X, has_constant='add')
    X.columns = (['const'] if add_constant else []) + list(features.keys())
    return X, y


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor for each column (excluding 'const' if present).

    Returns DataFrame with columns: feature, vif, flag (True if VIF > 5).
    """
    cols = [c for c in X.columns if c.lower() != 'const']
    vifs = [variance_inflation_factor(X[cols].values, i) for i in range(len(cols))]
    out = pd.DataFrame({'feature': cols, 'vif': vifs})
    out['flag'] = out['vif'] > 5
    return out.sort_values('vif', ascending=False).reset_index(drop=True)


def coef_forest_plot(
    coef_df: pd.DataFrame,
    ax,
    *,
    estimate_col: str = 'estimate',
    lo_col: str = 'lo',
    hi_col: str = 'hi',
    sig_col: str = 'sig',
    label_col: str = 'feature',
    ref_value: float = 1.0,
    ref_label: str = 'No effect',
    x_max: float | None = None,
    sig_color: str = '#08306b',
    ns_color: str = '#9ecae1',
    label_suffix: str = 'x',
    title: str | None = None,
):
    """Generic forest plot — works for both OLS β (ref=0) and Logit OR (ref=1).

    Parameters
    ----------
    coef_df : DataFrame sorted ascending by `estimate_col`.
    ax : matplotlib axes.
    estimate_col, lo_col, hi_col, sig_col, label_col : columns in coef_df.
    ref_value : x-position of reference line (1 for OR, 0 for β).
    x_max : optional CI clip for display (clipped CIs get an arrow cap).
    """
    coef_df = coef_df.reset_index(drop=True)
    n = len(coef_df)
    y_pos = np.arange(n)

    hi_disp = coef_df[hi_col].copy()
    clipped = pd.Series([False] * n)
    if x_max is not None:
        clipped = coef_df[hi_col] > x_max
        hi_disp = hi_disp.clip(upper=x_max)

    colors = [sig_color if s else ns_color for s in coef_df[sig_col]]

    ax.hlines(y_pos, coef_df[lo_col], hi_disp,
              colors=colors, linewidth=2.4, alpha=0.8)
    for i, is_clip in enumerate(clipped):
        if is_clip:
            ax.annotate(
                '', xy=(x_max + 0.05, i), xytext=(x_max - 0.10, i),
                arrowprops=dict(arrowstyle='->', color=colors[i],
                                lw=2.2, alpha=0.85)
            )
    ax.scatter(coef_df[estimate_col], y_pos,
               color=colors, s=100, zorder=5,
               linewidths=0.8, edgecolors='white')

    ax.axvline(ref_value, color='#888888', linestyle='--', linewidth=1.2, zorder=1)
    ax.text(ref_value, n - 0.35, ref_label,
            fontsize=9, color='#636363', style='italic', ha='center',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.95))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_df[label_col], fontsize=11)
    ax.set_ylim(-0.7, n - 0.3)
    if title:
        ax.set_title(title, pad=14)

    return ax


def fit_logit_with_or(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str] | None = None,
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
    """Fit a logistic regression and return (model, coef_df with odds-ratio + CI)."""
    model = sm.Logit(y, X).fit(disp=False)
    params = model.params.drop('const', errors='ignore')
    bse = model.bse.loc[params.index]
    pvals = model.pvalues.loc[params.index]
    out = pd.DataFrame({
        'feature':  feature_names or list(params.index),
        'coef':     params.values,
        'se':       bse.values,
        'pval':     pvals.values,
    })
    out['estimate'] = np.exp(out['coef'])
    out['lo']       = np.exp(out['coef'] - 1.96 * out['se'])
    out['hi']       = np.exp(out['coef'] + 1.96 * out['se'])
    out['sig']      = out['pval'] < 0.05
    return model, out


def fit_ols_with_pct(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str] | None = None,
    cov_type: str = 'HC3',
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
    """Fit OLS with robust SEs and return (model, coef_df with %-effect + CI)."""
    model = sm.OLS(y, X).fit(cov_type=cov_type)
    params = model.params.drop('const', errors='ignore')
    bse = model.bse.loc[params.index]
    pvals = model.pvalues.loc[params.index]
    out = pd.DataFrame({
        'feature':  feature_names or list(params.index),
        'coef':     params.values,
        'se':       bse.values,
        'pval':     pvals.values,
    })
    out['estimate']    = out['coef']
    out['lo']          = out['coef'] - 1.96 * out['se']
    out['hi']          = out['coef'] + 1.96 * out['se']
    out['pct_effect']  = (np.exp(out['coef']) - 1) * 100
    out['sig']         = out['pval'] < 0.05
    return model, out