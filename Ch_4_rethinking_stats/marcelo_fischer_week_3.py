import marimo

__generated_with = "0.19.6"
app = marimo.App(width="columns")


@app.cell
def _():
    import altair as alt
    import arviz as az
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import pymc as pm
    import scipy as sp
    from scipy.stats import beta, binom

    RANDOM_SEED = 1523
    rng = np.random.default_rng(RANDOM_SEED)

    az.style.use("arviz-darkgrid")
    return az, mo, np, pl, plt, pm


@app.cell
def _(pl):
    df = pl.read_csv(
        "/home/marcelo/git/rethinking_stats_and_PyMC/data/Howell1.csv", separator=";"
    )


    def center_age(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(age_centered=pl.col("age") - pl.mean("age"))


    def filter_age(df: pl.DataFrame, age: int) -> pl.DataFrame:
        return df.filter(pl.col("age") >= age)


    df_adults = (
        df
        .pipe(filter_age, 18)
        .pipe(center_age)
    )

    df_adults
    return (df_adults,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    \begin{align*}
    h_i &\sim \text{Normal}(\mu_i, \sigma) & [\text{likelihood}] \\
    \mu_i &= \alpha + \beta \bar{age_i} & [\text{linear model}] \\
    \alpha &\sim \text{Normal}(172, 20) & [\alpha \text{ prior}] \\
    \beta &\sim \text{Normal}(0, 0.1) & [\beta \text{ prior}] \\
    \sigma &\sim \text{Uniform}(0, 30) & [\sigma \text{ prior}]
    \end{align*}
    """)
    return


@app.cell
def _(df_adults, np, pm):
    m1_coords = {"obs": np.arange(df_adults["height"].len())}

    with pm.Model(coords=m1_coords) as m1:
        # ===== DATA LAYER =====
        # Define all your data variables once with dims
        # This prevents typos later because you reference the pm.Data object
        m1_age_c = pm.Data("age_c", df_adults["age_centered"], dims="obs")

        # ===== PRIORS =====
        m1_alpha = pm.Normal("alpha", mu=172, sigma=20)
        m1_beta = pm.Normal("beta", mu=0, sigma=0.1)
        m1_sigma = pm.Uniform("sigma", lower=0, upper=30)

        # ===== LINEAR MODEL =====
        # Because age_c has dims="obs", mu automatically inherits it
        m1_mu = pm.Deterministic("mu", m1_alpha + m1_beta * m1_age_c, dims="obs")

        # ===== LIKELIHOOD =====
        # Link to actual observed heights (for posterior sampling)
        # For prior predictive, don't include "observed=parameter"
        height = pm.Normal("height", mu=m1_mu, sigma=m1_sigma, dims="obs")

        # ===== PRIOR PREDICTIVE =====
        m1_idata_prior = pm.sample_prior_predictive()
    return (m1_idata_prior,)


@app.cell
def _(az, df_adults, m1_idata_prior, plt):
    # Plotting
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 5))

    m1_prior_mu = m1_idata_prior["prior"]["mu"].stack(sample=("chain", "draw")).values

    _ax[0].plot(
        df_adults["age"],
        m1_prior_mu,
        c="steelblue", alpha=0.2, linewidth=2
    )
    _ax[0].set_title("Prior Regression Lines")
    _ax[0].set_xlabel("Age")
    _ax[0].set_ylabel("Height (cm)")

    # Extract prior predictive samples for h
    m1_prior_height = m1_idata_prior["prior"]["height"].stack(sample=("chain", "draw")).values
    az.plot_kde(m1_prior_height, ax=_ax[1])
    _ax[1].set_title("Prior Predictive Distribution of h")
    _ax[1].set_xlabel("Height (cm)")
    _ax[1].set_yticks([])

    plt.gca()
    return


if __name__ == "__main__":
    app.run()
