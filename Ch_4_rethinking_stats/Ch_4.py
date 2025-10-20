import marimo

__generated_with = "0.17.0"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""## Imports""")
    return


@app.cell
def _():
    import numpy as np
    import scipy.stats as stats

    import polars as pl
    import matplotlib.pyplot as plt
    import altair as alt
    import arviz as az

    import pymc as pm

    import marimo as mo

    RANDOM_SEED = 42
    rng = np.random.default_rng(RANDOM_SEED)
    az.style.use("arviz-darkgrid")
    az.rcParams["stats.ci_prob"] = 0.89  # sets default credible interval used by arviz
    return az, mo, np, pl, plt, stats


@app.cell
def _(pl):
    data = pl.read_csv("/home/marcelo/git/rethinking_stats_and_PyMC/data/Howell1.csv", separator=";")
    data = data.filter(pl.col("age") >= 18)

    data
    return


@app.cell
def _(np, plt, stats):
    _x = np.linspace(100, 250, 100)
    plt.plot(_x, stats.norm.pdf(_x, 178, 20))

    _x = np.linspace(-10, 60, 100)
    plt.plot(_x, stats.uniform.pdf(_x, 0, 50))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \[
    h_i \sim \text{Normal}(\mu, \sigma)
    \]

    \[
    \mu \sim \text{Normal}(178, 20)
    \]

    \[
    \sigma \sim \text{Uniform}(0, 50)
    \]
    """
    )
    return


@app.cell
def _(az, stats):
    # We can use our model to check if priors make sense before actually feeding any data in.

    _n = 1000
    _mu = stats.norm.rvs(loc=178, scale=20, size=_n)
    _sigma = stats.uniform.rvs(loc=0, scale=1, size=_n)
    _prior_h = stats.norm.rvs(loc=_mu, scale=_sigma, size=_n)

    az.plot_kde(_prior_h)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
