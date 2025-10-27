import marimo

__generated_with = "0.17.0"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""## Imports and Constants""")
    return


@app.cell
def _():
    import numpy as np
    import scipy.stats as stats

    import polars as pl
    import pandas as pd
    import matplotlib.pyplot as plt
    import altair as alt
    import arviz as az

    import pymc as pm

    import marimo as mo

    RANDOM_SEED = 42
    rng = np.random.default_rng(RANDOM_SEED)
    az.style.use("arviz-darkgrid")
    az.rcParams["stats.ci_prob"] = 0.89  # sets default credible interval used by arviz
    return az, mo, np, pl, plt, pm, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Data Cleaning""")
    return


@app.cell
def _(pl):
    raw_data = pl.read_csv("/home/marcelo/git/rethinking_stats_and_PyMC/data/Howell1.csv", separator=";")


    def filter_data(df: pl.DataFrame):
        return df.filter(pl.col("age") >= 18)
    return filter_data, raw_data


@app.cell
def _(filter_data, raw_data):
    data = raw_data.pipe(filter_data)

    data
    return (data,)


@app.cell
def _(data):
    MEAN_W = data.select("weight").mean().item()
    return (MEAN_W,)


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
    ## Our Model

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
    # note that the code is written in opposite order as the mathematical definitions

    _n = 1000
    _mu = stats.norm.rvs(loc=178, scale=20, size=_n)
    _sigma = stats.uniform.rvs(loc=0, scale=1, size=_n)
    _prior_h = stats.norm.rvs(loc=_mu, scale=_sigma, size=_n)

    az.plot_kde(_prior_h)
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""## Quadratic Approximation""")
    return


@app.cell
def _(data, pm):
    with pm.Model() as m4_1:
        # initival can be given to tell the algo where to start looking for the peak
        mu = pm.Normal("mu", mu=178, sigma=20, initval=data.select("height").mean().item())
        sigma = pm.Uniform("sigma", lower=0, upper=50, initval=data.select("height").std().item())
        height = pm.Normal("height", mu=mu, sigma=sigma, observed=data.select("height"))
        idata_4_1 = pm.sample(1_000, tune=1_000)
    return (idata_4_1,)


@app.cell
def _(az, idata_4_1):
    az.plot_trace(idata_4_1), az.summary(idata_4_1, round_to=2, kind="stats")
    return


@app.cell
def _(az, idata_4_1):
    idata_df = az.extract(idata_4_1).to_dataframe()
    idata_df, idata_df.cov(), idata_df.corr()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Linear Prediction""")
    return


@app.cell
def _(data, plt):
    plt.plot(data.select("height"), data.select("weight"), ".")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    h_i &\sim \text{Normal}(\mu_i, \sigma) & [\text{likelihood}] \\
    \mu_i &= \alpha + \beta w_i \quad  (\beta(x_i - \bar{x})) & [\text{linear model}] \\
    \alpha &\sim \text{Normal}(178, 20) & [\alpha \text{ prior}] \\
    \beta &\sim \text{Normal}(0, 10) & [\beta \text{ prior}] \\
    \sigma &\sim \text{Uniform}(0, 50) & [\sigma \text{ prior}]
    \end{align*}

    The golem is assuming (by using a Gaussian) that the relationship to construct the mean is additive. We do grown a bit every year/day/minute and the weight, on avg, increases a bit as well.

    As the image below show, this does not make a good model as it allows or negative slopes and also slopes that are clearly not resoanble.

    Let's restrict beta to only positive values as: 

    \begin{align*}
    h_i &\sim \text{Normal}(\mu_i, \sigma) & [\text{likelihood}] \\
    \mu_i &= \alpha + \beta w_i & [\text{linear model}] \\
    \alpha &\sim \text{Normal}(178, 20) & [\alpha \text{ prior}] \\
    \beta &\sim \text{Log-Normal}(0, 10) & [\beta \text{ prior}] \\
    \sigma &\sim \text{Uniform}(0, 50) & [\sigma \text{ prior}]
    \end{align*}

    Another way to is to expect that the slope would be between 0 and 1. Weight is in kg and height is in cm. Usually, weight is less than height.

    \begin{align*}
    h_i &\sim \text{Normal}(\mu_i, \sigma) & [\text{likelihood}] \\
    \mu_i &= \alpha + \beta w_i & [\text{linear model}] \\
    \alpha &\sim \text{Normal}(178, 20) & [\alpha \text{ prior}] \\
    \beta &\sim \text{Uniform}(0, 1) & [\beta \text{ prior}] \\
    \sigma &\sim \text{Uniform}(0, 50) & [\sigma \text{ prior}]
    \end{align*}
    """
    )
    return


@app.cell
def _(MEAN_W, data, np, plt, stats):
    # to see the effects of beta (prior) and understand what it entails, we need to simulate several heights using beta

    _N = 100  # 100 lines
    _alpha = stats.norm.rvs(loc=178, scale=20, size=_N)
    _, _ax = plt.subplots(1, 3, sharey=True, figsize=(10, 5))
    _x = np.linspace(data.select("weight").min().item(), data.select("weight").max().item(), _N)

    _beta = stats.norm.rvs(loc=0, scale=10, size=_N)
    for i in range(_N):
        _ax[0].plot(_x, _alpha[i] + _beta[i] * (_x - MEAN_W), "k", alpha=0.2)
        _ax[0].set_xlim(data.select("weight").min().item(), data.select("weight").max().item())
        _ax[0].set_ylim(-100, 400)
        _ax[0].axhline(0, c="k", ls="--")
        _ax[0].axhline(272, c="k")
        _ax[0].set_xlabel("weight")
        _ax[0].set_ylabel("height")

    _beta = stats.lognorm.rvs(s=1, scale=1, size=_N)
    for i in range(_N):
        _ax[1].plot(_x, _alpha[i] + _beta[i] * (_x - MEAN_W), "k", alpha=0.2)
        _ax[1].set_xlim(data.select("weight").min().item(), data.select("weight").max().item())
        _ax[1].set_ylim(-100, 400)
        _ax[1].axhline(0, c="k", ls="--", label="embryo")
        _ax[1].axhline(272, c="k")
        _ax[1].set_xlabel("weight")
        _ax[1].text(x=35, y=282, s="World's tallest person (272cm)")
        _ax[1].text(x=35, y=-25, s="Embryo")

    _beta = stats.uniform.rvs(loc=0, scale=1, size=_N)
    for i in range(_N):
        _ax[2].plot(_x, _alpha[i] + _beta[i] * (_x - MEAN_W), "k", alpha=0.2)
        _ax[2].set_xlim(data.select("weight").min().item(), data.select("weight").max().item())
        _ax[2].set_ylim(-100, 400)
        _ax[2].axhline(0, c="k", ls="--", label="embryo")
        _ax[2].axhline(272, c="k")
        _ax[2].set_xlabel("weight")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    h_i &\sim \text{Normal}(\mu_i, \sigma) & [\text{likelihood}] \\
    \mu_i &= \alpha + \beta(x_i - \bar{x}) & [\text{linear model}] \\
    \alpha &\sim \text{Normal}(178, 20) & [\alpha \text{ prior}] \\
    \beta &\sim \text{Log-Normal}(0, 1) & [\beta \text{ prior}] \\
    \sigma &\sim \text{Uniform}(0, 50) & [\sigma \text{ prior}]
    \end{align*}

    The log-normal prior for beta ensures that the slopes are positive and within a reasonable range.
    """
    )
    return


@app.cell
def _(MEAN_W, data, pm):
    with pm.Model() as m4_3:
        alpha_4_3 = pm.Normal("alpha", mu=178, sigma=20)
        beta_4_3 = pm.Lognormal("beta", mu=0, sigma=1)
        sigma_4_3 = pm.Uniform("sigma", 0, 50)
        mu_4_3 = alpha_4_3 + beta_4_3 * (data.get_column("weight").to_numpy() - MEAN_W)
        # CANNOT USE data.select('height') AS IT RETURNS A DATAFRAME. NEEDS TO BE A SERIES. RESULTS ARE COMPLETELY DIFFERENT
        # try type(data.select("height")), type(data.get_column("height"))
        height_4_3 = pm.Normal("height", mu=mu_4_3, sigma=sigma_4_3, observed=data.get_column("height"))
        idata_4_3 = pm.sample(1000, tune=1000)
    return (idata_4_3,)


@app.cell
def _(az, idata_4_3):
    _idata_df = az.extract(idata_4_3).to_dataframe()
    az.summary(idata_4_3, kind="stats"), _idata_df.cov(), _idata_df.corr()
    return


@app.cell
def _(MEAN_W, data, idata_4_3, plt):
    plt.plot(data["weight"], data["height"], ".")

    plt.plot(
        data["weight"],
        idata_4_3.posterior["alpha"].mean().item() + idata_4_3.posterior["beta"].mean().item() * (data["weight"] - MEAN_W),
    )

    plt.xlabel("Weight")
    plt.ylabel("Height")
    plt.show()
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
