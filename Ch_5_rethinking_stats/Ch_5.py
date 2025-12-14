# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.0.0",
#     "arviz==0.22.0",
#     "matplotlib==3.10.8",
#     "numpy==2.3.5",
#     "openai==2.11.0",
#     "polars==1.36.1",
#     "pymc==5.26.1",
#     "python-lsp-ruff==2.3.0",
#     "python-lsp-server==1.14.0",
#     "ruff==0.14.9",
#     "scipy==1.16.3",
#     "vegafusion==2.0.3",
#     "vl-convert-python==1.8.0",
#     "websockets==15.0.1",
# ]
# ///

import marimo

__generated_with = "0.18.4"
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
    import scipy.stats as stats
    from pathlib import Path

    RANDOM_SEED = 42
    rng = np.random.default_rng(RANDOM_SEED)
    az.style.use("arviz-darkgrid")
    az.rcParams["stats.ci_prob"] = (
        0.89  # sets default credible interval used by arviz
    )
    return Path, az, np, pl, plt, pm


@app.cell
def _(Path, pl):
    data_file_path = Path(__file__).parent.parent / "data" / "WaffleDivorce.csv"

    data = pl.read_csv(data_file_path, separator=";")

    # add standardised columns
    data = data.with_columns(
        [
            ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(
                f"{col}_std"
            )
            for col in ["Divorce", "MedianAgeMarriage", "Marriage"]
        ]
    )

    data
    return (data,)


@app.cell
def _(data, plt):
    _, _ax = plt.subplots(1, 2, figsize=(14, 5))

    _ax[0].scatter(data["Marriage"], data["Divorce"])
    _ax[0].set_xlabel("Marriage Rate")
    _ax[0].set_ylabel("Divorce Rate")
    _ax[0].set_title("Divorce vs Marriage Rate")

    _ax[1].scatter(data["MedianAgeMarriage"], data["Divorce"])
    _ax[1].set_xlabel("Median Age Marriage")
    _ax[1].set_ylabel("Diverce Rate")
    _ax[1].set_title("Marriage vs Median Age of Marriage")
    return


@app.cell
def _(data, np, plt, pm):
    with pm.Model() as m5_1:
        _a = pm.Normal("a", 0, 0.2)
        _bA = pm.Normal("bA", 0, 0.5)
        _sigma = pm.Exponential("sigma", 1)
        _mu = _a + _bA * data.get_column("MedianAgeMarriage_std").to_numpy()

        _div_rate = pm.Normal(
            "divorce_rate",
            mu=_mu,
            sigma=_sigma,
            observed=data["Divorce_std"].to_numpy(),
        )

        _prior_samples = pm.sample_prior_predictive()

        m5_1_idata = pm.sample(1000)

    _x = np.linspace(
        data["MedianAgeMarriage_std"].min(), data["MedianAgeMarriage_std"].max()
    )

    _a_plot = _prior_samples["prior"]["a"].to_numpy().flatten()
    _bA_plot = _prior_samples["prior"]["bA"].to_numpy().flatten()
    _mu_plot = _a_plot[:, None] + _bA_plot[:, None] * _x

    plt.plot(_x, _mu_plot.T, c="g", alpha=0.4)
    return (m5_1_idata,)


@app.cell
def _(data):
    data.select(["MedianAgeMarriage", "Divorce", "Marriage"]).corr()
    return


@app.cell
def _(az, data, m5_1_idata, np, plt):
    _a_samples = (
        m5_1_idata["posterior"]["a"].to_numpy().flatten()
    )  # shape is (4 * n_samples,)
    _bA_samples = (
        m5_1_idata["posterior"]["bA"].to_numpy().flatten()
    )  # shape is (4 * n_samples,)
    _x = np.linspace(
        data["MedianAgeMarriage_std"].min(),
        data["MedianAgeMarriage_std"].max(),
        data.shape[0],
    )

    # Raw data
    plt.scatter(data["MedianAgeMarriage"], data["Divorce"], c="red", label="data")

    # MAP - convert back to original scale
    _divorce_mean = data["Divorce"].mean()
    _divorce_std = data["Divorce"].std()

    _map_line = (
        _a_samples.mean()
        + _bA_samples.mean() * data["MedianAgeMarriage_std"].to_numpy()
    ) * _divorce_std + _divorce_mean

    plt.plot(
        data["MedianAgeMarriage"],
        _map_line,
        c="green",
        lw=3,
        label="MAP regression line",
    )

    # 2. Manually calculate mu for each posterior sample in original scale
    # Broadcasting: (n_samples, 1) + (n_samples, 1) * (n_points,)
    # Result: (n_samples, n_points)
    _mu_std = (
        _a_samples[:, np.newaxis]
        + _bA_samples[:, np.newaxis] * data["MedianAgeMarriage_std"].to_numpy()
    )
    _mu = _mu_std * _divorce_std + _divorce_mean

    # 89% HDI mean
    _mu_hdi = az.hdi(_mu, hdi_prob=0.89)
    az.plot_hdi(
        x=data["MedianAgeMarriage"].to_numpy(), hdi_data=_mu_hdi, color="green"
    )

    plt.xlabel("MedianAgeMarriage")
    plt.ylabel("Divorce")
    plt.legend()
    plt.gca()
    return


@app.cell
def _(az, m5_1_idata):
    az.summary(m5_1_idata)
    return


if __name__ == "__main__":
    app.run()
