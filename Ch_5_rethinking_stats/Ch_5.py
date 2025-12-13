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

__generated_with = "0.18.1"
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
    return Path, pl, plt


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
    _ax[0].set_title("Divorce vs Median Age of Marriage")

    _ax[1].scatter(data["MedianAgeMarriage"], data["Divorce"])
    _ax[1].set_xlabel("Median Age Marriage")
    _ax[1].set_ylabel("Diverce Rate")
    _ax[1].set_title("Marriage vs Median Age of Marriage")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
