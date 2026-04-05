import marimo

__generated_with = "0.22.4"
app = marimo.App(width="columns")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    import operator
    import altair as alt
    from typing import Optional
    import arviz as az
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import polars as pl
    import pymc as pm
    import scipy.stats as stats
    from pathlib import Path
    from wigglystuff import EdgeDraw

    RANDOM_SEED = 1523
    rng = np.random.default_rng(RANDOM_SEED)
    plt.style.use("fivethirtyeight")
    # Set default figure size to 14 inches wide by 5 inches tall
    plt.rcParams["figure.figsize"] = (14, 5)
    # You can also set the DPI (dots per inch) for crisper images
    plt.rcParams["figure.dpi"] = 100
    # Make the layout "tight" by default so labels don't overlap
    plt.rcParams["figure.autolayout"] = True
    # sets default credible interval used by arviz
    az.rcParams["stats.ci_prob"] = 0.89
    return Path, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Raw Data
    """)
    return


@app.cell
def _(Path, pl):
    #############
    # Constants #
    #############
    MILK_PATH = Path(__file__).parent.parent / "data" / "milk.csv"

    raw_milk_data = pl.read_csv(MILK_PATH, separator=";")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
