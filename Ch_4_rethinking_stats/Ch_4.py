import marimo

__generated_with = "0.17.0"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""## Imports""")
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import arviz as az
    import altair as alt
    import pymc as pm
    import scipy.stats as stats
    import numpy as np
    return mo, pl


@app.cell
def _(pl):
    data = pl.read_csv("/home/marcelo/git/rethinking_stats_and_PyMC/data/Howell1.csv", separator=";")
    data
    return


@app.cell(column=1)
def _():
    return


if __name__ == "__main__":
    app.run()
