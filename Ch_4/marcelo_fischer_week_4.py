import marimo

__generated_with = "0.20.4"
app = marimo.App(width="columns")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Don't know how to do it
    """)
    return


@app.cell
def _():
    import arviz as az
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    RANDOM_SEED = 1523
    rng = np.random.default_rng(RANDOM_SEED)

    az.style.use("arviz-darkgrid")
    return mo, pl, plt


@app.cell
def _(pl):
    df = pl.read_csv(
        "/home/marcelo/git/rethinking_stats_and_PyMC/data/Howell1.csv", separator=";"
    )


    def filter_age(df: pl.DataFrame, age: int) -> pl.DataFrame:
        return df.filter(pl.col("age") < age)


    data = df.pipe(filter_age, age=13)

    data
    return (data,)


@app.cell
def _(data, plt):
    scatter = plt.scatter(data["height"], data["weight"], c=data["male"], cmap="RdYlGn")
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.title("Weight vs Height (Children < 13)")
    plt.legend(handles=scatter.legend_elements()[0], labels=["Female", "Male"])
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
