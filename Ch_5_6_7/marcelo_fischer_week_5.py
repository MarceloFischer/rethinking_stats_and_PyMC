import marimo

__generated_with = "0.20.4"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    RANDOM_SEED = 1523
    rng = np.random.default_rng(RANDOM_SEED)

    az.style.use("fivethirtyeight")
    return pl, plt


@app.cell
def _(pl):
    SEX = ["women", "men"]

    df = pl.read_csv(
        "/home/marcelo/git/rethinking_stats_and_PyMC/data/Howell1.csv", separator=";"
    )


    def filter_age_lt(df: pl.DataFrame, age: int) -> pl.DataFrame:
        return df.filter(pl.col("age") >= age)


    data = df.pipe(filter_age_lt, age=18)

    data
    return SEX, data


@app.cell
def _(SEX, data, pl, plt):
    def _() -> plt.Axes:
        # Split by the Sex Category
        for i, sex in enumerate(SEX):
            plt.scatter(
                data.filter(pl.col("male") == i)["weight"].to_numpy(),
                data.filter(pl.col("male") == i)["height"].to_numpy(),
                c=f"C{i}",
                label=sex,
            )

        plt.legend()
        plt.xlabel("Height (cm)")
        plt.ylabel("Weight (kg)")
        plt.title("Weight vs Height pre Sex")

        # graph = mo.ui.matplotlib(plt.gca())
        return plt.gca()


    _()
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


if __name__ == "__main__":
    app.run()
