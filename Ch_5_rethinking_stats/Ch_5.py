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

__generated_with = "0.20.4"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import operator
    import altair as alt
    import arviz as az
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import pymc as pm
    import scipy.stats as stats
    from pathlib import Path

    RANDOM_SEED = 1523
    rng = np.random.default_rng(RANDOM_SEED)
    plt.style.use("fivethirtyeight")

    az.rcParams["stats.ci_prob"] = 0.89  # sets default credible interval used by arviz
    return Path, az, mo, np, operator, pl, plt, pm, rng


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Constants
    """)
    return


@app.cell
def _(Path, pl):
    #############
    # Constants #
    #############
    WAFFLE_PATH = Path(__file__).parent.parent / "data" / "WaffleDivorce.csv"
    HOWELL_PATH = Path(__file__).parent.parent / "data" / "Howell1.csv"

    # women=0 and men=1 in the dataset
    SEX = ["women", "men"]

    raw_waffle_data = pl.read_csv(WAFFLE_PATH, separator=";")
    raw_howell_data = pl.read_csv(HOWELL_PATH, separator=";")
    return SEX, raw_howell_data, raw_waffle_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Helper Functions
    """)
    return


@app.cell
def _(operator, pl):
    def cols_to_lowercase(dataf: pl.DataFrame) -> pl.DataFrame:
        return dataf.rename({col: col[0].lower() + col[1:] for col in dataf.columns})


    def std_cols_of_interest(dataf: pl.DataFrame) -> pl.DataFrame:
        # add standardised columns
        return dataf.with_columns(
            [
                ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(f"{col}_std")
                for col in ["divorce", "medianAgeMarriage", "marriage"]
            ]
        )


    def filter_by_comparison(
        df: pl.DataFrame, col_name: str, value: float, op_str: str
    ) -> pl.DataFrame:
        # Map string labels to python operators
        ops = {
            ">": operator.gt,
            ">=": operator.ge,
            "<": operator.lt,
            "<=": operator.le,
            "==": operator.eq,
        }

        # Use the operator to create a Polars expression
        # Equivalent to: df.filter(pl.col(col_name) >= value)
        return df.filter(ops[op_str](pl.col(col_name), value))

    return cols_to_lowercase, filter_by_comparison, std_cols_of_interest


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Prep.
    """)
    return


@app.cell
def _(
    cols_to_lowercase,
    filter_by_comparison,
    raw_howell_data,
    raw_waffle_data,
    std_cols_of_interest,
):
    waffle_data = raw_waffle_data.pipe(cols_to_lowercase).pipe(std_cols_of_interest)

    howell_adults = raw_howell_data.pipe(filter_by_comparison, "age", 18, ">=")

    howell_children = raw_howell_data.pipe(filter_by_comparison, "age", 13, "<=")

    # raw_howell_data, raw_waffle_data
    return howell_adults, waffle_data


@app.cell
def _(pl, plt, raw_howell_data):
    def _() -> plt.Axes:
        # 1. Create the boolean expression/mask
        is_adult = pl.col("age") >= 18

        # 2. Apply filtering
        adult_howell = raw_howell_data.filter(is_adult)
        child_howell = raw_howell_data.filter(~is_adult)

        plt.scatter(adult_howell["height"], adult_howell["weight"], label="over 18")
        plt.scatter(child_howell["height"], child_howell["weight"], label="under 18")

        plt.xlabel("Height (cm)")
        plt.ylabel("Weight (kg)")

        plt.legend()

        return plt.gca()


    # _()
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


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lecture
    """)
    return


@app.cell
def _(SEX, az, howell_adults, pl, plt):
    def plot_howell_HW_dist():
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        plt.sca(axs[0])
        for sex_idx, label in enumerate(SEX):
            plt.scatter(
                howell_adults.filter(pl.col("male") == sex_idx)["height"],
                howell_adults.filter(pl.col("male") == sex_idx)["weight"],
                label=label,
                alpha=0.7,
                s=50,
            )
        plt.xlabel("Height (cm)")
        plt.ylabel("Weight (kg)")
        plt.title("Weight vs Height")
        plt.legend(title="Sex")

        for var_idx, col in enumerate(["height", "weight"]):
            plt.sca(axs[var_idx + 1])
            for sex_idx2, sex in enumerate(SEX):
                az.plot_dist(
                    howell_adults.filter(howell_adults["male"] == sex_idx2)[col],
                    label=sex,
                    color=f"C{sex_idx2}",
                )
            plt.ylabel(f"{col}".capitalize())
            plt.title(f"Dist. of {col.capitalize()} Split by Sex")
        return plt.gca()


    plot_howell_HW_dist()
    return


@app.cell
def _(np, rng):
    a = np.array([1] * 5 + [0] * 3)
    b = np.array([0, 5])
    a, b, b[a]

    b, rng.binomial(1, 0.5, 4), b[rng.binomial(1, 0.5, 4)], b[a]
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chapter Notes
    """)
    return


@app.cell
def _(plt, waffle_data):
    _, _ax = plt.subplots(1, 2, figsize=(14, 5))

    _ax[0].scatter(waffle_data["marriage"], waffle_data["divorce"])
    _ax[0].set_xlabel("Marriage Rate")
    _ax[0].set_ylabel("Divorce Rate")
    _ax[0].set_title("Divorce vs Marriage Rate")

    _ax[1].scatter(waffle_data["medianAgeMarriage"], waffle_data["divorce"])
    _ax[1].set_xlabel("Median Age Marriage")
    _ax[1].set_ylabel("Divorce Rate")
    _ax[1].set_title("Marriage vs Median Age of Marriage")
    return


@app.cell
def _(np, pm, waffle_data):
    def m5_1(draws: int = 50):
        with pm.Model() as m5_1:
            a = pm.Normal("a", 0, 0.2)
            bA = pm.Normal("bA", 0, 0.5)
            sigma = pm.Exponential("sigma", 1)
            mu = a + bA * waffle_data.get_column("medianAgeMarriage_std").to_numpy()

            div_rate = pm.Normal(
                "divorce_rate",
                mu=mu,
                sigma=sigma,
                observed=waffle_data["divorce_std"].to_numpy(),
            )

            prior_samples = pm.sample_prior_predictive(draws)

            idata = pm.sample(draws)

        x = np.linspace(
            waffle_data["medianAgeMarriage_std"].min(), waffle_data["medianAgeMarriage_std"].max()
        )

        return prior_samples, idata, x


    # m5_1_prior, m5_1_idata, m5_1_x = m5_1(draws=50)

    # a_plot = m5_1_prior["prior"]["a"].to_numpy().flatten()
    # bA_plot = m5_1_prior["prior"]["bA"].to_numpy().flatten()
    # mu_plot = a_plot[:, None] + bA_plot[:, None] * m5_1_x

    # plt.plot(m5_1_x, mu_plot.T, c="g", alpha=0.4)
    return


@app.cell
def _(waffle_data):
    waffle_data.select(["medianAgeMarriage", "divorce", "marriage"]).corr()
    return


@app.cell
def _(az, m5_1_idata, np, plt, waffle_data):
    _a_samples = (
        m5_1_idata["posterior"]["a"].to_numpy().flatten()
    )  # shape is (4 * n_samples,)
    _bA_samples = (
        m5_1_idata["posterior"]["bA"].to_numpy().flatten()
    )  # shape is (4 * n_samples,)
    _x = np.linspace(
        waffle_data["medianAgeMarriage_std"].min(),
        waffle_data["medianAgeMarriage_std"].max(),
        waffle_data.shape[0],
    )

    # Raw data
    plt.scatter(waffle_data["medianAgeMarriage"], waffle_data["divorce"], c="red", label="waffle_data")

    # MAP - convert back to original scale
    _divorce_mean = waffle_data["divorce"].mean()
    _divorce_std = waffle_data["divorce"].std()

    _map_line = (
        _a_samples.mean() + _bA_samples.mean() * waffle_data["medianAgeMarriage_std"].to_numpy()
    ) * _divorce_std + _divorce_mean

    plt.plot(
        waffle_data["medianAgeMarriage"],
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
        + _bA_samples[:, np.newaxis] * waffle_data["medianAgeMarriage_std"].to_numpy()
    )
    _mu = _mu_std * _divorce_std + _divorce_mean

    # 89% HDI mean
    _mu_hdi = az.hdi(_mu, hdi_prob=0.89)
    az.plot_hdi(x=waffle_data["medianAgeMarriage"].to_numpy(), hdi_data=_mu_hdi, color="green")

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
