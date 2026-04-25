# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.0.0",
#     "arviz==0.23.4",
#     "hvplot==0.12.2",
#     "marimo>=0.22.4",
#     "matplotlib==3.10.8",
#     "numpy==2.4.4",
#     "polars==1.39.3",
#     "pyarrow==23.0.1",
#     "pymc==5.28.3",
#     "scipy==1.17.1",
#     "seaborn==0.13.2",
#     "wigglystuff==0.3.1",
# ]
# ///

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    import altair as alt
    import arviz as az
    import hvplot.polars
    import seaborn as sns
    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import polars as pl
    import pymc as pm
    import scipy.stats as stats
    from pathlib import Path
    from wigglystuff import EdgeDraw

    from linear_models_funcs import (
        remove_period_col_name,
        cols_to_lowercase,
        std_cols_of_interest,
        std_log_mass,
        set_dtypes_float64,
        plot_simple_regression_on_chosen_scale,
        plot_linear_regression_prior_predictive,
        plot_counterfactual,
        run_linear_model,
    )

    RANDOM_SEED = 1523
    rng = np.random.default_rng(RANDOM_SEED)
    return (
        Path,
        az,
        mo,
        pl,
        plt,
        remove_period_col_name,
        run_linear_model,
        set_dtypes_float64,
        sns,
        std_cols_of_interest,
        std_log_mass,
    )


@app.cell
def _(az, plt):
    plt.style.use("fivethirtyeight")

    # Set default figure size to 14 inches wide by 5 inches tall
    plt.rcParams["figure.figsize"] = (14, 5)
    # You can also set the DPI (dots per inch) for crisper images
    plt.rcParams["figure.dpi"] = 100
    # Make the layout "tight" by default so labels don't overlap
    plt.rcParams["figure.autolayout"] = True
    # sets default credible interval used by arviz
    az.rcParams["stats.ci_prob"] = 0.89
    return


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
    return (raw_milk_data,)


@app.cell
def _(
    raw_milk_data,
    remove_period_col_name,
    set_dtypes_float64,
    std_cols_of_interest,
    std_log_mass,
):
    # fmt: off
    milk_data = (
        raw_milk_data
        .pipe(remove_period_col_name)
        .pipe(set_dtypes_float64, ["kcal_per_g", "neocortex_perc"])
        .pipe(std_cols_of_interest, ["kcal_per_g", "neocortex_perc", "perc_fat", "perc_lactose"])
        .pipe(std_log_mass)
    )
    # fmt: on
    return (milk_data,)


@app.cell
def _(milk_data):
    milk_data
    return


@app.cell
def _(milk_data):
    milk_data.hvplot.scatter(x="perc_lactose", y="perc_fat", by="clade", aspect=2.5, fontscale=1.5)
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    # Chater Notes
    """)
    return


@app.cell
def _(milk_data, sns):
    sns.pairplot(
        data=milk_data.select(
            ["perc_fat_std", "perc_lactose_std", "kcal_per_g_std", "clade"]
        ).to_pandas(),
        # hue="clade",
        diag_kind="kde",
        kind="reg",
        height=2,
        aspect=1.7,
    )
    return


@app.cell
def _(milk_data, run_linear_model):
    fat_model = run_linear_model(
        predictors=[milk_data["perc_fat_std"]],
        predictors_names=["perc_fat_std"],
        outcome=milk_data["kcal_per_g_std"],
        outcome_name="kcal_per_g_std",
        prior_predictive=False,
        draws=1000,
        alpha=0.2,
        beta=0.5,
    )

    lactose_model = run_linear_model(
        predictors=[milk_data["perc_lactose_std"]],
        predictors_names=["perc_lactose_std"],
        outcome=milk_data["kcal_per_g_std"],
        outcome_name="kcal_per_g_std",
        prior_predictive=False,
        draws=1000,
        alpha=0.2,
        beta=0.5,
    )

    lactose_fat_model = run_linear_model(
        predictors=[milk_data["perc_fat_std"], milk_data["perc_lactose_std"]],
        predictors_names=["perc_fat_std", "perc_lactose_std"],
        outcome=milk_data["kcal_per_g_std"],
        outcome_name="kcal_per_g_std",
        prior_predictive=False,
        draws=1000,
        alpha=0.2,
        beta=0.5,
    )
    return fat_model, lactose_fat_model, lactose_model


@app.cell
def _(az, fat_model, lactose_fat_model, lactose_model):
    # az.summary(fat_model, kind="stats"), az.summary(lactose_model, kind="stats"), az.summary(lactose_fat_model, kind="stats")
    az.plot_forest(
        [fat_model, lactose_model, lactose_fat_model],
        model_names=["fat", "lactose", "both"],
        var_names=["beta"],
        combined=True,
        figsize=(11, 5),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The posterior distributions for perc.fat and perc.lactose are essentially mirror images of one another. The posterior mean of perc.fat is as positive as the mean of perc.lactose is negative. Both are narrow posterior distributions that lie almost entirely on one side or the other of zero. Given the strong association of each predictor with the outcome, we might conclude that both variables are reliable predictors of total energy in milk, across species. The more fat, the more kilocalories in the milk. The more lactose, the fewer kilocalories in milk.

    Now the posterior means of both perc.fat and perc.lactose are closer to zero. And the standard deviations for both parameters are twice as large as in the bivariate models. This is the same statistical phenomenon as in the leg length example. What has happened is that the variables perc.fat and perc.lactose contain much of the same information. They are almost substitutes for one another. As a result, when you include both in a regression, the posterior distribution ends up describing a long ridge of combinations of perc.fat and perc.lactose that are equally plausible.
    """)
    return


@app.cell
def _():
    return


@app.cell(column=2)
def _():
    return


if __name__ == "__main__":
    app.run()
