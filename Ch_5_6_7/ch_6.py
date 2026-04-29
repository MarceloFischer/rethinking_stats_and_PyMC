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

__generated_with = "0.23.3"
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
        alt,
        az,
        mo,
        np,
        pl,
        plt,
        pm,
        remove_period_col_name,
        rng,
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
def _(alt, milk_data, mo):
    _chart = (
        alt.Chart(milk_data)
        .mark_circle(size=100, opacity=0.7)
        .encode(
            x=alt.X(
                "perc_lactose:Q",
                title="Percentage Lactose",
                axis=alt.Axis(labelFontSize=12, titleFontSize=14, gridOpacity=0.3),
            ),
            y=alt.Y(
                "perc_fat:Q",
                title="Percentage Fat",
                axis=alt.Axis(labelFontSize=12, titleFontSize=14, gridOpacity=0.3),
            ),
            color=alt.Color(
                "clade:N",
                title="Clade",
                scale=alt.Scale(scheme="tableau10"),
                legend=alt.Legend(
                    titleFontSize=13, labelFontSize=11, symbolSize=80, orient="right"
                ),
            ),
        )
        .properties(
            width="container",
            height=400,
            title={
                "text": "Relationship between Lactose and Fat by Clade",
                "fontSize": 16,
                "fontWeight": "bold",
                "subtitle": "Each point represents a mammal species",
                "subtitleFontSize": 12,
                "subtitleColor": "#666666",
                "offset": 10,
            },
        )
        .configure_view(strokeWidth=0)
        .interactive()
    )

    mo.ui.altair_chart(_chart)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multicollinearity
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Post-treatment Bias
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fungus example page 175
    """)
    return


@app.cell
def _(np, pl, rng):
    _n = 100
    _h0 = rng.normal(10, 2, size=_n)
    _treatment = np.repeat([0, 1], _n / 2)
    _fungus = rng.binomial(n=1, p=0.5 - _treatment * 0.4, size=_n)
    _h1 = _h0 + rng.normal(5 - 3 * _fungus, size=_n)

    _d = pl.DataFrame({"h0": _h0, "h1": _h1, "treatment": _treatment, "funfus": _fungus})

    _d
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We know that the plants at time $t = 1$ should be taller than at time $t = 0$, whatever scale they are measured on. So if we put the parameters on a scale of proportion of height at time $t = 0$, rather than on the absolute scale of the data, we can set the priors more easily. To make this simpler, let’s focus right now only on the height variables, ignoring the predictor variables. We might have a linear model like:

    $$ h_{1,i} \sim \text{Normal}(\mu_i, \sigma) $$
    $$ \mu_i = h_{0,i} \times p $$

    where h0,i is plant i’s height at time $t = 0$, h1,i is its height at time $t = 1$, and p is a parameter measuring the proportion of h0,i that h1,i is. More precisely, $p=\frac{h_{1,i}}{h_{0,i}}$ . If p = 1, the plant hasn’t changed at all from time $t = 0$ to time $t = 1$. If p = 2, it has doubled in height. So if we center our prior for p on 1, that implies an expectation of no change in height. That is less than we know. But we should allow p to be less than 1, in case the experiment goes horribly wrong and we kill all the plants. We also have to ensure that p > 0, because it is a proportion. A Log-Normal distribution, because it is always positive. If we use p ∼ Log-Normal(0, 0.25) (draw garph to see what it looks like).
    """)
    return


@app.cell
def _(az, rng):
    (
        az.summary(rng.lognormal(0, 0.25, 1000), kind="stats"),
        az.plot_dist(rng.lognormal(0, 0.25, 1000)),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So this prior expects anything from 40% shrinkage up to 40% growth. Now to include the treatment and fungus variables. We’ll include both of them, following the notion that we’d like to measure the impact of both the treatment and the fungus itself. The parameters for these variables will also be on the pro- portion scale. They will be changes in proportion growth. So we’re going to make a linear model of p:

    \begin{aligned}
    h_{1,i} &\sim \mathcal{N}(\mu_i,\sigma)\\
    \mu_i &= h_{0,i} \times p\\
    p &= \alpha + \beta_T T_i + \beta_F F_i\\
    \alpha &\sim \mathrm{Log\text{-}Normal}(0,0.25)\\
    \beta_T &\sim \mathcal{N}(0,0.5)\\
    \beta_F &\sim \mathcal{N}(0,0.5)\\
    \sigma &\sim \mathrm{Exponential}(1)
    \end{aligned}

    The proportion of growth p is now a function of the predictor variables. The priors on the slopes are almost certainly too flat. They place 95% of the prior mass between −1 (100% reduction) and +1 (100% increase) and two-thirds of the prior mass between −0.5 and +0.5.
    """)
    return


@app.cell
def _(az, np, pm, rng):
    def run_fungus_model() -> az.InferenceData:
        n = 1000
        h0 = rng.normal(10, 2, size=n)
        treatment = np.repeat([0, 1], n / 2)
        fungus = rng.binomial(n=1, p=0.5 - treatment * 0.4, size=n)
        h1 = h0 + rng.normal(5 - 3 * fungus, size=n)

        with pm.Model() as model:
            a = pm.Normal("a", 0, 0.2)
            bt = pm.Normal("treatment", 0, 0.5)
            bf = pm.Normal("fungus", 0, 0.5)
            sigma = pm.Exponential("sigma", 1)

            p = a + bt*treatment + bf*fungus
            mu = h0*p

            h1_inference = pm.Normal("h1", mu, sigma, observed=h1)

            idata = pm.sample()

        return idata

    return (run_fungus_model,)


@app.cell
def _(run_fungus_model):
    fungus_model_wrong = run_fungus_model()
    return (fungus_model_wrong,)


@app.cell
def _(az, fungus_model_wrong):
    az.summary(fungus_model_wrong, kind="stats")
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


@app.cell(column=2)
def _():
    return


if __name__ == "__main__":
    app.run()
