import marimo

__generated_with = "0.23.4"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import arviz as az
    import preliz as pz
    import matplotlib.pyplot as plt
    import marimo as moe
    import numpy as np
    import polars as pl
    import pymc as pm
    import scipy.stats as stats
    from pathlib import Path
    from wigglystuff import EdgeDraw

    RANDOM_SEED = 1523
    rng = np.random.default_rng(RANDOM_SEED)

    plt.style.use("fivethirtyeight")
    # Set default figure size to 10 inches wide by 6 inches tall
    plt.rcParams["figure.figsize"] = (14, 5)
    # Make the layout "tight" by default so labels don't overlap
    plt.rcParams["figure.autolayout"] = True
    # sets default credible interval used by arviz
    az.rcParams["stats.ci_prob"] = 0.89
    return Path, alt, az, mo, pl, plt, pm, pz, rng


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Constants
    """)
    return


@app.cell
def _(Path, pl):
    TIPS_PATH = Path(__file__).parent.parent / "data" / "tips.csv"

    tips = pl.read_csv(TIPS_PATH)

    tips
    return (tips,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Distribution Overview
    """)
    return


@app.cell
def _(mo):
    mu_slider = mo.ui.slider(0, 10, step=0.1, show_value=True, label="mu")
    sigma_slider = mo.ui.slider(1, 10, step=0.1, show_value=True, label="sigma")
    nu_slider = mo.ui.slider(0.1, 20, step=0.5, show_value=True, label="nu")
    return mu_slider, nu_slider, sigma_slider


@app.cell
def _(mo, mu_slider, nu_slider, pz, sigma_slider):
    mean, var, skewness, kurtosis = pz.StudentT(
        mu=mu_slider.value, sigma=sigma_slider.value, nu=nu_slider.value
    ).moments()

    mo.vstack(
        [
            mo.hstack([mu_slider, sigma_slider, nu_slider]),
            mo.md(
                f"**Mean:** {mean:.2f} | **Variance:** {var:.2f} | **Skewness:** {skewness:.2f} | **Kurtosis:** {kurtosis:.2f}"
            ),
            pz.StudentT(
                mu=mu_slider.value, sigma=sigma_slider.value, nu=nu_slider.value
            ).plot_pdf(),
            # pz.Normal(0, 1).plot_pdf()
        ],
        align="stretch",
        gap=1,
    )
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    # Tips Example
    """)
    return


@app.cell
def _(az, pl, plt, tips):
    _tips_dict = {
        day: tips.filter(pl.col("day") == day)["tip"].to_numpy()
        for day in tips["day"].unique().to_list()
    }

    az.plot_forest(
        _tips_dict,
        kind="ridgeplot",
        ridgeplot_truncate=False,
        ridgeplot_quantiles=[0.25, 0.5, 0.75],
        ridgeplot_overlap=2.2,
    )

    plt.title("Distribution of Tips per Weekday")
    plt.xlabel("Tip Amount")
    plt.gca()
    return


@app.cell
def _(alt, tips):
    ridge_plot = (
        alt.Chart(tips)
        .transform_density("tip", as_=["tip", "density"], groupby=["day"])
        .mark_area(
            interpolate="monotone", fillOpacity=0.8, stroke="lightgray", strokeWidth=0.5
        )
        .encode(
            alt.X("tip:Q", title="Tip Amount ($)"),
            alt.Y("density:Q", stack=None, title=None, axis=None),
            alt.Fill("day:N", legend=None, scale=alt.Scale(scheme="viridis")),
            tooltip=[
                alt.Tooltip("day:N", title="Day"),
                alt.Tooltip("tip:Q", title="Tip", format="$.2f"),
                alt.Tooltip("density:Q", title="Density", format=".4f"),
            ],
        )
        .properties(height=90, width="container")
        .facet(
            row=alt.Row(
                "day:N",
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="right"),
                sort=["Thur", "Fri", "Sat", "Sun"],
            )
        )
        .configure_facet(spacing=0)
        .configure_view(stroke=None)
    )

    ridge_plot
    return


@app.cell
def _(pl, tips):
    def create_group_vars_tips(dataf: pl.DataFrame = tips):
        """
        Creates a mapping of unique days and their corresponding integer indices from the tips dataset.

        Args:
            dataf: A polars DataFrame containing a 'day' column. Defaults to the global 'tips' dataframe.

        Returns:
            tuple: A tuple containing (days, day_idx) where 'days' is an array of unique day names
                   and 'day_idx' is an array of integer codes mapping each row to a day.
        """
        days = dataf["day"].unique().to_numpy()
        day_idx = dataf["day"].cast(pl.Enum(days)).to_physical().to_numpy()

        return days, day_idx


    days, day_idx = create_group_vars_tips()
    return day_idx, days


@app.cell
def _(day_idx, days, pm, rng, tips):
    def tips_model():
        """
        Constructs and samples from a Bayesian model to estimate tips per weekday.

        The model assumes a Normal likelihood for tip amounts, with separate
        mu and sigma parameters for each day of the week.

        Returns:
            tuple: (model, idata) where 'model' is the PyMC model object
                   and 'idata' is the InferenceData object containing samples.
        """
        coords = {"days": days, "days_flat": days[day_idx]}
        with pm.Model(coords=coords) as model:
            # priors
            mu = pm.Normal("mu", mu=0, sigma=10, dims="days", rng=rng)
            sigma = pm.HalfNormal("sigma", sigma=10, dims="days", rng=rng)

            # likelihood
            obs = pm.Normal(
                "obs",
                mu=mu[day_idx],
                sigma=sigma[day_idx],
                observed=tips["tip"],
                dims="days_flat",
                rng=rng,
            )

            idata = pm.sample(random_seed=rng)

        return model, idata


    day_tip_model, day_tip_idata = tips_model()
    return day_tip_idata, day_tip_model


@app.cell
def _(day_tip_idata):
    day_tip_idata
    return


@app.cell
def _(az, day_tip_idata, day_tip_model, days, plt, pm, rng):
    with day_tip_model:
        day_tip_idata.extend(pm.sample_posterior_predictive(day_tip_idata))

    _, axes = plt.subplots(2, 2)
    az.plot_ppc(
        day_tip_idata,
        num_pp_samples=100,
        coords={"days_flat": days},
        flatten=[],
        ax=axes,
        random_seed=rng,
    )
    return


if __name__ == "__main__":
    app.run()
