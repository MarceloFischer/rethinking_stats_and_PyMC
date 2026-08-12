import marimo

__generated_with = "0.23.15"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    from pathlib import Path

    import arviz as az
    import preliz as pz

    import altair as alt
    import seaborn as sns
    import matplotlib.pyplot as plt

    import numpy as np
    import scipy.stats as stats
    import xarray as xr
    import pymc as pm
    import bambi as bmb
    import polars as pl

    #######################################################

    RANDOM_SEED = 1523
    rng = np.random.default_rng(RANDOM_SEED)

    alt.theme.enable('fivethirtyeight')
    plt.style.use("fivethirtyeight")

    # Set default figure size to 14 inches wide by 5 inches tall
    plt.rcParams["figure.figsize"] = (14, 5)
    # You can also set the DPI (dots per inch) for crisper images
    plt.rcParams["figure.dpi"] = 100
    # Make the layout "tight" by default so labels don't overlap
    plt.rcParams["figure.autolayout"] = True
    # sets default credible interval used by arviz
    az.rcParams["stats.ci_prob"] = 0.89
    return Path, alt, az, bmb, mo, np, pl, plt, pm, rng


@app.cell
def _(np, pl):
    _SIZE = 117
    data = pl.DataFrame(
        {
            "y": np.random.normal(size=_SIZE),
            "x": np.random.normal(size=_SIZE),
            "z": np.random.normal(size=_SIZE),
            "g": ["Group A", "Group B", "Group C"] * 39,
        }
    )
    return (data,)


@app.cell
def _(data, pm):
    with pm.Model() as pymc_model:
        α = pm.Normal('α', 0, 1)
        β = pm.Normal('β', 0, 1)
        σ = pm.HalfNormal('σ', 1)

        μ = α + β * data['x'].to_numpy()

        y = pm.Normal('y', μ, σ, observed=data['y'])
        # pymc_model_idata = pm.sample(1_000, random_seed=rng)
    return


@app.cell
def _(bmb, data):
    bmb.Model("y ~ x", data.to_pandas())
    return


@app.cell
def _(bmb, data):
    priors = {"x": bmb.Prior("HalfNormal", sigma=3),
              "sigma": bmb.Prior("Gamma",  mu=1, sigma=2),
              }
 
    bmb.Model("y ~ x", data.to_pandas(), priors=priors)
    return


@app.cell
def _(bmb, data):
    # Partially pooled (hierarchical) model.
    # Allows each group to have it's own mean, but makes sure they all come from a same distribution.
    model_h = bmb.Model("y ~ x + z + (x | g)", data.to_pandas())
    model_h
    return (model_h,)


@app.cell
def _(model_h):
    model_h.build()
    model_h.graph()
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    # Bikes Example - Bambi Style
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bikes Example - Temperature
    """)
    return


@app.cell
def _(Path, pl):
    BIKE_PATH = Path(__file__).parent.parent / "data" / "bikes.csv"

    bikes = pl.read_csv(BIKE_PATH)

    bikes
    return (bikes,)


@app.cell
def _(bikes, bmb, np, pm, rng):
    # Fits a negative binomial model to the bikes data. Regressing "rented" on "temperature"
    def fn_bikes_neg_binom_model():
        coords = {"obs_id": np.arange(len(bikes))}
        with pm.Model(coords=coords) as neg_binom_model:
            temps = pm.Data("temperature", bikes["temperature"].to_numpy(), dims="obs_id")
            # Priors
            α = pm.Normal("α", mu=0, sigma=1)
            β = pm.Normal("β", mu=0, sigma=5)
            σ = pm.HalfNormal("σ", sigma=10)
            # Mean
            μ = pm.Deterministic("μ", pm.math.exp(α + β * temps))
            # Likelihood
            rented = pm.NegativeBinomial("rented", mu=μ, alpha=σ, observed=bikes["rented"], dims="obs_id")

            idata = pm.sample(random_seed=rng)
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)

            return neg_binom_model, idata

    ###################################################

    # "Same" as above. Priors are different.
    model_t = bmb.Model("rented ~ temperature", bikes.to_pandas(), family="negativebinomial")
    # idata_t = model_t.fit(random_seed=rng)
    model_t
    return (model_t,)


@app.cell
def _(mo, model_t):
    mo.stop("idata_t" not in dir(), mo.md("Fit model_t to continue"))
    model_t.graph()
    return


@app.cell
def _(mo, model_t):
    mo.stop("idata_t" not in dir(), mo.md("Fit model_t to continue"))

    model_t.plot_priors(figsize=(12, 5), col_wrap=2)
    return


@app.cell
def _(bikes, bmb, idata_t, mo, model_t, plt):
    mo.stop("idata_t" not in dir(), mo.md("Fit model_t to continue"))

    _fig, _axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5))

    _axes[0].scatter(bikes['temperature'], bikes['rented'], s=10)
    _axes[1].scatter(bikes['temperature'], bikes['rented'], s=10)

    _p1 = bmb.interpret.plot_predictions(
        model_t,
        idata_t,
        conditional="temperature",
        fig_kwargs={
            "xlabel": "Temperature",
            "ylabel": "Rented",
            "title": "Mean"
        }
    ).limit(y=(-100, 1050))

    _p2 = bmb.interpret.plot_predictions(
        model_t,
        idata_t,
        conditional="temperature",
        target='rented',
        fig_kwargs={
            "xlabel": "Temperature",
            "ylabel": "Rented",
            "title": "Predictions"
        }
    )

    _p1.on(_axes[0]).layout(engine="tight").plot()
    _p2.on(_axes[1]).layout(engine="tight").plot().show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bikes Example - Temperature + Humidity
    """)
    return


@app.cell
def _(bikes, bmb):
    model_th = bmb.Model("rented ~ temperature + humidity", bikes.to_pandas(), family="negativebinomial")
    # idata_th = model_th.fit(random_seed=rng)
    return (model_th,)


@app.cell
def _(mo, model_th):
    mo.stop("idata_th" not in dir(), mo.md("Fit model_th to continue"))

    model_th.graph()
    return


@app.cell
def _(bikes, bmb, idata_th, mo, model_th, np):
    mo.stop("idata_th" not in dir(), mo.md("Fit model_th to continue"))

    _conditional = {
        "temperature": np.linspace(bikes["temperature"].min(), bikes["temperature"].max(), 50),
        "humidity": [0.18, 0.5, 0.635, 0.78, 1.0],
    }

    _p = bmb.interpret.plot_predictions(
        model_th,
        idata_th,
        conditional=_conditional,
        # target='rented',
        subplot_kwargs={
            "main": "temperature",
            "group": None,
            "panel": "humidity"
        },
        fig_kwargs={
            "theme": {"figure.figsize": (10, 6)},
            "wrap": 3,
            "title": lambda h: f"humidity = {h}",
        },
    )

    _p.share(x=False).layout(size=(14, 7)).show()
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    # Distributional Models

    Models that also allow parameters to vary. For example, allow the variance also to be a linear function (variable variance)
    """)
    return


@app.cell
def _(Path, pl):
    BABIES_PATH = Path(__file__).parent.parent / "data" / "babies.csv"

    babies = pl.read_csv(BABIES_PATH)
    # Done so that bambi can plot month as a continuous variable (line, and not dots)
    babies = babies.with_columns(month = pl.col("month").cast(pl.Float64))
    babies.head()
    return (babies,)


@app.cell
def _(babies, bmb):
    _formula = bmb.Formula(
        "length ~ np.sqrt(month)",
        "sigma ~ month"
    )

    model_babies = bmb.Model(_formula, babies.to_pandas())
    # idata_babies = model_babies.fit(random_seed=rng)

    # model_babies.plot_priors(col_wrap=2, figsize=(12, 6))
    return (model_babies,)


@app.cell
def _(babies, bmb, idata_babies, mo, model_babies, plt):
    mo.stop("idata_babies" not in dir(), mo.md("Fit model_babies to continue"))

    _fig, _axes = plt.subplots(figsize=(12, 5))

    _axes.scatter(babies['month'], babies['length'], s=10, c='black')

    _p = bmb.interpret.plot_predictions(
            model_babies,
            idata_babies,
            conditional="month",
            target="length",
            fig_kwargs={
                "xlabel": "Month",
                "ylabel": "Length",
                "title": "Predictions"
            },
        prob=[0.65, 0.94]
        )

    _p.on(_axes).layout(engine="tight").plot().show()
    return


@app.cell
def _(idata_babies, mo, model_babies, pl):
    mo.stop("idata_babies" not in dir(), mo.md("Fit model_babies to continue"))

    # Make a prediction for the lenght of a baby with 0.5 months old. This creates a "posterior_predictive" group in the idata object.
    model_babies.predict(idata_babies, kind="response", data=pl.DataFrame({"month":[0.5]}).to_pandas())
    return


@app.cell
def _(az, idata_babies, mo):
    mo.stop("idata_babies" not in dir(), mo.md("Fit model_babies to continue"))

    _ref = 52.5

    _pc = az.plot_dist(idata_babies, group="posterior_predictive")

    _percentile = (idata_babies.posterior_predictive["length"].stack(sample=("chain", "draw")) <= _ref).mean() * 100
    print(f"Percentile of the reference value ({_ref}) is {_percentile.item()}")

    az.add_lines(_pc, _ref)

    _pc.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Models with Categories - Bambi Style
    """)
    return


@app.cell
def _(Path, alt, mo, pl):
    PENGUINS_PATH = Path(__file__).parent.parent / "data" / "penguins.csv"

    penguins = pl.read_csv(PENGUINS_PATH)

    _c = alt.Chart(penguins).mark_point().encode(
        x=alt.X("bill_length:Q", scale=alt.Scale(domain=(2.9, 6.1))),
        y=alt.Y("body_mass:Q", scale=alt.Scale(domain=(2, 6.4))),
        color="species:N",
    ).properties(
        width='container',
        title="Bill Length vs Body Mass by Species"
    )

    mo.ui.altair_chart(_c)
    return (penguins,)


@app.cell
def _(bmb, penguins):
    #  0 + bill_length + species would create one slope and one intercept for each category of species.
    # Each species coefficient would now be the mean body mass of that species (holding bill_length constant)
    model_p = bmb.Model("body_mass ~ bill_length + species", data=penguins.to_pandas(), dropna=True)
    # idata_p = model_p.fit(random_seed=rng)

    model_p
    return (model_p,)


@app.cell
def _(az, idata_p, mo):
    mo.stop("idata_p" not in dir(), mo.md("Fit model_p to continue"))

    az.plot_trace_dist(idata_p)
    return


@app.cell
def _(az, idata_p, mo):
    mo.stop("idata_p" not in dir(), mo.md("Fit model_p to continue"))

    az.plot_forest(idata_p, combined=True, figure_kwargs={'figsize':(12,5)})
    return


@app.cell
def _(bmb, idata_p, mo, model_p, plt):
    mo.stop("idata_p" not in dir(), mo.md("Fit model_p to continue"))

    _fig, _ax = plt.subplots(figsize=(10, 5))

    _p = bmb.interpret.plot_predictions(
        model_p,
        idata_p,
        conditional=['bill_length', 'species'],
        fig_kwargs={
            "xlabel": "Bill Length (mm)",
            "ylabel": "Body Mass (g)",
            "title": "Body Mass Linear Model on Bill Length and Species"
        }
    )

    _p.on(_ax).plot()
    _fig.legend()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
