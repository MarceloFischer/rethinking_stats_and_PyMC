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
    return Path, bmb, mo, np, pl, plt, pm, rng


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
    idata_t = model_t.fit(random_seed=rng)
    model_t
    return idata_t, model_t


@app.cell
def _(model_t):
    model_t.graph()
    return


@app.cell
def _(model_t):
    model_t.plot_priors(figsize=(12, 5), col_wrap=2)
    return


@app.cell
def _(bikes, bmb, idata_t, model_t, plt):
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
def _(bikes, bmb, rng):
    model_th = bmb.Model("rented ~ temperature + humidity", bikes.to_pandas(), family="negativebinomial")
    idata_th = model_th.fit(random_seed=rng)
    return idata_th, model_th


@app.cell
def _(model_th):
    model_th.graph()
    return


@app.cell
def _(bikes, bmb, idata_th, model_th, np):
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


if __name__ == "__main__":
    app.run()
