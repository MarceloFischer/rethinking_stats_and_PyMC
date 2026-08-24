import marimo

__generated_with = "0.24.0"
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

    import kulprit as kpt

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
    return Path, alt, az, bmb, kpt, mo, np, pl, plt, pm, rng


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
    bikes = bikes.with_columns(hour = pl.col("hour").cast(pl.Float64))

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
            "wrap": 3,
            "title": lambda h: f"humidity = {h}",
        },
    )

    _p.share(
        x=False
    ).layout(
        size=(14, 7)
    ).show()
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


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""
    # Penguins - Bambi Style
    """)
    return


@app.cell
def _(Path, alt, mo, pl):
    PENGUINS_PATH = Path(__file__).parent.parent / "data" / "penguins.csv"

    penguins = pl.read_csv(PENGUINS_PATH).drop_nulls()

    penguins_spe = penguins['species'].unique().to_numpy()
    penguins_spe_idx = penguins['species'].cast(pl.Enum(penguins_spe)).to_physical().to_numpy()

    _c = (
        alt.Chart(penguins)
        .mark_point()
        .encode(
            x=alt.X("bill_length:Q", scale=alt.Scale(domain=(2.9, 6.1)), axis=alt.Axis(labelFontSize=14)),
            y=alt.Y("body_mass:Q", scale=alt.Scale(domain=(2, 6.4)), axis=alt.Axis(labelFontSize=14)),
            color="species:N",
        )
        .properties(
            width='container',
            title="Bill Length vs Body Mass by Species"
        )
    )

    mo.ui.altair_chart(_c)
    return (penguins,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Models with Categories
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Models with Categories - Hierarchical
    """)
    return


@app.cell
def _(bmb, penguins):
    #  0 + bill_length + species would create one slope and one intercept for each category of species.
    # Each species coefficient would now be the mean body mass of that species (holding bill_length constant)
    model_p_hie = bmb.Model("body_mass ~ (bill_length|species)", data=penguins.to_pandas(), dropna=True)
    # idata_p_hie = model_p_hie.fit(random_seed=rng, target_accept=0.95)

    model_p_hie
    return (model_p_hie,)


@app.cell
def _(az, idata_p_hie, mo):
    mo.stop("idata_p_hie" not in dir(), mo.md("Fit model_p_hie to continue"))

    az.plot_trace_dist(idata_p_hie)
    return


@app.cell
def _(az, idata_p_hie, mo):
    mo.stop("idata_p_hie" not in dir(), mo.md("Fit model_p_hie to continue"))

    az.plot_forest(idata_p_hie, combined=True, figure_kwargs={'figsize':(12,5)})
    return


@app.cell
def _(bmb, idata_p_hie, mo, model_p_hie, plt):
    mo.stop("idata_p_hie" not in dir(), mo.md("Fit model_p_hie to continue"))

    _fig, _ax = plt.subplots(figsize=(10, 5))

    _p = bmb.interpret.plot_predictions(
        model_p_hie,
        idata_p_hie,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactions
    """)
    return


@app.cell
def _(bmb, penguins, rng):
    # model_int = bmb.Model("body_mass ~ bill_length + bill_depth + bill_length:bill_depth", data=penguins.to_pandas(), dropna=True)
    model_int = bmb.Model("body_mass ~ scale(bill_length) * scale(bill_depth)", data=penguins.to_pandas(), dropna=True)
    idata_int = model_int.fit(random_seed=rng)
    return idata_int, model_int


@app.cell
def _(az, idata_int, mo):
    mo.stop("idata_int" not in dir(), mo.md("Fit model_int to continue"))
    az.summary(idata_int).round(2)
    return


@app.cell
def _(bmb, idata_int, mo, model_int, np, penguins, plt):
    mo.stop("idata_int" not in dir(), mo.md("Fit model_int to continue"))
    _fig, _ax = plt.subplots(figsize=(10, 5))

    _conditional = {
        "bill_depth": np.linspace(penguins["bill_depth"].min(), penguins["bill_depth"].max(), 50),
        "bill_length": [3.21, 3.95, 4.45, 4.86, 5.96],
    }

    _p = bmb.interpret.plot_predictions(
        model_int,
        idata_int,
        conditional=_conditional,
        # target='rented',
        subplot_kwargs={
            "main": "bill_depth",
            "group": "bill_length",
            # "panel": "bill_length"
        },
        fig_kwargs={
            "xlabel": "Bill Depth (mm)",
            "ylabel": "Body Mass (g)"
        },
    )

    _p.on(_ax).plot()
    _fig.legend()
    return


@app.cell
def _(bmb, idata_int, model_int):
    bmb.interpret.plot_comparisons(
        model=model_int,
        idata=idata_int,
        contrast={'bill_depth':[1.4, 1.8]},
        conditional={'bill_length':[3.5, 4.5, 5.5]},
    ).show()
    return


@app.cell
def _(bmb, idata_int, model_int):
    bmb.interpret.plot_slopes(
        model=model_int,
        idata=idata_int,
        wrt={'bill_depth':1.8},
        conditional={'bill_length':[3.5, 4.5, 5.5]},
    ).show()
    return


@app.cell(column=4)
def _():
    # this model is not equivalent to bmb.Model("body_mass ~ (bill_length|species)", data=penguins.to_pandas(), dropna=True)
    # The bambi model has correlation between intercepts and slopes for each species. The PyMC model below has an independent slope
    # and intercept for each species. They have different modelling assumptions.
    # Look at "PyMC hierarchical model vs Bambi formula equivalence" Claude chat for further clarifications.

    # def penguin_hier_pymc():
    #     coords = {
    #         "obs_id": np.arange(len(penguins)),
    #         "penguins_spe": penguins_spe,
    #     }

    #     centered_bill_length = penguins['bill_length'].to_numpy() - penguins['bill_length'].mean()

    #     with pm.Model(coords=coords) as model:
    #         # Data
    #         bill_len = pm.Data('bill_length', centered_bill_length, dims='obs_id')
    #         # Hyperpriors
    #         μ_line = pm.Normal('μ_line', mu=4.5, sigma=1)
    #         σ_line = pm.HalfNormal('σ_line', sigma=2)
    #         σ = pm.Exponential('σ', 1, dims='penguins_spe')
    #         # Priors
    #         α = pm.Normal('α', mu=μ_line, sigma=σ_line, dims='penguins_spe')
    #         β = pm.Normal('β', mu=μ_line, sigma=σ_line, dims='penguins_spe')
    #         # Mean
    #         μ = pm.Deterministic('μ', α[penguins_spe_idx] + β[penguins_spe_idx] * bill_len, dims='obs_id')
    #         # Likelihood
    #         body_mass = pm.Normal(
    #             'body_mass',
    #             mu=μ,
    #             sigma=σ[penguins_spe_idx],
    #             observed=penguins['body_mass'].to_numpy(),
    #             dims='obs_id'
    #         )
    #         idata = pm.sample(1_000, random_seed=rng)
    #         pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)

    #     return model, idata
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Kulprit
    """)
    return


@app.cell
def _(Path, pl):
    BODY_FAT_PATH = Path(__file__).parent.parent / "data" / "body_fat.csv"

    body_fat = pl.read_csv(BODY_FAT_PATH)
    return (body_fat,)


@app.cell
def _(bmb, body_fat, rng):
    body_fat_model = bmb.Model("siri ~ age + weight + height + abdomen + thigh + wrist", data=body_fat.to_pandas())
    body_fat_idata = body_fat_model.fit(random_seed=rng, idata_kwargs={'log_likelihood': True})
    return body_fat_idata, body_fat_model


@app.cell
def _(body_fat_idata, body_fat_model, kpt, mo):
    mo.stop("body_fat_idata" not in dir(), mo.md("Fit body_fat_model to continue"))
    body_fat_ppi = kpt.ProjectionPredictive(body_fat_model, body_fat_idata)
    body_fat_ppi.project()
    return (body_fat_ppi,)


@app.cell
def _(body_fat_ppi, kpt):
    kpt.plot_compare(
        body_fat_ppi.compare(),
        figure_kwargs={'figsize':(12, 5)}
    )
    return


@app.cell
def _():
    return


@app.cell(column=5, hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercises
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 2

    Apply what you learned in the previous point and specify a HalfNormal prior for the slope of model_t.
    """)
    return


@app.cell
def _(bikes, bmb):
    def ex2():
        priors = {
            'temperature': bmb.Prior('HalfNormal', sigma=1)
        }
    
        model_t = bmb.Model("rented ~ temperature", bikes.to_pandas(), priors=priors, family="negativebinomial")
        # idata_t = model_t.fit(random_seed=rng)
        return model_t

    ex2()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 3

    Define a model like model_poly4, but using raw polynomials, compare the coefficients and the mean fit of both models.
    """)
    return


@app.cell
def _(bikes, bmb, rng):
    def ex3():
        model_poly4 = bmb.Model(
            "rented ~ poly(scale(hour), degree=4)",
            bikes.to_pandas(), family="negativebinomial"
        )
        model_poly4_raw = bmb.Model(
            "rented ~ scale(hour) + {scale(hour) ** 2} + {scale(hour) ** 3} + {scale(hour) ** 4}",
            bikes.to_pandas(), family="negativebinomial"
        )

        idata_poly4 = model_poly4.fit(random_seed=rng, idata_kwargs={'log_likelihood': True})
        idata_poly4_raw = model_poly4_raw.fit(random_seed=rng, idata_kwargs={'log_likelihood': True})

        return model_poly4, idata_poly4, model_poly4_raw, idata_poly4_raw

    # model_poly4, idata_poly4, model_poly4_raw, idata_poly4_raw = ex3()
    return


@app.cell
def _(az, idata_poly4, idata_poly4_raw, mo):
    mo.stop("idata_poly4" not in dir(), mo.md("Fit model_poly4 to continue"))

    az.compare({
        'idata_poly4': idata_poly4,
        "idata_poly4_raw": idata_poly4_raw
    })
    return


@app.cell
def _(az, idata_poly4, idata_poly4_raw, mo):
    mo.stop("idata_poly4" not in dir(), mo.md("Fit model_poly4 to continue"))
    az.plot_dist(idata_poly4, figure_kwargs={'figsize':(12, 5)}), az.plot_dist(idata_poly4_raw, figure_kwargs={'figsize':(12, 5)})
    return


@app.cell
def _(
    bikes,
    bmb,
    idata_poly4,
    idata_poly4_raw,
    mo,
    model_poly4,
    model_poly4_raw,
    plt,
):
    mo.stop("idata_poly4" not in dir(), mo.md("Fit model_poly4 to continue"))
    _fig, _axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5))

    _axes[0].scatter(bikes['hour'], bikes['rented'], s=10)
    _axes[1].scatter(bikes['hour'], bikes['rented'], s=10)

    _p1 = bmb.interpret.plot_predictions(
            model_poly4_raw,
            idata_poly4_raw,
            conditional="hour",
            # target='rented',
            fig_kwargs={
                "xlabel": "Hour",
                "ylabel": "Rented",
                "title": "Predictions Raw"
            }
        )

    _p2 = bmb.interpret.plot_predictions(
            model_poly4,
            idata_poly4,
            conditional="hour",
            # target='rented',
            fig_kwargs={
                "xlabel": "Hour",
                "ylabel": "Rented",
                "title": "Predictions Poly()"
            }
        )

    _p1.on(_axes[0]).layout(engine="tight").plot()
    _p2.on(_axes[1]).layout(engine="tight").plot().show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 4

    Explain in your own words what a distributional model is.

    Ans:
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 5
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():


    return


if __name__ == "__main__":
    app.run()
