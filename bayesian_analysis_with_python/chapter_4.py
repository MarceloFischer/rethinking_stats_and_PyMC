# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.1.0",
#     "arviz==1.1.0",
#     "marimo>=0.23.6",
#     "matplotlib==3.10.9",
#     "numpy==2.4.5",
#     "openai==2.38.0",
#     "polars==1.40.1",
#     "preliz==0.25.0",
#     "pymc==6.0.0",
#     "pytest==9.0.3",
#     "ruff==0.15.15",
#     "vegafusion==2.0.3",
#     "vl-convert-python==1.9.0.post1",
#     "xarray==2026.4.0",
# ]
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    from itertools import combinations
    import numpy as np
    import xarray as xr
    import polars as pl
    import matplotlib.pyplot as plt
    import altair as alt
    import pymc as pm
    import arviz as az
    import preliz as pz
    from pathlib import Path

    RANDOM_SEED = 1523
    rng = np.random.default_rng(RANDOM_SEED)

    alt.theme.enable('fivethirtyeight')
    # az.style.use("arviz-variat")
    plt.style.use("fivethirtyeight")
    # Set default figure size to 14 inches wide by 7 inches tall
    plt.rcParams["figure.figsize"] = (14, 7)
    # Make the layout "tight" by deffault so labels don't overlap
    plt.rcParams["figure.autolayout"] = True
    return Path, alt, az, mo, np, pl, plt, pm, rng, xr


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    # Bikes Example
    """)
    return


@app.cell
def _(Path, pl):
    BIKE_PATH = Path(__file__).parent.parent / "data" / "bikes.csv"

    bikes = pl.read_csv(BIKE_PATH)

    bikes_temp_sort_idx = bikes["temperature"].arg_sort()

    bikes.plot.point(x="temperature", y="rented").properties(width="container")
    return bikes, bikes_temp_sort_idx


@app.cell
def _():
    # pz.HalfCauchy(beta=10).plot_pdf(support=(0, 200))
    return


@app.cell
def _(az, bikes, np, pm, rng):
    def fn_bikes_linear_model():
        coords = {"obs_id": np.arange(len(bikes))}
        with pm.Model(coords=coords) as bikes_linear:
            temps = pm.Data("temperature", bikes["temperature"].to_numpy(), dims="obs_id")
            # Priors
            α = pm.Normal("α", mu=0, sigma=100)
            β = pm.Normal("β", mu=0, sigma=10)
            σ = pm.HalfCauchy("σ", beta=10)
            # Mean
            μ = pm.Deterministic("μ", α + β * temps, dims="obs_id")
            # Likelihood
            rented = pm.Normal(
                "rented", mu=μ, sigma=σ, observed=bikes["rented"], dims="obs_id"
            )

            idata = pm.sample(random_seed=rng)
        return bikes_linear, idata


    bikes_lin_model, bikes_idata = fn_bikes_linear_model()

    az.plot_dist(bikes_idata, var_names=["~μ"], figure_kwargs={"figsize": (14, 5)})
    return bikes_idata, bikes_lin_model


@app.cell
def _():
    # # The value of the arrays below come from the post
    # post = az.extract(bikes_idata, num_samples=3, rng=rng)
    # x_plot = xr.DataArray(
    #     np.linspace(bikes["temperature"].min(), bikes["temperature"].max(), 3),
    #     dims="temps",
    # )
    # post["α"], post["β"], x_plot, (post["α"] + post["β"] * x_plot).T

    # # example of extracting 3 samples from the posterior
    # # column vector of α's. Dimension name 'sample' (see above)
    # _α = np.array([70.4022797 , 55.57923616, 64.91986758]).reshape(-1, 1)
    # # column vector of β's. Dimension name 'sample' (see above)
    # _β = np.array([8.00574532, 8.02860063, 8.31879763]).reshape(-1, 1)
    # # column vector of temperatures's. Dimension name 'temps' (see above)
    # _temps = np.array([-5.18, 15.03, 35.24])
    # # Because they have different dimension names, broadcasting happens. So the column vectors (3x1) => (3x3) get stretched horizontally and the row vectors (1x3) => (3x3) get stretched vertically.
    # # This generates a 3x3 result.
    # _lines = (
    #     np.array([70.4022797 , 55.57923616, 64.91986758]).reshape(-1, 1)
    #     + np.array([8.00574532, 8.02860063, 8.31879763]).reshape(-1, 1)
    #     * np.array([-5.18, 15.03, 35.24])
    # ).T

    # mo.hstack([_α, _β, _temps, _lines])
    return


@app.cell
def _(az, bikes, bikes_idata, np, pl, plt, xr):
    def plot_bikes_posterior_lines(
        dataf: pl.DataFrame = bikes, idata: az.InferenceData = bikes_idata, n: int = 50
    ):
        # stacks chains/draws
        posterior = az.extract(idata, num_samples=n, group="posterior")
        # Create the x-values to use in the plot
        x_plot = xr.DataArray(
            np.linspace(dataf["temperature"].min(), dataf["temperature"].max(), n),
            dims="temps",
        )
        # MAP line
        mean_line = posterior["α"].mean() + posterior["β"].mean() * x_plot
        # Mean lines - not needed as μ wa saved in the inference data
        mean_lines = posterior["α"] + posterior["β"] * x_plot

        # Main points
        plt.scatter(x=dataf["temperature"], y=dataf["rented"], alpha=0.3, label="Observed")

        # Plot mean lines - handle label to avoid multiple legend entries for 'lines'
        plt.plot(
            x_plot,
            mean_lines.T,
            c="b",
            alpha=0.5,
            label=[None] * (n - 1) + ["Posterior Samples"],
        )

        # plot mean line
        plt.plot(x_plot, mean_line, c="r", lw=5, label="Posterior Mean")

        plt.title("Mean Posterior Uncertainty (Samples)")
        plt.xlabel("Temperature")
        plt.ylabel("Rented Bikes")
        plt.legend()

        return plt.gca()

    return (plot_bikes_posterior_lines,)


@app.cell
def _(plot_bikes_posterior_lines):
    plot_bikes_posterior_lines()
    return


@app.cell
def _(bikes_idata, bikes_lin_model, pm, rng):
    pm.sample_posterior_predictive(
        trace=bikes_idata, model=bikes_lin_model, extend_inferencedata=True, random_seed=rng
    )
    return


@app.cell
def _(az, bikes_idata):
    az.plot_lm(
        dt=bikes_idata,
        x="temperature",
        y="rented",
        ci_kind="hdi",
        ci_prob=(0.5, 0.89),
    )
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    # Counting Bikes
    """)
    return


@app.cell
def _(bikes, np, pm, rng):
    def fn_bikes_neg_binom__model():
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
            rented = pm.NegativeBinomial(
                "rented", mu=μ, alpha=σ, observed=bikes["rented"], dims="obs_id"
            )

            idata = pm.sample(random_seed=rng)
            pm.sample_posterior_predictive(
                idata, extend_inferencedata=True, random_seed=rng
            )

        return neg_binom_model, idata

    bikes_neg_binom_model, bikes_neg_binom_idata = fn_bikes_neg_binom__model()
    return (bikes_neg_binom_idata,)


@app.cell
def _():
    return


@app.cell
def _(bikes, bikes_neg_binom_idata, bikes_temp_sort_idx, plt):
    # _x_plot = np.linspace()
    plt.scatter(bikes["temperature"], bikes["rented"])

    plt.plot(
        bikes["temperature"][bikes_temp_sort_idx],
        bikes_neg_binom_idata["posterior"]["μ"].mean(("chain", "draw"))[
            bikes_temp_sort_idx
        ],
    )
    return


@app.cell
def _(az, bikes_neg_binom_idata):
    az.plot_lm(bikes_neg_binom_idata, ci_kind="eti", ci_prob=[0.5, 0.89])
    return


@app.cell
def _(az, bikes_neg_binom_idata):
    # az.summary(bikes_neg_binom_idata, var_names=['~μ'], kind='stats', ci_kind='eti').round(3)
    # az.plot_dist(bikes_neg_binom_idata, var_names=["~μ"], col_wrap=2, figure_kwargs={'figsize':(18, 6)})
    az.plot_trace_dist(
        bikes_neg_binom_idata, var_names=["~μ"], figure_kwargs={"figsize": (18, 8)}
    )
    return


@app.cell
def _(az, bikes_idata, bikes_neg_binom_idata):
    az.plot_ppc_dist(bikes_idata), az.plot_ppc_dist(bikes_neg_binom_idata)
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""
    # Logistic Regression
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Not done here. Will do during the exercises
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Variable Variance
    """)
    return


@app.cell
def _(Path, pl):
    BABIES_PATH = Path(__file__).parent.parent / "data" / "babies.csv"

    babies = pl.read_csv(BABIES_PATH)
    babies.head()
    return (babies,)


@app.cell
def _(babies, np, pl, pm, rng):
    def fn_babies_var_std():
        coords = {"obs_idx": np.arange(len(babies))}
        with pm.Model(coords=coords) as var_std_model:
            # Data
            month = pm.Data(
                "month", babies["month"].cast(pl.Float32).to_numpy(), dims="obs_idx"
            )
            # Priors
            α = pm.Normal("α", mu=0, sigma=10)
            β = pm.Normal("β", mu=0, sigma=10)
            γ = pm.HalfNormal("γ", sigma=10)
            δ = pm.HalfNormal("δ", sigma=10)
            # Deterministic relationships
            μ = pm.Deterministic("μ", α + β * month**0.5, dims="obs_idx")
            σ = pm.Deterministic("σ", γ + δ * month, dims="obs_idx")
            # likelihood
            length = pm.Normal(
                "length",
                mu=μ,
                sigma=σ,
                observed=babies["length"].to_numpy(),
                dims="obs_idx",
            )
            idata = pm.sample(random_seed=rng)
            pm.sample_posterior_predictive(
                idata, extend_inferencedata=True, random_seed=rng
            )
        return var_std_model, idata


    # Same as the above, but using coordinates for the parameters as well
    # def fn_babies_var_std():
    #     coords = {'obs_idx': np.arange(len(babies)), 'parameters':['intercept', 'slope']}
    #     with pm.Model(coords=coords) as var_std_model:
    #         # Data
    #         month = pm.Data('month', babies['month'].cast(pl.Float32).to_numpy(), dims='obs_idx')
    #         # Priors
    #         mean_params = pm.Normal('mean_params', mu=0, sigma=10, dims='parameters')
    #         sigma_params = pm.HalfNormal('sigma_params', sigma=10, dims='parameters')
    #         # Deterministic relationships
    #         μ = pm.Deterministic('μ', mean_params[0] + mean_params[1] * month**0.5, dims='obs_idx')
    #         σ = pm.Deterministic('σ', sigma_params[0] + sigma_params[1] * month, dims='obs_idx')
    #         # likelihood
    #         length = pm.Normal('length', mu=μ, sigma=σ, observed=babies['length'].to_numpy(), dims='obs_idx')
    #         idata = pm.sample(random_seed=rng)
    # pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)
    #     return var_std_model, idata

    babies_model, babies_idata = fn_babies_var_std()
    return babies_idata, babies_model


@app.cell
def _(az, babies_idata):
    az.plot_dist(babies_idata, var_names=['~μ', '~σ'], col_wrap=2, figure_kwargs={'figsize':(14, 5)})
    return


@app.cell
def _(az, babies_idata):
    az.plot_lm(
        babies_idata,
        ci_prob=[0.6, 0.89],
        visuals={
            "pe_line": {"color": "red"},
            "ci_band": {"alpha": 0.4, "color": "green"},
            "observed_scatter": {"marker": "C2"},
        },
    )
    return


@app.cell
def _(az, babies, babies_idata, plt):
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(babies["month"], babies["length"], "C0.", alpha=0.5)

    posterior = az.extract(babies_idata)

    μ_m = posterior["μ"].mean("sample").values
    σ_m = posterior["σ"].mean("sample").values

    axes[0].plot(babies["month"], μ_m, c="k")
    axes[0].fill_between(babies["month"], μ_m + 1 * σ_m, μ_m - 1 * σ_m, alpha=0.6, color="C1")
    axes[0].fill_between(babies["month"], μ_m + 2 * σ_m, μ_m - 2 * σ_m, alpha=0.4, color="C1")

    axes[0].set_xlabel("months")
    axes[0].set_ylabel("length")

    axes[1].plot(babies["month"], σ_m)
    axes[1].set_xlabel("months")
    axes[1].set_ylabel(r"$\bar \sigma$", rotation=0)
    return


@app.cell
def _(az, babies_idata, babies_model, plt, pm, rng):
    # To get the posterior predictive distribution for a specific value of month, we can give new data to the model
    with babies_model:
        pm.set_data({"month": [0.5]}, coords={"obs_idx": [0]})
        # Setting predictions=True will add a new "predictions" group to our idata. This lets us store the posterior,
        # posterior_predictive, and predictions all in the same object.
        pm.sample_posterior_predictive(
            babies_idata, extend_inferencedata=True, predictions=True, random_seed=rng
        )

    _ref_vals = babies_idata.predictions.dataset.quantile(
        [0.35, 0.5, 0.89], dim=["chain", "draw"])["length"].values.flatten()

    _pc = az.plot_dist(
        babies_idata, group="predictions", labeller=az.labels.DimCoordLabeller()
    )

    az.add_lines(_pc, _ref_vals)

    plt.gca()
    return


@app.cell
def _():
    return


@app.cell(column=4, hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercises
    """)
    return


@app.cell
def _(Path, np, pl):
    HOWELL_PATH = Path(__file__).parent.parent / "data" / "howell.csv"

    howell = pl.read_csv(HOWELL_PATH, separator=';')
    howell_mean_height = howell['height'].mean()
    howell = howell.with_columns(
        height_c = pl.col('height') - pl.col('height').mean(),
        height_std = (pl.col('height') - pl.col('height').mean()) / (pl.col('height').std()),
        over_17 = pl.col('age') >= 18
    )

    h_adults = howell.filter(pl.col('age')>=18)
    h_adults_mean_height = h_adults['height'].mean()

    howell_coords = {'obs_idx': np.arange(len(howell))}
    adults_howell_coords = {'obs_idx': np.arange(len(h_adults))}

    ###########
    ##  Iris ##
    ###########

    IRIS_PATH = Path(__file__).parent.parent / "data" / "iris.csv"
    iris = pl.read_csv(IRIS_PATH)

    iris_coords = {'obs_idx': np.arange(len(iris))}
    return (
        adults_howell_coords,
        h_adults,
        h_adults_mean_height,
        howell,
        howell_coords,
        howell_mean_height,
        iris,
        iris_coords,
    )


@app.cell
def _(alt, howell, np):
    howell.plot.point(
        x=alt.X('height', title='Exp of Height', scale=alt.Scale(type='pow', exponent=np.e)),
        y=alt.Y('weight'),
        color='over_17',
        tooltip=['height', 'weight', 'over_17']
    ).properties(
        width='container',
        title='Exp of Height vs Weight'
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises 1 and 2
    """)
    return


@app.cell
def _(adults_howell_coords, h_adults, h_adults_mean_height, pm, rng):
    def fn_howell_adults_linear_model():
        with pm.Model(coords=adults_howell_coords) as adults_linear:
            # heights
            height = pm.Data('height', h_adults['height'].to_numpy(), dims='obs_idx')
            # Priors
            α = pm.Normal('α', mu=60, sigma=15)
            β = pm.HalfNormal('β', sigma=5)
            σ = pm.HalfNormal('σ', sigma=15)
            # mean
            μ = pm.Deterministic('μ', α + β * (height - h_adults_mean_height))
            # likelihood
            weight = pm.Normal('weight', mu=μ, sigma=σ, observed=h_adults['weight'], dims='obs_idx')
            # sampling
            idata = pm.sample(random_seed=rng)
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)
        return adults_linear, idata

    howell_adults_model, howell_adults_idata = fn_howell_adults_linear_model()
    return howell_adults_idata, howell_adults_model


@app.cell
def _(az, howell_adults_idata):
    # az.plot_dist(howell_adults_idata, var_names=['~μ'], col_wrap=1)
    az.plot_trace_dist(howell_adults_idata, var_names=['~μ'], figure_kwargs={"figsize":(18, 8)})
    return


@app.cell
def _(az, howell_adults_idata):
    az.plot_lm(howell_adults_idata)
    return


@app.cell
def _(az, howell_adults_idata):
    az.plot_ppc_dist(howell_adults_idata)
    return


@app.cell
def _(az, howell_adults_idata, howell_adults_model, np, pm, rng):
    with howell_adults_model:
        _new_heights = np.array([142.5, 155.3, 132, 150])
        pm.set_data({'height': _new_heights}, coords={'obs_idx': range(len(_new_heights))})
        # _ppc = az.extract(
        #     pm.sample_posterior_predictive(
        #         howell_adults_idata,
        #         random_seed=rng,
        #         extend_inferencedata=True,
        #         predictions=True
        #     ),
        #     group='predictions'
        # )
        pm.sample_posterior_predictive(
                howell_adults_idata,
                random_seed=rng,
                extend_inferencedata=True,
                predictions=True
            )

    az.plot_dist(
        howell_adults_idata,
        group='predictions',
        ci_prob=0.89,
        labeller=az.labels.DimCoordLabeller(),
        col_wrap=2,
        figure_kwargs={'figsize':(14, 5)}
    )

    # The below is to match the answer from the book. The new Arviz version is fucking my life as the old one was much easiert to work with because of the plt arguments (ax, ...)

    # _pred_weight = howell_adults_idata.predictions["weight"]

    # _fig_overlay, _ax_overlay = plt.subplots(figsize=(10, 6))

    # for _obs in _pred_weight.coords["obs_idx"].values:
    #     _samples = _pred_weight.sel(obs_idx=_obs).stack(_sample=("chain", "draw")).to_numpy()
    #     _density, _edges = np.histogram(_samples, bins=70, density=True)
    #     _centers = 0.5 * (_edges[:-1] + _edges[1:])
    #     _ax_overlay.plot(_centers, _density, lw=2, label=str(_obs))

    # _ax_overlay.set_title("Posterior predictive distributions (all heights)")
    # _ax_overlay.set_xlabel("Predicted weight (kg)")
    # _ax_overlay.set_ylabel("Density")
    # _ax_overlay.legend(title="obs_idx")
    # plt.gca()
    return


@app.cell
def _(h_adults_mean_height, howell_adults_idata, np):
    howell_adults_idata.posterior.α.mean().values + howell_adults_idata.posterior.β.mean().values * (np.array([142.5, 155.3, 132, 150]) - h_adults_mean_height)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 3
    """)
    return


@app.cell
def _(howell, howell_coords, howell_mean_height, pm, rng):
    def fn_howell_linear_model():
        with pm.Model(coords=howell_coords) as howell_linear:
            # heights
            height = pm.Data('height', howell['height'].to_numpy(), dims='obs_idx')
            # Priors
            α = pm.Normal('α', mu=35, sigma=20)
            β = pm.HalfNormal('β', sigma=5)
            σ = pm.HalfNormal('σ', sigma=15)
            # mean
            μ = pm.Deterministic('μ', α + β * (height - howell_mean_height))
            # likelihood
            weight = pm.Normal('weight', mu=μ, sigma=σ, observed=howell['weight'], dims='obs_idx')
            # sampling
            idata = pm.sample(random_seed=rng)
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)
        return howell_linear, idata

    howell_model, howell_idata = fn_howell_linear_model()
    return (howell_idata,)


@app.cell
def _(az, howell_idata):
    # az.plot_dist(howell_idata, var_names=['~μ'], col_wrap=1)
    az.plot_trace_dist(howell_idata, var_names=['~μ'], figure_kwargs={"figsize":(18, 8)})
    return


@app.cell
def _(az, howell_idata):
    az.plot_lm(howell_idata)
    return


@app.cell
def _(az, howell_idata):
    az.plot_ppc_dist(howell_idata)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 4
    """)
    return


@app.cell
def _(howell, howell_coords, pm, rng):
    def fn_howell_linear_exp_model():
        with pm.Model(coords=howell_coords) as howell_linear_exp:
            # heights
            height = pm.Data('height', howell['height'].to_numpy(), dims='obs_idx')
            height_std = pm.Data('height_std', howell['height_std'].to_numpy(), dims='obs_idx')
            # Priors
            α = pm.Normal('α', mu=3.5, sigma=1)
            β = pm.Normal('β', mu=0, sigma=0.5)
            σ = pm.HalfNormal('σ', sigma=15)
            # mean
            # If log(weight) ~ height, then weight ~ exp(α + β * height)
            # Also, if log(weight) ~ Normal(height) => weight ~ lognormal(height)
            μ = pm.Deterministic('μ', pm.math.exp(α + β * height_std), dims='obs_idx')
            # likelihood
            weight = pm.Normal('weight', mu=μ, sigma=σ, observed=howell['weight'], dims='obs_idx')
            # sampling
            idata = pm.sample(random_seed=rng, target_accept=0.95)
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)
        return howell_linear_exp, idata


    def fn_howell_lognormal_model():
        with pm.Model(coords=howell_coords) as model:
            # Data
            height = pm.Data('height', howell['height'].to_numpy(), dims='obs_idx') # for plotting
            height_std = pm.Data("height_std", howell['height_std'].to_numpy(), dims='obs_idx')
            # Priors
            α = pm.Normal("α", mu=3.5, sigma=1)
            β = pm.Normal("β", mu=0, sigma=0.5)
            σ = pm.HalfNormal("σ", sigma=0.5)
            # linear model
            μ = pm.Deterministic('μ', α + β * height_std, dims='obs_idx')
            # We expect the log(w) ~ Normal(a + b*h, sigma)
            # Same as saying that the w ~ LogNormal(a + b*h, sigma)
            weight = pm.LogNormal(
                "weight",
                mu=μ,
                sigma=σ,
                observed=howell["weight"],
                dims='obs_idx'
            )

            idata = pm.sample(target_accept=0.9, random_seed=rng)
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)

        return model, idata

    howell_exp_model, howell_exp_idata = fn_howell_linear_exp_model()
    howell_logN_model, howell_logN_idata = fn_howell_lognormal_model()
    return howell_exp_idata, howell_logN_idata


@app.cell
def _(az, howell_exp_idata):
    az.plot_trace_dist(
        howell_exp_idata,
        var_names=['~μ'],
        figure_kwargs={'figsize':(14, 7)}
    )
    return


@app.cell
def _(az, howell_logN_idata):
    az.plot_trace_dist(
        howell_logN_idata,
        var_names=['~μ'],
        figure_kwargs={'figsize':(14, 7)}
    )
    return


@app.cell
def _(az, howell_exp_idata, howell_logN_idata):
    az.plot_lm(howell_exp_idata), az.plot_lm(howell_logN_idata)

    # The lognormal model looks more realistic as uncertainty increases with height.
    return


@app.cell
def _(az, howell_exp_idata, howell_logN_idata):
    az.plot_ppc_dist(howell_exp_idata), az.plot_ppc_dist(howell_logN_idata)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 6
    """)
    return


@app.cell
def _(iris, iris_coords, pl, pm, rng):
    def iris_logit_model(predictor_cols: list[str], unique_species: list[str]=['setosa', 'versicolor']):
        filtered_iris = iris.filter(pl.col('species').is_in(unique_species))
        y = filtered_iris['species'].cast(pl.Enum(unique_species)).to_physical().to_numpy()

        models = {}
        for col in predictor_cols:
            x = filtered_iris[col].to_numpy()
            x_c = x - x.mean()

            with pm.Model(coords=iris_coords) as model:
                # data
                pred_var = pm.Data(f'{predictor_cols}', x, dims='obx_idx')
                pred_var_c = pm.Data(f'{predictor_cols}_c', x_c, dims='obx_idx')

                # Priors
                a = pm.Normal('a', mu=0, sigma=1)
                b = pm.Normal('b', mu=0, sigma=3)

                # Logistic Model
                μ = a + pm.math.dot(pred_var_c, b)
                θ = pm.Deterministic('θ', pm.math.sigmoid(μ))
                bd = pm.Deterministic('bd', -a/b)

                # Likelihood
                y_pred = pm.Bernoulli('y_pred', p=θ, observed=y, dims='obx_idx')

                #sampling
                idata = pm.sample(random_seed=rng)
                pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)

            models[f'model_{col}'] = idata
        return models

    iris_models = iris_logit_model(predictor_cols=['sepal_length', 'petal_length', 'petal_width'])
    return (iris_models,)


@app.cell
def _(az, iris_models):
    (
        az.plot_lm(iris_models['model_sepal_length']),
        az.plot_lm(iris_models['model_petal_length']),
        az.plot_lm(iris_models['model_petal_width']),
    )
    return


@app.cell
def _(az, iris_models):
    for model_name, idata in iris_models.items():
        print(model_name)
        print(az.summary(idata, var_names=['~θ']))
        print('-'*50)
    return


@app.cell(column=5, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 7
    """)
    return


@app.cell
def _(Path, alt, pl):
    POKEMON_PATH = Path(__file__).parent.parent / "data" / "Pokemon.csv"

    pok = pl.read_csv(POKEMON_PATH)

    pok.columns = [
        col.lower().replace(". ", "_")
        for col in pok.columns
    ]

    pok_target_col = 'sp_atk'

    pok = pok.with_columns(
        ((pl.col(pok_target_col) - pl.col(pok_target_col).mean()) / (pl.col(pok_target_col).std())).alias(f'{pok_target_col}_std')
    )

    pok.plot.point(
        x=alt.X(pok_target_col),
        y=alt.Y('attack')
    ).properties(
        width='container'
    )
    return pok, pok_target_col


@app.cell
def _(np, pm, pok, pok_target_col, rng):
    def fn_pokemon_linear():
        coords = {'obs_idx': np.arange(len(pok))}
        with pm.Model(coords=coords) as model:
            # data
            predictor = pm.Data(f'{pok_target_col}', pok[pok_target_col].to_numpy(), dims='obs_idx')
            predictor_std = pm.Data(f'{pok_target_col}_std', pok[f'{pok_target_col}_std'].to_numpy(), dims='obs_idx')
            # Priors
            α = pm.Normal('α', mu=60, sigma=10)
            β = pm.Normal('β', mu=0, sigma=2)
            σ = pm.HalfNormal('σ', sigma=5)
            # Linear Model
            μ = pm.Deterministic('μ', α + pm.math.dot(predictor_std, β))
            # likelihood
            atk = pm.Normal('atk', mu=μ, sigma=σ, observed=pok['attack'], dims='obs_idx')
            idata = pm.sample(random_seed=rng)
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)
        return model, idata

    pok_model, pok_idata = fn_pokemon_linear()
    return (pok_idata,)


@app.cell
def _(az, pok_idata):
    az.summary(pok_idata, kind='stats', var_names=['~μ']), az.plot_trace_dist(pok_idata, var_names=['~μ'])
    return


@app.cell
def _(az, pok_idata):
    az.plot_lm(pok_idata)
    return


if __name__ == "__main__":
    app.run()
