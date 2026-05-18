import marimo

__generated_with = "0.23.6"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    from itertools import combinations
    import numpy as np
    import xarray as xr
    import polars as pl
    import matplotlib.pyplot as plt
    import pymc as pm
    import arviz as az
    import preliz as pz
    from pathlib import Path

    RANDOM_SEED = 1523
    rng = np.random.default_rng(RANDOM_SEED)

    # az.style.use("arviz-variat")
    plt.style.use("fivethirtyeight")
    # Set default figure size to 14 inches wide by 7 inches tall
    plt.rcParams["figure.figsize"] = (14, 7)
    # Make the layout "tight" by deffault so labels don't overlap
    plt.rcParams["figure.autolayout"] = True
    return Path, az, mo, np, pl, plt, pm, rng, xr


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
def _(bikes, np, pm, rng):
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
    return bikes_idata, bikes_lin_model


@app.cell
def _(az, bikes_idata):
    az.plot_dist(bikes_idata, var_names=["~μ"], figure_kwargs={"figsize": (14, 5)})
    return


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

    return (fn_bikes_neg_binom__model,)


@app.cell
def _(fn_bikes_neg_binom__model):
    bikes_neg_binom_model, bikes_neg_binom_idata = fn_bikes_neg_binom__model()
    return (bikes_neg_binom_idata,)


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
    axes[0].fill_between(
        babies["month"], μ_m + 1 * σ_m, μ_m - 1 * σ_m, alpha=0.6, color="C1"
    )
    axes[0].fill_between(
        babies["month"], μ_m + 2 * σ_m, μ_m - 2 * σ_m, alpha=0.4, color="C1"
    )

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


if __name__ == "__main__":
    app.run()
