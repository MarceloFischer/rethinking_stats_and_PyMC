import marimo

__generated_with = "0.23.5"
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

    plt.style.use("fivethirtyeight")
    # Set default figure size to 14 inches wide by 7 inches tall
    plt.rcParams["figure.figsize"] = (14, 7)
    # Make the layout "tight" by deffault so labels don't overlap
    plt.rcParams["figure.autolayout"] = True
    # sets default credible interval used by arviz
    az.rcParams["stats.ci_prob"] = 0.89
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
def _(bikes, pm, rng):
    def fn_bikes_linear_model():
        with pm.Model() as bikes_linear:
            # Priors
            α = pm.Normal("α", mu=0, sigma=100)
            β = pm.Normal("β", mu=0, sigma=10)
            σ = pm.HalfCauchy("σ", beta=10)
            # Mean
            μ = pm.Deterministic("μ", α + β * bikes["temperature"].to_numpy())
            # Likelihood
            y_pred = pm.Normal("y_pred", mu=μ, sigma=σ, observed=bikes["rented"])

            idata = pm.sample(random_seed=rng)
        return bikes_linear, idata


    bikes_lin_model, bikes_idata = fn_bikes_linear_model()
    return bikes_idata, bikes_lin_model


@app.cell
def _(az, bikes_idata):
    az.plot_posterior(bikes_idata, var_names=["~μ"], figsize=(16, 7))
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

        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 7))
        # Main points
        axs[0].scatter(
            x=dataf["temperature"], y=dataf["rented"], alpha=0.3, label="Observed"
        )
        axs[1].scatter(
            x=dataf["temperature"], y=dataf["rented"], alpha=0.3, label="Observed"
        )

        # Plot mean lines - handle label to avoid multiple legend entries for 'lines'
        axs[0].plot(
            x_plot,
            mean_lines.T,
            c="b",
            alpha=0.5,
            label=[None] * (n - 1) + ["Posterior Samples"],
        )

        # plot mean line
        axs[0].plot(x_plot, mean_line, c="r", lw=5, label="Posterior Mean")
        axs[1].plot(x_plot, mean_line, c="r", lw=5, label="Posterior Mean")

        ### Plot HDI
        ## Both below work fine. The hdi_lines will be the same in every run because we are using the whole data to plot it
        # az.plot_hdi(
        #     x_plot,
        #     mean_lines,
        #     hdi_prob=0.89,
        #     ax=axs[1],
        #     color="b",
        #     fill_kwargs={"label": "89% HDI"},
        # )

        # hdi lines
        hdi_lines = az.hdi(idata.posterior["μ"])["μ"]
        az.plot_hdi(
            x=dataf["temperature"],
            # y=idata.posterior["μ"], # can plot straight from the posterior as y_data
            hdi_data=hdi_lines,
            ax=axs[1],
            color="b",
            fill_kwargs={"label": "89% HDI"},
        )

        axs[0].set_title("Mean Posterior Uncertainty (Samples)")
        axs[0].set_xlabel("Temperature")
        axs[0].set_ylabel("Rented Bikes")
        axs[0].legend()

        axs[1].set_title("Mean Posterior Uncertainty (HDI)")
        axs[1].set_xlabel("Temperature")
        axs[1].legend()

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
def _(az, bikes, bikes_idata, bikes_temp_sort_idx, plt):
    plt.scatter(x=bikes["temperature"], y=bikes["rented"], alpha=0.3, label="Observed")

    # plot mean line
    plt.plot(
        bikes["temperature"][bikes_temp_sort_idx],
        bikes_idata["posterior"]["μ"].mean(("chain", "draw"))[bikes_temp_sort_idx],
        c="r",
        lw=5,
        label="Posterior Mean",
    )

    # plot multiple HDI levels sequentially to avoid TypeError with list in hdi_prob
    for _prob in [0.5, 0.89]:
        az.plot_hdi(
            bikes["temperature"],
            bikes_idata["posterior_predictive"]["y_pred"],
            hdi_prob=_prob,
            color="green",
            fill_kwargs={
                "alpha": 0.3 if _prob == 0.5 else 0.2,
                "label": f"{int(_prob * 100)}% HDI",
            },
        )

    plt.title("Posterior Predictive Intervals")
    plt.xlabel("Temperature")
    plt.ylabel("Rented Bikes")
    plt.legend()
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    # Counting Bikes
    """)
    return


@app.cell
def _(bikes, pm, rng):
    def fn_bikes_neg_binom__model():
        with pm.Model() as neg_binom_model:
            # Priors
            α = pm.Normal("α", mu=0, sigma=1)
            β = pm.Normal("β", mu=0, sigma=5)
            σ = pm.HalfNormal("σ", sigma=10)
            # Mean
            μ = pm.Deterministic("μ", pm.math.exp(α + β * bikes["temperature"].to_numpy()))
            # Likelihood
            y_pred = pm.NegativeBinomial("y_pred", mu=μ, alpha=σ, observed=bikes["rented"])

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
def _(az, bikes, bikes_neg_binom_idata, bikes_temp_sort_idx, plt):
    # _x_plot = np.linspace()
    plt.scatter(bikes["temperature"], bikes["rented"])

    plt.plot(
        bikes["temperature"][bikes_temp_sort_idx],
        bikes_neg_binom_idata["posterior"]["μ"].mean(("chain", "draw"))[
            bikes_temp_sort_idx
        ],
    )

    # for _hdi_prob in [0.5, 0.89]:
    az.plot_hdi(
        bikes["temperature"],
        bikes_neg_binom_idata["posterior_predictive"]["y_pred"],
        color="green",
        hdi_prob=0.94,
    )
    plt.gca()
    return


@app.cell
def _(az, bikes_neg_binom_idata):
    # az.summary(bikes_neg_binom_idata, var_names=['~μ'], kind='stats').round(3)
    az.plot_trace(bikes_neg_binom_idata, var_names=["~μ"])
    return


@app.cell
def _(az, bikes_idata, bikes_neg_binom_idata):
    (
        az.plot_ppc(bikes_idata, num_pp_samples=200),
        az.plot_ppc(bikes_neg_binom_idata, num_pp_samples=200),
    )
    return


@app.cell
def _():
    return


@app.cell(column=3)
def _():
    return


if __name__ == "__main__":
    app.run()
