import marimo

__generated_with = "0.23.3"
app = marimo.App(width="columns")


@app.cell
def _():
    import altair as alt
    import arviz as az
    import preliz as pz
    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import polars as pl
    import pymc as pm
    import scipy.stats as stats
    from pathlib import Path
    from wigglystuff import EdgeDraw

    RANDOM_SEED = 1523
    rng = np.random.default_rng(RANDOM_SEED)
    return mo, plt, pz


@app.cell
def _(pz):
    # Maximum entropy Beta distribution with 90% of the mass between 0.1 and 0.7
    pz.maxent(pz.Normal(), 2, 3, 0.9)
    return


@app.cell
def _(pz):
    pz.Normal(4, 1).plot_pdf(), pz.Normal(4, 1).plot_cdf()
    return


@app.cell
def _(plt, pz):
    # Define the distribution
    dist = pz.Normal(4, 1)

    # 1. Let's find the value for the 95th percentile
    p_value = 0.95
    quantile_value = dist.ppf(p_value)

    print(f"95% of the data in a Normal(4, 1) lies below: {quantile_value:.2f}")

    # 2. Visual Plotting
    dist.plot_ppf()
    plt.axvline(p_value, color="r", linestyle="--", label=f"p={p_value}")
    plt.axhline(
        quantile_value, color="g", linestyle="--", label=f"Value={quantile_value:.2f}"
    )
    plt.legend()
    plt.show()
    return


@app.cell
def _(mo):
    mu_slider = mo.ui.slider(-10, 10, step=0.1, show_value=True, label="mu")
    alpha_slider = mo.ui.slider(-10, 10, step=0.1, show_value=True, label="alpha")
    sigma_slider = mo.ui.slider(0.1, 10, step=0.1, show_value=True, label="sigma")
    return alpha_slider, mu_slider, sigma_slider


@app.cell
def _(alpha_slider, mo, mu_slider, pz, sigma_slider):
    mean, var, skewness, kurtosis = pz.SkewNormal(
        mu=mu_slider.value, sigma=sigma_slider.value, alpha=alpha_slider.value
    ).moments()

    mo.vstack(
        [
            mo.hstack([mu_slider, sigma_slider, alpha_slider]),
            mo.md(
                f"**Mean:** {mean:.2f} | **Variance:** {var:.2f} | **Skewness:** {skewness:.2f} | **Kurtosis:** {kurtosis:.2f}"
            ),
            pz.SkewNormal(
                mu=mu_slider.value, sigma=sigma_slider.value, alpha=alpha_slider.value
            ).plot_pdf(),
        ],
        align="stretch",
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
