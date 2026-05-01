import marimo

__generated_with = "0.23.4"
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
    return (pm,)


@app.cell
def _(pm):
    def shift_model():
        with pm.Model() as model:
            # Priors
            mu = pm.Uniform("mu", 40, 75)
            sigma = pm.HalfNormal("sigma", 5)

            # Likelihood
            obs = pm.Normal("obs", mu=mu, sigma=sigma)

    return


if __name__ == "__main__":
    app.run()
