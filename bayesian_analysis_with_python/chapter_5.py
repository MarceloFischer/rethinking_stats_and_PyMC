import marimo

__generated_with = "0.23.15"
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
    import polars as pl

    #######################################################

    RANDOM_SEED = 1523
    rng = np.random.default_rng(RANDOM_SEED)

    alt.theme.enable('fivethirtyeight')
    plt.style.use("fivethirtyeight")

    # Set default figure size to 14 inches wide by 5 inches tall
    plt.rcParams["figure.figsize"] = (12, 6)
    # You can also set the DPI (dots per inch) for crisper images
    plt.rcParams["figure.dpi"] = 100
    # Make the layout "tight" by default so labels don't overlap
    plt.rcParams["figure.autolayout"] = True
    # sets default credible interval used by arviz
    az.rcParams["stats.ci_prob"] = 0.89
    return az, mo, np, plt, pm, rng


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Helper Functions
    """)
    return


@app.cell
def _(az, plt):
    def plot_ppc_tstats(models, t_stat="mean"):
        colors = [
            '#008fd5',  # Azul (Principal)
            '#fc4f30',  # Vermelho
            '#e5ae38',  # Amarelo/Dourado
            '#6d904f',  # Verde
            '#8b8b8b',  # Cinza escuro
            '#810f7c'   # Roxo
        ]
        pc = None

        for i, (label, idata) in enumerate(models.items()):
            pc = az.plot_ppc_tstat(
                idata,
                t_stat=t_stat,
                backend="matplotlib",
                plot_collection=pc,
                visuals={
                    "dist": {
                        "label": label,
                        'color': colors[i]
                    }
                },
            )

        plt.legend()

        return plt.gca()

    return (plot_ppc_tstats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 5.1 - Posterior Predictive Checks
    """)
    return


@app.cell
def _(np, plt):
    dummy_data = np.loadtxt("data/dummy.csv")
    x = dummy_data[:, 0]
    y = dummy_data[:, 1]

    order = 2
    x_p = np.vstack([x**i for i in range(1, order + 1)])
    x_c = (x_p - x_p.mean(axis=1, keepdims=True)) / x_p.std(axis=1, keepdims=True)
    y_c = (y - y.mean()) / y.std()

    plt.scatter(x_c[0], y_c)
    plt.xlabel("x")
    plt.ylabel("y")
    return x_c, y_c


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two models:

    \[
    y=\alpha+\beta x
    \]

    and

    \[
    y=\alpha+\beta_0 x+\beta_1 x^2
    \]
    """)
    return


@app.cell
def _(pm, rng, x_c, y_c):
    def fn_5_1_linear():
        with pm.Model() as model_linear:
            α = pm.Normal('α', mu=0, sigma=1)
            β = pm.Normal('β', mu=0, sigma=10)
            σ = pm.HalfNormal('σ', sigma=10)

            # only using the linear (first column of x_c) component
            μ = α + β * x_c[0]

            y_pred = pm.Normal('y_pred', mu=μ, sigma=σ, observed=y_c)
            idata = pm.sample(2_000, idata_kwargs={'log_likelihood': True}, random_seed=rng)
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)
        return model_linear, idata


    def fn_5_1_quadratic():
        with pm.Model() as model_quadratic:
            α = pm.Normal('α', mu=0, sigma=1)
            β = pm.Normal('β', mu=0, sigma=10, shape=x_c.shape[0])
            σ = pm.HalfNormal('σ', sigma=10)

            μ = α + pm.math.dot(β, x_c)

            y_pred = pm.Normal('y_pred', mu=μ, sigma=σ, observed=y_c)
            idata = pm.sample(2_000, idata_kwargs={'log_likelihood': True}, random_seed=rng)
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)
        return model_quadratic, idata

    l_5_1_model, l_5_1_idata = fn_5_1_linear()
    q_5_1_model, q_5_1_idata = fn_5_1_quadratic()
    return l_5_1_idata, l_5_1_model, q_5_1_idata


@app.cell
def _(az, l_5_1_idata, np, plt, q_5_1_idata, x_c, y_c):
    posterior_l = az.extract(l_5_1_idata)
    posterior_q = az.extract(q_5_1_idata)
    _sort_idx = x_c[0].argsort()

    plt.scatter(x_c[0], y_c)

    plt.plot(
        x_c[0][_sort_idx],
        posterior_l['α'].mean().item() + posterior_l['β'].mean().item() * x_c[0][_sort_idx],
        label='linear'
    )

    plt.plot(
        x_c[0][_sort_idx],
        posterior_q['α'].mean().item() + np.dot(posterior_q['β'].mean('sample'), x_c)[_sort_idx],
        label = 'quadratic'
    )

    plt.legend()
    return


@app.cell
def _(az, l_5_1_idata):
    l_5_1_idata.sel(chain=0, draw=[0, 1]), az.extract(l_5_1_idata)#.sel(sample=0)
    return


@app.cell
def _(az, l_5_1_idata, q_5_1_idata):
    _pc1 = az.plot_ppc_dist(l_5_1_idata, kind='kde', visuals={"observed_dist": {"color": "black", "label": "Observed"}})
    _pc1.add_title("Linear model")
    _pc1.viz["figure"].item().legend()  # grabs the labelled lines automatically

    _pc2 = az.plot_ppc_dist(q_5_1_idata, kind='kde', visuals={"observed_dist": {"color": "black", "label": "Observed"}})
    _pc2.add_title("Quadratic model")
    _pc2.viz["figure"].item().legend()  # grabs the labelled lines automatically

    _pc1, _pc2
    return


@app.cell
def _(l_5_1_idata, plot_ppc_tstats, q_5_1_idata):
    models = {
        "Linear": l_5_1_idata,
        "Quadratic": q_5_1_idata,
    }

    plot_ppc_tstats(models, t_stat='iqr')
    return (models,)


@app.cell(column=1)
def _(az, models):
    az.compare(models)
    return


@app.cell
def _(az, models):
    az.plot_compare(az.compare(models), figure_kwargs={'figsize':(12, 5)})
    return


@app.cell
def _(az, l_5_1_idata, l_5_1_model, pm, rng):
    with l_5_1_model:
        l_5_1_idata.update(pm.sample_prior_predictive(random_seed=rng))

    az.plot_bf(l_5_1_idata, var_names=['α'], ref_val=0, figure_kwargs={"figsize":(12, 5)})
    return


if __name__ == "__main__":
    app.run()
