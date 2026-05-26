# /// script
# dependencies = [
#     "altair==6.1.0",
#     "marimo",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "polars==1.41.0",
#     "preliz>0.2",
#     "pymc==5.28.5",
# ]
# requires-python = ">=3.11"
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="columns")


@app.cell
def _():
    import polars as pl 
    import marimo as mo
    import numpy as np
    import altair as alt
    import pymc as pm
    import preliz as pz
    import matplotlib.pyplot as plt
    import arviz as az

    rng = np.random.default_rng(1523)
    alt.theme.enable('fivethirtyeight')
    return az, mo, pl, pm, pz, rng


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 4
    """)
    return


@app.cell
def _(pl):
    howell= pl.read_csv(r"C:\Users\marce\Documents\github\rethinking_stats_and_PyMC\data\howell.csv", separator=';')

    howell = howell.with_columns(
        height_std = (pl.col('height') - pl.col('height').mean()) / pl.col('height').std(),
        over_17 = pl.col('age')> 17
    )

    howell
    return (howell,)


@app.cell
def _(howell, pm, rng):
    def fn_howell_lognormal_model():
        with pm.Model() as model:
            height_std = pm.Data("height_std", howell['height_std'].to_numpy())

            α = pm.Normal("α", mu=3.5, sigma=1)
            β = pm.Normal("β", mu=0, sigma=0.5)

            σ = pm.HalfNormal("σ", sigma=0.5)

            μ = pm.Deterministic('mu', α + β * height_std)

            # We expect the log(w) ~ Normal(a + b*h, sigma)
            # Same as saying that the w ~ LogNormal(a + b*h, sigma)
            weight = pm.LogNormal(
                "weight",
                mu=μ,
                sigma=σ,
                observed=howell["weight"]
            )

            idata = pm.sample(target_accept=0.9, random_seed=rng)
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)

        return model, idata

    return


@app.cell
def _():
    # model, idata = fn_howell_lognormal_model()

    # idx_sort = howell['height'].arg_sort()

    # az.plot_trace(idata, var_names=['~mu'])
    # # az.plot_ppc(idata)

    # plt.scatter(howell['height'], howell['weight'])
    # plt.plot(
    #     howell['height'][idx_sort],
    #     np.exp(idata.posterior['mu'].mean(('chain', 'draw')))[idx_sort]
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 5
    """)
    return


@app.cell
def _(pz):
    pz.Exponential(1/30).plot_pdf(moments='md')
    pz.Gamma(mu=20, sigma=15).plot_pdf(moments='md')
    pz.Gamma(2, 0.1).plot_pdf(moments='md')
    return


@app.cell
def _(pl, pm, rng):
    ans = pl.read_csv(r'C:\Users\marce\Documents\github\rethinking_stats_and_PyMC\data\anscombe.csv')

    y_4 = ans.filter(pl.col('group') == 'IV')['y'].to_numpy()
    x_4 = ans.filter(pl.col('group') == 'IV')['x'].to_numpy()

    def model_t2_gamma():
        with pm.Model() as model_t2:
            α = pm.Normal('α', mu=0, sigma=100)
            β = pm.Normal('β', mu=0, sigma=1)
            ϵ = pm.HalfCauchy('ϵ', 5)
            ν = pm.Gamma('ν', mu=20, sigma=15)

            y_pred = pm.StudentT('y_pred', mu=α + β * x_4, sigma=ϵ, nu=ν, observed=y_4)
            idata = pm.sample_prior_predictive(random_seed=rng)
            idata.extend(pm.sample(2000, target_accept=0.85, random_seed=rng))
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)
        return idata

    def model_t2_exp():
        with pm.Model() as model_t2:
            α = pm.Normal('α', mu=0, sigma=100)
            β = pm.Normal('β', mu=0, sigma=1)
            ϵ = pm.HalfCauchy('ϵ', 5)
            ν = pm.Exponential('ν', 1/30)

            y_pred = pm.StudentT('y_pred', mu=α + β * x_4, sigma=ϵ, nu=ν, observed=y_4)
            idata = pm.sample_prior_predictive(random_seed=rng)
            idata.extend(pm.sample(2000, target_accept=0.85, random_seed=rng))
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)
        return idata

    # idata_t2_gamma = model_t2_gamma()
    # idata_t2_exp = model_t2_exp()
    return


@app.cell
def _(az, idata_t2_exp, idata_t2_gamma):
    az.plot_forest([idata_t2_exp, idata_t2_gamma], model_names=['exp', 'gamma'], combined=True, var_names=['ν'])
    return


@app.cell
def _(az, idata_t2_exp, idata_t2_gamma):
    az.plot_dist_comparison(idata_t2_gamma, var_names=['ν'], figsize=(10,4)), az.plot_dist_comparison(idata_t2_exp, var_names=['ν'], figsize=(10,4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 6
    """)
    return


@app.cell
def _(pl, pm, rng):
    iris = pl.read_csv(r'C:\Users\marce\Documents\github\rethinking_stats_and_PyMC\data\iris.csv')

    def iris_logit_model(predictor_cols: list[str], unique_species: list[str]=['setosa', 'versicolor']):
        filtered_iris = iris.filter(pl.col('species').is_in(unique_species))
        y = filtered_iris['species'].cast(pl.Enum(unique_species)).to_physical().to_numpy()

        models = {}
        for col in predictor_cols:
            x = filtered_iris[col].to_numpy()
            x_c = x - x.mean()
        
            with pm.Model() as model:
                # data
                pred_var = pm.Data(f'{predictor_cols}_c', x_c)
            
                # Priors
                a = pm.Normal('a', mu=0, sigma=2)
                b = pm.Normal('b', mu=0, sigma=3)
            
                # Logistic Model
                mu = a + pm.math.dot(pred_var, b)
                theta = pm.Deterministic('theta', pm.math.sigmoid(mu))
                bd = pm.Deterministic('bd', -a/b)
            
                # Likelihood
                y_pred = pm.Bernoulli('y_pred', p=theta, observed=y)
            
                #sampling
                idata = pm.sample(random_seed=rng)
                pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)
        
            models[f'model_{col}'] = idata
        return models

    iris_models = iris_logit_model(predictor_cols=['sepal_length', 'petal_length', 'petal_width'])
    return (iris_models,)


@app.cell
def _(az, iris_models):
    for model_name, idata in iris_models.items():
        print(model_name)
        print(az.summary(idata, var_names=['~theta']))
        print('-'*50)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
