import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    from itertools import combinations
    import numpy as np
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
    return Path, mo, pl, pm, rng


@app.cell
def _(Path, pl):
    CS_PATH = Path(__file__).parent.parent / "data" / "chemical_shifts_theo_exp.csv"

    cs_data = pl.read_csv(CS_PATH)
    cs_data = cs_data.with_columns(diff=pl.col("theo") - pl.col("exp"))
    return (cs_data,)


@app.cell
def _(pl):
    def code_cat_vars(dataf: pl.DataFrame, col_to_code: str):
        """
        Extract unique amino acid categories and their corresponding physical indices from a Polars DataFrame.

        Args:
            dataf (pl.DataFrame): A DataFrame containing an 'aa' column with amino acid labels.

        Returns:
            tuple: A tuple containing:
                - aa_cats (np.ndarray): Array of unique amino acid category strings.
                - aa_idx (np.ndarray): Array of physical integer indices corresponding to the 'aa' column.
        """
        cats = dataf[col_to_code].unique().to_numpy()
        cats_idx = dataf[col_to_code].cast(pl.Enum(cats)).to_physical().to_numpy()

        return cats, cats_idx

    return (code_cat_vars,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chemical Shift Hierarchical Model
    """)
    return


@app.cell
def _(cs_data):
    cs_data
    return


@app.cell
def _(code_cat_vars, cs_data):
    aa_cats, aa_idx = code_cat_vars(cs_data, "aa")
    return aa_cats, aa_idx


@app.cell
def _(aa_cats, aa_idx, cs_data, pm, rng):
    def fn_cs_nh_model():
        coords = {"aa_cats": aa_cats, "aa_idx": aa_idx, "aa_flat": aa_cats[aa_idx]}
        with pm.Model(coords=coords) as cs_nh_model:
            # priors
            mu = pm.Normal("mu", mu=0, sigma=10, dims="aa_cats")
            sigma = pm.HalfNormal("sigma", sigma=10, dims="aa_cats")
            # likelihood
            obs = pm.Normal(
                "obs",
                mu=mu[aa_idx],
                sigma=sigma[aa_idx],
                observed=cs_data["diff"],
                dims="aa_flat",
            )

            idata = pm.sample(random_seed=rng)
        return cs_nh_model, idata


    def fn_cs_h_model():
        coords = {"aa_cats": aa_cats, "aa_idx": aa_idx, "aa_flat": aa_cats[aa_idx]}
        with pm.Model(coords=coords) as cs_h_model:
            # hyperpriors
            mu_global = pm.Normal("mu_global", mu=0, sigma=10)
            sigma_global = pm.HalfNormal("sigma_global", sigma=10)
            # priors
            mu = pm.Normal("mu", mu=mu_global, sigma=sigma_global, dims="aa_cats")
            sigma = pm.HalfNormal("sigma", sigma=10, dims="aa_cats")
            # likelihood
            obs = pm.Normal(
                "obs",
                mu=mu[aa_idx],
                sigma=sigma[aa_idx],
                observed=cs_data["diff"],
                dims="aa_flat",
            )

            idata = pm.sample(random_seed=rng)
        return cs_h_model, idata

    return


@app.cell
def _():
    # cs_nh_model, cs_nh_idata = fn_cs_nh_model()
    # cs_h_model, cs_h_idata = fn_cs_h_model()

    # _axes = az.plot_forest(
    #     [cs_nh_idata, cs_h_idata],
    #     model_names=["non-hierarchical", "hierarchical"],
    #     var_names="mu",
    #     combined=True,
    #     figsize=(14, 7),
    # )

    # y_lims = _axes[0].get_ylim()
    # _axes[0].vlines(cs_h_idata.posterior["mu_global"].mean(), *y_lims, color="k", ls=":")
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _():
    # pz.maxent(pz.Gamma(), 50, 200, 0.9)
    # pz.Gamma(6.94, 0.0549).plot_pdf(moments='md', pointinterval=True)
    # pz.Gamma(mu=125, sigma=50).plot_pdf(moments='md', pointinterval=True, figsize=(14, 7), levels=[0.89, 0.95])
    # pz.Gamma(mu=5, sigma=4).plot_pdf(moments='md')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Footbal Example
    """)
    return


@app.cell
def _(Path, code_cat_vars, pl):
    FB_PATH = Path(__file__).parent.parent / "data" / "football_players.csv"

    fb_data = pl.read_csv(FB_PATH)

    pos_cats, pos_idx = code_cat_vars(fb_data, "position")
    fb_data
    return fb_data, pos_cats, pos_idx


@app.cell
def _(fb_data, pm, pos_cats, pos_idx, rng):
    def fn_fb_h_model():
        coords = {"pos_cats": pos_cats, "pos_idx": pos_idx}
        with pm.Model(coords=coords) as fb_model:
            # Hyperpriors (glboal parameters - top league professional players avg and precision)
            mu = pm.Beta("mu", alpha=1.7, beta=5.8)
            nu = pm.Gamma("nu", mu=125, sigma=50)
            # Parameters for positions
            mu_p = pm.Beta("mu_p", mu=mu, nu=nu, dims="pos_cats")
            nu_p = pm.HalfNormal("nu_p", sigma=50, dims="pos_cats")
            # Parameter for each player
            p = pm.Beta("p", mu=mu_p[pos_idx], nu=nu_p[pos_idx])
            # likelihood
            # goals per shot = success_rate
            gps = pm.Binomial(
                "gps", n=fb_data["shots"].to_numpy(), p=p, observed=fb_data["goals"]
            )

            idata = pm.sample(3_000, random_seed=rng)

        return fb_model, idata

    return


@app.cell
def _():
    # fb_h_model, fb_h_idata = fn_fb_h_model()

    # _, _axes = plt.subplots(3, 1, figsize=(14, 7.5), sharex=True)
    # az.plot_posterior(fb_h_idata, var_names=["mu"], ax=_axes[0])
    # az.plot_posterior(
    #     fb_h_idata.posterior.sel(pos_cats="FW"), var_names=["mu_p"], ax=_axes[1]
    # )
    # az.plot_posterior(fb_h_idata.posterior.sel(p_dim_0=1457), var_names=["p"], ax=_axes[2])
    # _axes[0].set_title("Global Mean")
    # _axes[1].set_title("FW Pos Mean")
    # _axes[2].set_title("Messi Mean")
    return


@app.cell
def _():
    # az.plot_forest(fb_h_idata, var_names=["mu_p"], combined=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 3
    """)
    return


@app.cell
def _(Path, code_cat_vars, pl, pm, rng):
    TIPS_PATH = Path(__file__).parent.parent / "data" / "tips.csv"

    tips = pl.read_csv(TIPS_PATH)

    days, days_idx = code_cat_vars(tips, "day")


    def fn_tips_h_model():
        coords = {"days": days, "days_idx": days_idx, "days_flat": days[days_idx]}
        with pm.Model(coords=coords) as tips_h_model:
            # Hyperpriors
            mu = pm.HalfNormal("mu", sigma=8)
            # Priors
            mu_t = pm.HalfNormal("mu_t", sigma=mu, dims="days")
            sigma_t = pm.HalfNormal("sigma_t", sigma=10, dims="days")
            # likelihhod
            tip = pm.Gamma(
                "tip",
                mu=mu_t[days_idx],
                sigma=sigma_t[days_idx],
                observed=tips["tip"],
                dims="days_flat",
            )

            idata = pm.sample(2_000, random_seed=rng)
        return tips_h_model, idata


    # tips_h_model, tips_h_idata = fn_tips_h_model()
    # with tips_h_model:
    #     tips_h_idata.extend(pm.sample_posterior_predictive(tips_h_idata))

    # (
    #     az.plot_forest(tips_h_idata, var_names=["mu", "mu_t"], combined=True),
    #     az.summary(tips_h_idata, kind="stats").round(2),
    # )
    return


@app.cell
def _():
    # _, axes = plt.subplots(2, 2)
    # az.plot_ppc(
    #     tips_h_idata,
    #     num_pp_samples=100,
    #     coords={"days_flat": days},
    #     flatten=[],
    #     ax=axes,
    #     random_seed=RANDOM_SEED,
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 5
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
