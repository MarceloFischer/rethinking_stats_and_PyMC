import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo
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
    # Make the layout "tight" by deffault so labels don't overlap
    plt.rcParams["figure.autolayout"] = True
    # sets default credible interval used by arviz
    az.rcParams["stats.ci_prob"] = 0.89
    return Path, az, pl, pm, rng


@app.cell
def _(Path, pl):
    CS_PATH = Path(__file__).parent.parent / "data" / "chemical_shifts_theo_exp.csv"

    cs_data = pl.read_csv(CS_PATH)
    cs_data = cs_data.with_columns(diff=pl.col("theo") - pl.col("exp"))
    return (cs_data,)


@app.cell
def _(cs_data, pl):
    def create_cs_cat_vars(dataf: pl.DataFrame = cs_data):
        """
        Extract unique amino acid categories and their corresponding physical indices from a Polars DataFrame.

        Args:
            dataf (pl.DataFrame): A DataFrame containing an 'aa' column with amino acid labels.

        Returns:
            tuple: A tuple containing:
                - aa_cats (np.ndarray): Array of unique amino acid category strings.
                - aa_idx (np.ndarray): Array of physical integer indices corresponding to the 'aa' column.
        """
        aa_cats = dataf["aa"].unique().to_numpy()
        aa_idx = dataf["aa"].cast(pl.Enum(aa_cats)).to_physical().to_numpy()

        return aa_cats, aa_idx


    aa_cats, aa_idx = create_cs_cat_vars()
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

    return fn_cs_h_model, fn_cs_nh_model


@app.cell
def _(fn_cs_h_model, fn_cs_nh_model):
    cs_nh_model, cs_nh_idata = fn_cs_nh_model()
    cs_h_model, cs_h_idata = fn_cs_h_model()
    return cs_h_idata, cs_nh_idata


@app.cell
def _(az, cs_h_idata, cs_nh_idata):
    _axes = az.plot_forest(
        [cs_nh_idata, cs_h_idata],
        model_names=["non-hierarchical", "hierarchical"],
        var_names="mu",
        combined=True,
        figsize=(14, 7),
    )

    y_lims = _axes[0].get_ylim()
    _axes[0].vlines(cs_h_idata.posterior["mu_global"].mean(), *y_lims, color="k", ls=":")
    return


if __name__ == "__main__":
    app.run()
