import marimo

__generated_with = "0.20.4"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import arviz as az
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import pymc as pm
    from scipy.stats import norm

    RANDOM_SEED = 1523
    rng = np.random.default_rng(RANDOM_SEED)

    az.style.use("arviz-darkgrid")
    return RANDOM_SEED, az, mo, norm, np, pl, plt, pm


@app.cell
def _(pl):
    df = pl.read_csv(
        "/home/marcelo/git/rethinking_stats_and_PyMC/data/Howell1.csv", separator=";"
    )


    def center_age(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(age_centered=pl.col("age") - pl.mean("age"))


    def create_age_indicator(df: pl.DataFrame, age: int) -> pl.DataFrame:
        return df.with_columns(is_adult=(pl.col("age") >= age).cast(pl.Int32))


    def filter_age(df: pl.DataFrame, age: int) -> pl.DataFrame:
        return df.filter(pl.col("age") >= age)


    data = df.pipe(center_age).pipe(create_age_indicator, age=18)

    df_adults = df.pipe(filter_age, age=18).pipe(center_age)

    data
    return (df_adults,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    \begin{align*}
    h_i &\sim \text{Normal}(\mu_i, \sigma) & [\text{likelihood}] \\
    \mu_i &= \alpha + \beta \bar{age_i} & [\text{linear model}] \\
    \alpha &\sim \text{Normal}(172, 20) & [\alpha \text{ prior}] \\
    \beta &\sim \text{Normal}(0, 0.01) & [\beta \text{ prior}] \\
    \sigma &\sim \text{Uniform}(0, 30) & [\sigma \text{ prior}]
    \end{align*}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Functions
    """)
    return


@app.cell
def _(az, np, pl, plt, pm):
    def generate_model(
        data: pl.DataFrame,
        **kwargs: dict[str, pm.Distribution],
    ) -> az.InferenceData:

        coords = {"obs": np.arange(data["height"].len())}

        with pm.Model(coords=coords) as model:
            # ===== DATA LAYER =====
            # Define all your data variables once with dims
            # This prevents typos later because you reference the pm.Data object
            age_c = pm.Data("age_c", data["age_centered"], dims="obs")

            # ===== PRIORS =====
            alpha = kwargs["alpha"]
            beta = kwargs["beta"]
            sigma = kwargs["sigma"]
            # alpha = pm.Normal("alpha", mu=172, sigma=20)
            # beta = pm.Normal("beta", mu=0, sigma=0.1)
            # sigma = pm.Uniform("sigma", lower=0, upper=30)

            # ===== LINEAR MODEL =====
            # Because age_c has dims="obs", mu automatically inherits it
            mu = pm.Deterministic("mu", alpha + beta * age_c, dims="obs")

            # ===== LIKELIHOOD =====
            # Link to actual observed heights (for posterior sampling)
            # For prior predictive, don't include "observed=parameter"
            height = pm.Normal("height", mu=mu, sigma=sigma, dims="obs")

            # ===== PRIOR PREDICTIVE =====
            idata_prior = pm.sample_prior_predictive(samples=500)

            return idata_prior


    def generate_split_model(
        data: pl.DataFrame,
        **kwargs,  # priors for under_18 (0) and above_18 (1) respectively
    ) -> az.InferenceData:

        # Create an indicator column: 0 for under 18, 1 for 18 and over
        data = data.with_columns(is_adult=(pl.col("age") >= 18).cast(pl.Int32))

        coords = {
            "obs": np.arange(data.height),
            "group": ["under_18", "above_18"],  # 0: under_18, 1: above_18
        }

        with pm.Model(coords=coords) as model:
            # ===== DATA LAYER =====
            age_c = pm.Data("age_c", data["age_centered"], dims="obs")
            # Define the group index for each observation
            group_idx = pm.Data("group_idx", data["is_adult"], dims="obs")

            # ===== PRIORS =====
            # Kwargs should now provide distributions with shape=2 (or dims="group")
            # Example: alpha = pm.Normal("alpha", mu=[120, 172], sigma=[20, 20], dims="group")
            alpha = kwargs["alpha"]
            beta = kwargs["beta"]
            sigma = kwargs["sigma"]

            # ===== LINEAR MODEL =====
            # Use group_idx to select the correct alpha and beta for each person
            mu = pm.Deterministic(
                "mu", alpha[group_idx] + beta[group_idx] * age_c, dims="obs"
            )

            # ===== LIKELIHOOD =====
            height = pm.Normal("height", mu=mu, sigma=sigma, dims="obs")

            # ===== PRIOR PREDICTIVE =====
            idata_prior = pm.sample_prior_predictive(samples=10)

            return idata_prior


    def plot_prior_predictive(idata: az.InferenceData, data: pl.DataFrame) -> None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        prior_mu = idata["prior"]["mu"].stack(sample=("chain", "draw")).values

        ax[0].plot(data["age"], prior_mu, c="steelblue", alpha=0.2, linewidth=2)
        ax[0].set_title("Prior Regression Lines")
        ax[0].set_xlabel("Age")
        ax[0].set_ylabel("Height (cm)")

        # Extract prior predictive samples for h
        prior_height = idata["prior"]["height"].stack(sample=("chain", "draw")).values
        az.plot_kde(prior_height, ax=ax[1])
        ax[1].set_title("Prior Predictive Distribution of h")
        ax[1].set_xlabel("Height (cm)")
        ax[1].set_yticks([])

        # plt.gca()
        plt.show()

    return generate_model, plot_prior_predictive


@app.cell
def _():
    print("""
    The reasoning for both models is that after 18 most adults change their height very little or nothing perceptible at all. However, we know that at later ages it is common for people to shrink a little. This is the rationale for allowing negative slopes to exist.

    Also, we know that the rate of growth/shrinkage has to be very small given that height barely change from adulthood until death. By choosing a rate parameter of 0.02 cm/year, I am allowing a person to grow/shrink:

    0.02*40 = 0.8 cm in 40 years.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## PyMC Models
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Only Adults Linear Model
    """)
    return


@app.cell
def _(df_adults, generate_model, plot_prior_predictive, pm):
    priors_1 = {
        "alpha": pm.Normal.dist(mu=172, sigma=20),  # cm
        "beta": pm.Normal.dist(mu=0, sigma=0.02),  # cm/year
        "sigma": pm.Uniform.dist(lower=0, upper=30),  # cm
    }

    normal_beta_model = generate_model(data=df_adults, **priors_1)

    plot_prior_predictive(idata=normal_beta_model, data=df_adults)
    return


@app.cell
def _(df_adults, generate_model, plot_prior_predictive, pm):
    priors_2 = {
        "alpha": pm.Normal.dist(mu=172, sigma=20),  # cm
        "beta": pm.Uniform.dist(lower=-0.02, upper=0.02),  # cm/year
        "sigma": pm.Uniform.dist(lower=0, upper=30),  # cm
    }

    uniform_beta_model = generate_model(data=df_adults, **priors_2)

    plot_prior_predictive(idata=uniform_beta_model, data=df_adults)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Linear Model to All Ages
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simpler Models
    """)
    return


@app.cell
def _(RANDOM_SEED, norm, plt, pm):
    # simple model that allows for the age to stop growing to vary
    # Constant accelerated growth until a specific age, then plateau until death
    # does not allow to shrink at older ages
    def generate_prior_lines(n_samples: int = 10) -> None:
        alpha = norm.rvs(50, 5, n_samples)
        beta = norm.rvs(5, 1, n_samples)
        age_terminal_growth = norm.rvs(20, 2, n_samples)

        plt.figure(figsize=(10, 5))
        for i in range(n_samples):
            plt.plot(
                [0, age_terminal_growth[i]],
                [alpha[i] + beta[i] * 0, alpha[i] + beta[i] * age_terminal_growth[i]],
                color="red",
                linewidth=2,
            )
            plt.plot(
                [age_terminal_growth[i], 100],
                [
                    alpha[i] + beta[i] * age_terminal_growth[i],
                    alpha[i] + beta[i] * age_terminal_growth[i],
                ],
                color="red",
                linewidth=2,
            )

        return plt.gca()


    def generate_prior_lines_pymc(n_samples: int = 10) -> None:
        with pm.Model() as _model:
            alpha = pm.Normal("alpha", mu=50, sigma=5)
            beta = pm.Normal("beta", mu=5, sigma=1)
            age_terminal_growth = pm.Normal("age_terminal_growth", mu=20, sigma=2)
            idata = pm.sample_prior_predictive(samples=n_samples, random_seed=RANDOM_SEED)

        prior = idata.prior.stack(sample=("chain", "draw"))
        alphas = prior["alpha"].values
        betas = prior["beta"].values
        ages_terminal_growth = prior["age_terminal_growth"].values

        plt.figure(figsize=(10, 5))
        for i in range(n_samples):
            # Growth phase: 0 to 20
            plt.plot(
                [0, ages_terminal_growth[i]],
                [alphas[i] + betas[i] * 0, alphas[i] + betas[i] * ages_terminal_growth[i]],
                color="red",
                linewidth=2,
            )
            # Plateau phase: ages_terminal_growth[i] to 100
            plt.plot(
                [ages_terminal_growth[i], 100],
                [
                    alphas[i] + betas[i] * ages_terminal_growth[i],
                    alphas[i] + betas[i] * ages_terminal_growth[i],
                ],
                color="red",
                linewidth=2,
            )

        plt.xlabel("Age")
        plt.ylabel("Height (cm)")
        plt.title(f"PyMC Prior Predictive Lines (n={n_samples})")
        return plt.gca()

    return generate_prior_lines, generate_prior_lines_pymc


@app.cell
def _(generate_prior_lines):
    generate_prior_lines()
    return


@app.cell
def _(generate_prior_lines_pymc):
    generate_prior_lines_pymc()
    return


if __name__ == "__main__":
    app.run()
