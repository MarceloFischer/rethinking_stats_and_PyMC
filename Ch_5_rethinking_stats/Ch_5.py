# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.0.0",
#     "arviz==0.22.0",
#     "matplotlib==3.10.8",
#     "numpy==2.3.5",
#     "openai==2.11.0",
#     "polars==1.36.1",
#     "pymc==5.26.1",
#     "python-lsp-ruff==2.3.0",
#     "python-lsp-server==1.14.0",
#     "ruff==0.14.9",
#     "scipy==1.16.3",
#     "vegafusion==2.0.3",
#     "vl-convert-python==1.8.0",
#     "websockets==15.0.1",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import operator
    import altair as alt
    import arviz as az
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import pymc as pm
    import scipy.stats as stats
    from pathlib import Path

    RANDOM_SEED = 1523
    rng = np.random.default_rng(RANDOM_SEED)
    plt.style.use("fivethirtyeight")

    az.rcParams["stats.ci_prob"] = 0.89  # sets default credible interval used by arviz
    return Path, az, mo, np, operator, pl, plt, pm, rng


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Constants
    """)
    return


@app.cell
def _(Path, pl):
    #############
    # Constants #
    #############
    WAFFLE_PATH = Path(__file__).parent.parent / "data" / "WaffleDivorce.csv"
    HOWELL_PATH = Path(__file__).parent.parent / "data" / "Howell1.csv"

    # women=0 and men=1 in the dataset
    SEX = ["women", "men"]

    raw_waffle_data = pl.read_csv(WAFFLE_PATH, separator=";")
    raw_howell_data = pl.read_csv(HOWELL_PATH, separator=";")
    return SEX, raw_howell_data, raw_waffle_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Helper Functions
    """)
    return


@app.cell
def _(operator, pl):
    def cols_to_lowercase(dataf: pl.DataFrame) -> pl.DataFrame:
        return dataf.rename({col: col[0].lower() + col[1:] for col in dataf.columns})


    def std_cols_of_interest(dataf: pl.DataFrame) -> pl.DataFrame:
        # add standardised columns
        return dataf.with_columns(
            [
                ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(f"{col}_std")
                for col in ["divorce", "medianAgeMarriage", "marriage"]
            ]
        )


    def filter_by_comparison(
        df: pl.DataFrame, col_name: str, value: float, op_str: str
    ) -> pl.DataFrame:
        # Map string labels to python operators
        ops = {
            ">": operator.gt,
            ">=": operator.ge,
            "<": operator.lt,
            "<=": operator.le,
            "==": operator.eq,
        }

        # Use the operator to create a Polars expression
        # Equivalent to: df.filter(pl.col(col_name) >= value)
        return df.filter(ops[op_str](pl.col(col_name), value))

    return cols_to_lowercase, filter_by_comparison, std_cols_of_interest


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Prep.
    """)
    return


@app.cell
def _(
    cols_to_lowercase,
    filter_by_comparison,
    raw_howell_data,
    raw_waffle_data,
    std_cols_of_interest,
):
    waffle_data = raw_waffle_data.pipe(cols_to_lowercase).pipe(std_cols_of_interest)

    howell_adults = raw_howell_data.pipe(filter_by_comparison, "age", 18, ">=")

    howell_children = raw_howell_data.pipe(filter_by_comparison, "age", 13, "<=")

    # raw_howell_data, raw_waffle_data
    return howell_adults, waffle_data


@app.cell
def _(pl, plt, raw_howell_data):
    def _() -> plt.Axes:
        # 1. Create the boolean expression/mask
        is_adult = pl.col("age") >= 18

        # 2. Apply filtering
        adult_howell = raw_howell_data.filter(is_adult)
        child_howell = raw_howell_data.filter(~is_adult)

        plt.scatter(adult_howell["height"], adult_howell["weight"], label="over 18")
        plt.scatter(child_howell["height"], child_howell["weight"], label="under 18")

        plt.xlabel("Height (cm)")
        plt.ylabel("Weight (kg)")

        plt.legend()

        return plt.gca()


    # _()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
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
    # Lecture
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Helper Functions
    """)
    return


@app.cell
def _(SEX, az, howell_adults, np, pl, plt, pm, rng):
    def plot_howell_HW_dist(data: pl.DataFrame):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        plt.sca(axs[0])
        for sex_idx, label in enumerate(SEX):
            plt.scatter(
                data.filter(pl.col("male") == sex_idx)["height"],
                data.filter(pl.col("male") == sex_idx)["weight"],
                label=label,
                alpha=0.7,
                s=50,
            )
        plt.xlabel("Height (cm)")
        plt.ylabel("Weight (kg)")
        plt.title("Weight vs Height")
        plt.legend(title="Sex")

        for var_idx, col in enumerate(["height", "weight"]):
            plt.sca(axs[var_idx + 1])
            for sex_idx2, sex in enumerate(SEX):
                az.plot_dist(
                    data.filter(data["male"] == sex_idx2)[col],
                    label=sex,
                    color=f"C{sex_idx2}",
                )
            plt.ylabel(f"{col}".capitalize())
            plt.title(f"Dist. of {col.capitalize()} Split by Sex")
        return plt.gca()


    def sim_synthetic_people(
        sex_arr: np.array,
        alphas: np.array = np.array([0, 0]),
        betas: np.array = np.array([0.5, 0.6]),
        male_avg_height: float = 160.0,
        female_avg_height: float = 150.0,
    ) -> pl.DataFrame:
        """
        Simulates synthetic height and weight data based on sex.

        Args:
            sex_arr: An array of integers (0 for female, 1 for male) indicating sex.
            alphas: Intercepts for the weight linear model [female, male].
            betas: Slopes (height coefficients) for the weight linear model [female, male].
            male_avg_height: Mean height for males in cm.
            female_avg_height: Mean height for females in cm.

        Returns:
            A polars DataFrame containing simulated weight, height, and male indicator.
        """
        n_samples = len(sex_arr)
        h = np.where(sex_arr, male_avg_height, female_avg_height) + rng.normal(
            0, 5, n_samples
        )
        w = alphas[sex_arr] + betas[sex_arr] * h + rng.normal(0, 5, n_samples)

        return pl.DataFrame({"weight": w, "height": h, "male": sex_arr})


    def fit_total_effect_sex_weight(
        data: pl.DataFrame = howell_adults, draws: int = 100, prior: bool = True
    ) -> pm.Model:
        coords = {
            "obs": np.arange(data.shape[0]),
            "sex": ["F", "M"],  # Named categories
        }

        with pm.Model(coords=coords) as sex_weight_model:
            ## Data ##
            sex_idx = pm.Data("sex_cat", data["male"].to_numpy(), dims="obs")

            ## Priors ##
            alpha = pm.Normal("alpha", mu=60, sigma=10, dims="sex")
            sigma = pm.Uniform("sigma", lower=0, upper=10)

            ## Likelihood ##
            # MEMORY EFFICIENT: Define mu as a temporary variable (no pm.Deterministic)
            # This is used for math but NOT saved in the final results
            mu = alpha[sex_idx]

            # MEMORY EFFICIENT: Save only the contrast (size 1 per draw)
            # This gives you the effect size without saving a value for every row
            sex_diff = pm.Deterministic("sex_diff", alpha[1] - alpha[0])

            if prior:
                weight = pm.Normal("weight", mu=mu, sigma=sigma)
                idata = pm.sample_prior_predictive(draws, random_seed=rng)
            else:
                weight = pm.Normal("weight", mu=mu, sigma=sigma, observed=data["weight"])
                idata = pm.sample(draws, random_seed=rng)

            return idata


    def howell_SW_testing(
        n_samples: int = 200, **kwargs
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        females = rng.binomial(n=1, p=0, size=n_samples)
        males = rng.binomial(n=1, p=1, size=n_samples)

        sim_F = sim_synthetic_people(sex_arr=females, **kwargs)
        sim_M = sim_synthetic_people(sex_arr=males, **kwargs)

        return sim_F, sim_M


    def plot_post_dist_mean_and_weight_howell(idata: az.InferenceData) -> plt.Axes:
        fig, axs = plt.subplots(2, 2, figsize=(15, 8))

        plt.sca(axs[0, 0])
        az.plot_dist(idata["posterior"]["alpha"].sel(sex="F"), color="C0", label="Female")
        az.plot_dist(idata["posterior"]["alpha"].sel(sex="M"), color="C1", label="Male")
        plt.title("Posterior Dist of Mean Weight by Sex")
        plt.xlabel("Mean Weight (kg)")
        plt.ylabel("Density")
        plt.legend()

        plt.sca(axs[0, 1])
        _obs_weights_chapter = idata["observed_data"]["weight"].to_numpy()
        _sex_indicator_chapter = idata["constant_data"]["sex_cat"].to_numpy()

        az.plot_dist(
            _obs_weights_chapter[_sex_indicator_chapter == 0],
            color="C0",
            label="Female Obs",
        )
        az.plot_dist(
            _obs_weights_chapter[_sex_indicator_chapter == 1],
            color="C1",
            label="Male Obs",
        )
        plt.title("Posterior Weight")
        plt.xlabel("Weight (kg)")
        plt.legend()

        plt.sca(axs[1, 0])
        az.plot_dist(
            idata["posterior"]["sex_diff"].to_numpy().ravel(),
            color="black",
            label="Posterior Contrast (M - F)",
        )
        # plt.title("Observed Weight Contrast")
        plt.xlabel("Mean Weight Contrast (kg)")
        plt.legend()

        # Second row, Right: Remove the empty axis
        fig.delaxes(axs[1, 1])

        plt.tight_layout()

        return plt.gca()

    return (
        fit_total_effect_sex_weight,
        howell_SW_testing,
        plot_howell_HW_dist,
        plot_post_dist_mean_and_weight_howell,
        sim_synthetic_people,
    )


@app.cell
def _(howell_adults, plot_howell_HW_dist):
    plot_howell_HW_dist(data=howell_adults)
    return


@app.cell
def _(np, plot_howell_HW_dist, rng, sim_synthetic_people):
    synthetic_howell = sim_synthetic_people(
        sex_arr=rng.binomial(n=1, p=0.5, size=300), betas=np.array([0.5, 0.6])
    )

    plot_howell_HW_dist(data=synthetic_howell)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Total Effect of Sex on Weight

    Goal is to study the distribution of weight for different sex. The model is:

    $$W_i \sim \text{Normal}(\mu_i, \sigma)$$
    $$\mu_i = \alpha_{S[i]}$$
    $$\mu \sim \text{Normal}(60, 10)$$
    $$\sigma \sim \text{Uniform}(0, 10)$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Testing the Generative Model
    """)
    return


@app.cell
def _(az, howell_SW_testing, np, plt):
    sim_F, sim_M = howell_SW_testing(
        male_avg_height=160, female_avg_height=150, betas=np.array([0.5, 0.6])
    )

    sim_delta = sim_M - sim_F
    print(f"Mean difference is: {sim_delta['weight'].mean()}")

    az.plot_dist(sim_F["weight"], color="C0", label="Female")
    az.plot_dist(sim_M["weight"], color="C1", label="Male")
    az.plot_dist(sim_delta["weight"], color="black", label="Difference")

    plt.title("Distributions of Weight by Sex and Contrast")
    plt.xlabel("Weight (kg)")
    plt.gca()
    return sim_F, sim_M


@app.cell
def _(fit_total_effect_sex_weight, pl, sim_F, sim_M):
    howell_sim_idata = fit_total_effect_sex_weight(
        data=pl.concat([sim_F, sim_M]), draws=100, prior=False
    )

    howell_sim_idata["posterior"]["sex_diff"].mean()
    return (howell_sim_idata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The model is able to capture back the effect generate by the generative model. Seems to be working fine.
    """)
    return


@app.cell
def _(howell_sim_idata, plot_post_dist_mean_and_weight_howell):
    plot_post_dist_mean_and_weight_howell(howell_sim_idata)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Prior Predictive Simulation
    """)
    return


@app.cell
def _():
    # Will leave to later chapters
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Model Predictive Simulation
    """)
    return


@app.cell
def _(fit_total_effect_sex_weight, howell_adults):
    howell_idata = fit_total_effect_sex_weight(data=howell_adults, prior=False, draws=200)
    return (howell_idata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The model is  confident that there is a difference in mean between men and women in the Kalahari in the 1960s. The difference is between 5 and 8.5.
    """)
    return


@app.cell
def _(howell_idata, plot_post_dist_mean_and_weight_howell):
    plot_post_dist_mean_and_weight_howell(howell_idata)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Total  Direct Effect of Sex on Weight

    Goal is to study the distribution of weight for different sex. The model is:

    $$W_i \sim \text{Normal}(\mu_i, \sigma)$$
    $$\mu_i = \alpha_{S[i]}$$
    $$\mu \sim \text{Normal}(60, 10)$$
    $$\sigma \sim \text{Uniform}(0, 10)$$
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chapter Notes
    """)
    return


@app.cell
def _(plt, waffle_data):
    _, _ax = plt.subplots(1, 2, figsize=(14, 5))

    _ax[0].scatter(waffle_data["marriage"], waffle_data["divorce"])
    _ax[0].set_xlabel("Marriage Rate")
    _ax[0].set_ylabel("Divorce Rate")
    _ax[0].set_title("Divorce vs Marriage Rate")

    _ax[1].scatter(waffle_data["medianAgeMarriage"], waffle_data["divorce"])
    _ax[1].set_xlabel("Median Age Marriage")
    _ax[1].set_ylabel("Divorce Rate")
    _ax[1].set_title("Marriage vs Median Age of Marriage")
    return


@app.cell
def _(np, pm, waffle_data):
    def m5_1(draws: int = 50):
        with pm.Model() as m5_1:
            a = pm.Normal("a", 0, 0.2)
            bA = pm.Normal("bA", 0, 0.5)
            sigma = pm.Exponential("sigma", 1)
            mu = a + bA * waffle_data.get_column("medianAgeMarriage_std").to_numpy()

            div_rate = pm.Normal(
                "divorce_rate",
                mu=mu,
                sigma=sigma,
                observed=waffle_data["divorce_std"].to_numpy(),
            )

            prior_samples = pm.sample_prior_predictive(draws)

            idata = pm.sample(draws)

        x = np.linspace(
            waffle_data["medianAgeMarriage_std"].min(),
            waffle_data["medianAgeMarriage_std"].max(),
        )

        return prior_samples, idata, x


    # m5_1_prior, m5_1_idata, m5_1_x = m5_1(draws=50)

    # a_plot = m5_1_prior["prior"]["a"].to_numpy().flatten()
    # bA_plot = m5_1_prior["prior"]["bA"].to_numpy().flatten()
    # mu_plot = a_plot[:, None] + bA_plot[:, None] * m5_1_x

    # plt.plot(m5_1_x, mu_plot.T, c="g", alpha=0.4)
    return


@app.cell
def _(waffle_data):
    waffle_data.select(["medianAgeMarriage", "divorce", "marriage"]).corr()
    return


@app.cell
def _(az, m5_1_idata, np, plt, waffle_data):
    _a_samples = (
        m5_1_idata["posterior"]["a"].to_numpy().flatten()
    )  # shape is (4 * n_samples,)
    _bA_samples = (
        m5_1_idata["posterior"]["bA"].to_numpy().flatten()
    )  # shape is (4 * n_samples,)
    _x = np.linspace(
        waffle_data["medianAgeMarriage_std"].min(),
        waffle_data["medianAgeMarriage_std"].max(),
        waffle_data.shape[0],
    )

    # Raw data
    plt.scatter(
        waffle_data["medianAgeMarriage"], waffle_data["divorce"], c="red", label="waffle_data"
    )

    # MAP - convert back to original scale
    _divorce_mean = waffle_data["divorce"].mean()
    _divorce_std = waffle_data["divorce"].std()

    _map_line = (
        _a_samples.mean()
        + _bA_samples.mean() * waffle_data["medianAgeMarriage_std"].to_numpy()
    ) * _divorce_std + _divorce_mean

    plt.plot(
        waffle_data["medianAgeMarriage"],
        _map_line,
        c="green",
        lw=3,
        label="MAP regression line",
    )

    # 2. Manually calculate mu for each posterior sample in original scale
    # Broadcasting: (n_samples, 1) + (n_samples, 1) * (n_points,)
    # Result: (n_samples, n_points)
    _mu_std = (
        _a_samples[:, np.newaxis]
        + _bA_samples[:, np.newaxis] * waffle_data["medianAgeMarriage_std"].to_numpy()
    )
    _mu = _mu_std * _divorce_std + _divorce_mean

    # 89% HDI mean
    _mu_hdi = az.hdi(_mu, hdi_prob=0.89)
    az.plot_hdi(
        x=waffle_data["medianAgeMarriage"].to_numpy(), hdi_data=_mu_hdi, color="green"
    )

    plt.xlabel("MedianAgeMarriage")
    plt.ylabel("Divorce")
    plt.legend()
    plt.gca()
    return


@app.cell
def _(az, m5_1_idata):
    az.summary(m5_1_idata)
    return


if __name__ == "__main__":
    app.run()
