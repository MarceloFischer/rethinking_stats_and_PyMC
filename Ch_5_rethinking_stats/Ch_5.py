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

__generated_with = "0.22.0"
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
    import scipy.stats as statszl
    from pathlib import Path
    from wigglystuff import EdgeDraw

    RANDOM_SEED = 1523
    rng = np.random.default_rng(RANDOM_SEED)
    plt.style.use("fivethirtyeight")
    # Set default figure size to 10 inches wide by 6 inches tall
    plt.rcParams["figure.figsize"] = (14, 5)
    # You can also set the DPI (dots per inch) for crisper images
    plt.rcParams["figure.dpi"] = 100
    # Make the layout "tight" by default so labels don't overlap
    plt.rcParams["figure.autolayout"] = True

    az.rcParams["stats.ci_prob"] = (
        0.89  # sets default credible interval used by arviz
    )
    return EdgeDraw, Path, az, mo, np, operator, pl, plt, pm, rng


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
    MILK_PATH = Path(__file__).parent.parent / "data" / "milk.csv"

    # women=0 and men=1 in the dataset
    SEX = ["F", "M"]

    raw_waffle_data = pl.read_csv(WAFFLE_PATH, separator=";")
    raw_howell_data = pl.read_csv(HOWELL_PATH, separator=";")
    raw_milk_data = pl.read_csv(MILK_PATH, separator=";")
    return SEX, raw_howell_data, raw_milk_data, raw_waffle_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Helper Functions
    """)
    return


@app.cell
def _(operator, pl):
    def cols_to_lowercase(dataf: pl.DataFrame) -> pl.DataFrame:
        return dataf.rename(
            {col: col[0].lower() + col[1:] for col in dataf.columns}
        )


    def std_cols_of_interest(dataf: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
        # add standardised columns
        return dataf.with_columns(
            [
                ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(
                    f"{col}_std"
                )
                for col in cols
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


    def set_dtypes_float64(dataf: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
        return dataf.with_columns(
            [pl.col(col).cast(pl.Float64, strict=False) for col in cols]
        )

    return (
        cols_to_lowercase,
        filter_by_comparison,
        set_dtypes_float64,
        std_cols_of_interest,
    )


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
    raw_milk_data,
    raw_waffle_data,
    set_dtypes_float64,
    std_cols_of_interest,
):
    waffle_data = raw_waffle_data.pipe(cols_to_lowercase).pipe(
        std_cols_of_interest, ["divorce", "medianAgeMarriage", "marriage"]
    )

    howell_adults = raw_howell_data.pipe(filter_by_comparison, "age", 18, ">=")

    howell_children = raw_howell_data.pipe(filter_by_comparison, "age", 13, "<=")

    milk_data = raw_milk_data.pipe(
        set_dtypes_float64, ["kcal.per.g", "mass", "neocortex.perc"]
    ).pipe(std_cols_of_interest, ["kcal.per.g", "mass", "neocortex.perc"])

    # raw_howell_data, raw_waffle_data
    return howell_adults, waffle_data


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
def _(SEX, az, pl, plt):
    def plot_howell_HW_dist(data: pl.DataFrame):
        """
        Plots the relationship between Height and Weight and their distributions split by sex.

        Args:
            data: A polars DataFrame containing 'height', 'weight', and 'male' columns.

        Returns:
            The current matplotlib Axes object.
        """
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
            plt.xlabel(f"{col}".capitalize())
            plt.title(f"Dist. of {col.capitalize()} Split by Sex")
        return plt.gca()

    return (plot_howell_HW_dist,)


@app.cell
def _(np, pl, rng):
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

    return (sim_synthetic_people,)


@app.cell
def _(pl, rng, sim_synthetic_people):
    def howell_SW_testing(
        n_samples: int = 200, **kwargs
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Simulates balanced synthetic datasets for females and males for testing.

        Args:
            n_samples: Number of samples to generate per sex group.
            **kwargs: Arguments passed to sim_synthetic_people.

        Returns:
            A tuple of two Polars DataFrames: (female_sim, male_sim).
        """
        females = rng.binomial(n=1, p=0, size=n_samples)
        males = rng.binomial(n=1, p=1, size=n_samples)

        sim_F = sim_synthetic_people(sex_arr=females, **kwargs)
        sim_M = sim_synthetic_people(sex_arr=males, **kwargs)

        return sim_F, sim_M

    return (howell_SW_testing,)


@app.cell
def _(SEX, az, howell_adults, np, pl, pm, rng):
    def fit_total_effect_sex_weight(
        data: pl.DataFrame = howell_adults,
        draws: int = 100,
        prior: bool = True,
    ) -> tuple[az.InferenceData, pm.Model]:
        """
        Fits a Bayesian model to estimate the total effect of sex on weight.

        Args:
            data: Polars DataFrame containing the data.
            draws: Number of samples to draw.
            prior: If True, samples from the prior predictive; otherwise fits to data.

        Returns:
            A tuple containing the ArviZ InferenceData and the PyMC Model.
        """
        coords = {
            "obs": np.arange(data.shape[0]),
            "sex": SEX,  # Named categories
        }

        with pm.Model(coords=coords) as sex_weight_model:
            ## Data ##
            sex_idx = pm.Data("sex_idx", data["male"].to_numpy(), dims="obs")

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
                weight = pm.Normal(
                    "weight", mu=mu, sigma=sigma, observed=data["weight"]
                )
                idata = pm.sample(draws, random_seed=rng)

            return idata, sex_weight_model

    return


@app.cell
def _(SEX, az, howell_adults, np, pl, pm, rng):
    def fit_direct_effect_sex_weight(
        data: pl.DataFrame = howell_adults,
        draws: int = 100,
        prior: bool = True,
    ) -> tuple[az.InferenceData, pm.Model]:
        """
        Fits a Bayesian model to estimate the direct effect of sex on weight, controlling for height.

        Args:
            data: Polars DataFrame containing the data.
            draws: Number of samples to draw.
            prior: If True, samples from the prior predictive; otherwise fits to data.

        Returns:
            A tuple containing the ArviZ InferenceData and the PyMC Model.
        """
        coords = {
            "obs": np.arange(data.shape[0]),
            "sex": SEX,  # Named categories
        }

        with pm.Model(coords=coords) as direct_sex_weight_model:
            ## Data ##
            sex_idx = pm.Data("sex_idx", data["male"].to_numpy(), dims="obs")
            height = pm.Data("height", data["height"].to_numpy(), dims="obs")
            h_bar = pm.Data("h_bar", data["height"].mean())

            ## Priors ##
            alpha = pm.Normal("alpha", mu=60, sigma=10, dims="sex")
            beta = pm.Uniform("beta", lower=0, upper=1, dims="sex")
            sigma = pm.Uniform("sigma", lower=0, upper=10)

            ## Likelihood ##
            # MEMORY EFFICIENT: Define mu as a temporary variable (no pm.Deterministic)
            # This is used for math but NOT saved in the final results
            mu = alpha[sex_idx] + beta[sex_idx] * (height - h_bar)

            pm.Deterministic("sex_diff", alpha[1] - alpha[0])

            if prior:
                weight = pm.Normal("weight", mu=mu, sigma=sigma)
                idata = pm.sample_prior_predictive(draws, random_seed=rng)
            else:
                weight = pm.Normal(
                    "weight",
                    mu=mu,
                    sigma=sigma,
                    observed=data["weight"],
                    dims="obs",
                )
                idata = pm.sample(draws, random_seed=rng)

            return idata, direct_sex_weight_model

    return


@app.cell
def _(SEX, az, np, plt, rng):
    def plot_post_dist_mean_and_weight_howell(
        idata: az.InferenceData,
    ) -> plt.Axes:
        """
        Visualizes the posterior distribution of means, observed weights, and contrasts by sex.

        Args:
            idata: ArviZ InferenceData containing posterior and observed data.

        Returns:
            The current matplotlib Axes object.
        """
        fig, axs = plt.subplots(2, 2, figsize=(15, 8))

        ########################################
        # Posterior Dist of Mean Weight by Sex #
        ########################################
        plt.sca(axs[0, 0])
        for idx, s in enumerate(SEX):
            az.plot_dist(
                idata["posterior"]["alpha"].sel(sex=s),
                color=f"C{idx}",
                label=f"{s}",
            )
        plt.title("Posterior Dist of Mean Weight by Sex")
        plt.xlabel("Mean Weight (kg)")
        plt.ylabel("Density")
        plt.legend()

        ####################
        # Posterior Weight #
        ####################
        plt.sca(axs[0, 1])
        obs_weights = idata["observed_data"]["weight"].to_numpy()
        sex_mask = idata["constant_data"]["sex_idx"].to_numpy()
        weight_m = idata["observed_data"]["weight"].to_numpy()[sex_mask == 1]
        weight_f = idata["observed_data"]["weight"].to_numpy()[sex_mask == 0]

        az.plot_dist(
            weight_f,
            color="C0",
            label="Female Obs",
        )
        az.plot_dist(
            weight_m,
            color="C1",
            label="Male Obs",
        )
        plt.title("Posterior Weight")
        plt.xlabel("Weight (kg)")
        plt.legend()

        ########################
        # Mean Weight Contrast #
        ########################
        plt.sca(axs[1, 0])
        az.plot_dist(
            idata["posterior"]["sex_diff"].to_numpy().ravel(),
            color="black",
            label="Posterior Contrast (M - F)",
        )
        # plt.title("Observed Weight Contrast")
        plt.xlabel("Mean Weight Contrast (kg)")
        plt.legend()

        #####################################################################
        # Proportion of hevier man (contrast for the posterior weight dist) #
        #####################################################################
        plt.sca(axs[1, 1])
        # get the average for each chain, element-wise. If 4 chains with k draws, will return a 1D array with k elements.
        avg_m_alpha = idata["posterior"]["alpha"].sel(sex="M").mean(dim=["chain"])
        avg_f_alpha = idata["posterior"]["alpha"].sel(sex="F").mean(dim=["chain"])
        avg_sigma = idata["posterior"]["sigma"].mean(dim="chain")
        male_post_samples = rng.normal(
            avg_m_alpha, avg_sigma
        )  # k draws from the normal
        female_post_samples = rng.normal(
            avg_f_alpha, avg_sigma
        )  # k draws from the normal
        post_weight_contrast = male_post_samples - female_post_samples

        post_weight_contrast_plot = az.plot_dist(
            post_weight_contrast, color="black"
        )

        ##################################################
        # Shade underneath posterior predictive contrast #
        ##################################################
        kde_x, kde_y = post_weight_contrast_plot.get_lines()[0].get_data()
        n_draws = len(post_weight_contrast)

        # Proportion of PPD contrast below zero
        neg_idx = kde_x < 0
        neg_prob = 100 * np.sum(post_weight_contrast < 0) / n_draws
        plt.fill_between(
            x=kde_x[neg_idx],
            y1=np.zeros(sum(neg_idx)),
            y2=kde_y[neg_idx],
            color="C0",
            alpha=0.5,
            label=f"{neg_prob:1.0f}%",
        )

        # Proportion of PPD contrast above zero (inclusive)
        pos_idx = kde_x >= 0
        pos_prob = 100 * np.sum(post_weight_contrast >= 0) / n_draws
        plt.fill_between(
            x=kde_x[pos_idx],
            y1=np.zeros(sum(pos_idx)),
            y2=kde_y[pos_idx],
            color="C1",
            alpha=0.5,
            label=f"{pos_prob:1.0f}%",
        )

        plt.xlabel("Posterior Weight Contrast (kg) [M - F]")
        plt.legend()

        plt.tight_layout()

        return plt.gca()

    return (plot_post_dist_mean_and_weight_howell,)


@app.cell
def _(SEX, az, howell_adults, pl, plt):
    def plot_direct_sw_howell(
        idata: az.InferenceData,
        original_data: pl.DataFrame = howell_adults,
    ) -> plt.Axes:
        """
        Plots the regression lines and original data points for the direct effect model.

        Args:
            idata: ArviZ InferenceData containing posterior and constant data.
            original_data: The original Polars DataFrame to plot the scatter points from.

        Returns:
            The current matplotlib Axes object.
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        posterior = idata["posterior"]
        obs_data = idata["observed_data"]
        cte_data = idata["constant_data"]

        for idx, s in enumerate(SEX):
            plt.plot(
                original_data["height"],
                posterior["alpha"].sel(sex=s).mean()
                + posterior["beta"].sel(sex=s).mean()
                * (cte_data["height"] - cte_data["h_bar"]),
                label=f"{s}",
                c=f"C{idx}",
            )
            plt.scatter(
                original_data.filter(pl.col("male") == idx)["height"],
                original_data.filter(pl.col("male") == idx)["weight"],
                c=f"C{idx}",
                label=f"{s}",
            )
        plt.legend()

        return plt.gca()

    return (plot_direct_sw_howell,)


@app.cell
def _(howell_adults, plot_howell_HW_dist):
    plot_howell_HW_dist(data=howell_adults)
    return


@app.cell
def _(np, plot_howell_HW_dist, rng, sim_synthetic_people):
    synthetic_howell = sim_synthetic_people(
        sex_arr=rng.binomial(n=1, p=0.5, size=300),
        betas=np.array([0.5, 0.6]),
    )

    plot_howell_HW_dist(data=synthetic_howell)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Total Effect of Sex on Weight

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
    sim_total_F, sim_total_M = howell_SW_testing(
        male_avg_height=160, female_avg_height=150, betas=np.array([0.5, 0.6])
    )

    sim_total_delta = sim_total_M - sim_total_F
    print(f"Mean difference is: {sim_total_delta['weight'].mean()}")

    az.plot_dist(sim_total_F["weight"], color="C0", label="Female")
    az.plot_dist(sim_total_M["weight"], color="C1", label="Male")
    az.plot_dist(sim_total_delta["weight"], color="black", label="Difference")

    plt.title("Distributions of Weight by Sex and Contrast")
    plt.xlabel("Weight (kg)")
    plt.gca()
    return


@app.cell
def _():
    # howell_sim_idata, howell_sim_model = fit_total_effect_sex_weight(
    #     data=pl.concat([sim_F, sim_M]), draws=100, prior=False
    # )

    # howell_sim_idata["posterior"]["sex_diff"].mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The model is able to capture back the effect generate by the generative model. Seems to be working fine.
    """)
    return


@app.cell
def _():
    # plot_post_dist_mean_and_weight_howell(howell_sim_idata)
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
def _():
    # howell_total_sw_idata, howell_total_sw_model = fit_total_effect_sex_weight(
    #     data=howell_adults, prior=False, draws=1000
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The model is  confident that there is a difference in mean between men and women in the Kalahari in the 1960s. The difference is between 5 and 8.5.
    """)
    return


@app.cell
def _(howell_total_sw_idata, plot_post_dist_mean_and_weight_howell):
    plot_post_dist_mean_and_weight_howell(howell_total_sw_idata)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Total  Direct Effect of Sex on Weight

    Goal is to study the distribution of weight for different sex. The model is:

    $$ W_i \sim \text{Normal}(\mu_i, \sigma) $$
    $$ \mu_i = \alpha_{S[i]} + \beta_{S[i]}*(h_i-\bar h) $$
    $$ \mu_j \sim \text{Normal}(60, 10) $$
    $$ \beta_j \sim \text{Uniform}(0, 1) $$
    $$ \sigma \sim \text{Uniform}(0, 10) $$
    """)
    return


@app.cell
def _(az, howell_SW_testing, np, plt):
    sim_direct_F, sim_direct_M = howell_SW_testing(
        male_avg_height=160,
        female_avg_height=150,
        alphas=np.array([0, 10]),
        betas=np.array([0.5, 0.6]),
    )

    sim_direct_delta = sim_direct_M - sim_direct_F
    print(f"Mean difference is: {sim_direct_delta['weight'].mean()}")

    az.plot_dist(sim_direct_F["weight"], color="C0", label="Female")
    az.plot_dist(sim_direct_M["weight"], color="C1", label="Male")
    az.plot_dist(sim_direct_delta["weight"], color="black", label="Difference")

    plt.title("Distributions of Weight by Sex and Contrast")
    plt.xlabel("Weight (kg)")
    plt.gca()
    return sim_direct_F, sim_direct_M


@app.cell
def _(pl, plot_howell_HW_dist, sim_direct_F, sim_direct_M):
    plot_howell_HW_dist(data=pl.concat([sim_direct_F, sim_direct_M]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Model Predictive Simulation
    """)
    return


@app.cell
def _():
    # howell_direct_sw_idata, howell_direct_sw_model = fit_direct_effect_sex_weight(
    #     data=howell_adults, prior=False, draws=1000
    # )
    return


@app.cell
def _(howell_direct_sw_idata, plot_direct_sw_howell):
    plot_direct_sw_howell(idata=howell_direct_sw_idata)
    return


@app.cell
def _(howell_direct_sw_idata, plot_post_dist_mean_and_weight_howell):
    plot_post_dist_mean_and_weight_howell(idata=howell_direct_sw_idata)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The above suggests that almost all of the causal effect of sex acts through height. Just sex does not change how likely it is to be heavier or lighter for this sample.

    This might not be true in other populations, and it might also not be true for this same population at a different point in time!
    """)
    return


@app.cell(hide_code=True)
def _():
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter Notes
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Helper Functions
    """)
    return


@app.cell
def _(plt, waffle_data):
    def plot_divorce_marriage_age() -> plt.Axes:
        _, _ax = plt.subplots(1, 2)

        _ax[0].scatter(waffle_data["marriage"], waffle_data["divorce"])
        _ax[0].set_xlabel("Marriage Rate")
        _ax[0].set_ylabel("Divorce Rate")
        _ax[0].set_title("Divorce vs Marriage Rate")

        _ax[1].scatter(waffle_data["medianAgeMarriage"], waffle_data["divorce"])
        _ax[1].set_xlabel("Median Age Marriage")
        _ax[1].set_ylabel("Divorce Rate")
        _ax[1].set_title("Marriage vs Median Age of Marriage")

        return plt.gca()

    return (plot_divorce_marriage_age,)


@app.cell
def _(az, np, pm, rng):
    def run_linear_model(
        predictors: list[np.array],
        predictors_names: list[str],
        outcome: np.array,
        outcome_name: str,
        prior_predictive: bool = False,
        draws: int = 100,
    ) -> az.InferenceData:
        """
        Fits a Bayesian linear regression model using PyMC.

        Args:
            predictors: A list of numpy arrays, each representing a predictor variable.
            predictors_names: A list of strings containing the names of the predictors.
            outcome: A numpy array containing the outcome variable.
            outcome_name: A string name for the outcome variable (used in the model).
            prior_predictive: If True, samples from the prior predictive distribution instead of the posterior.
            draws: Number of samples to draw.

        Returns:
            An ArViz InferenceData object containing the model samples.
        """
        coords = {
            "predictors": predictors_names,
            "obs_id": np.arange(len(outcome)),
        }
        # Make every column of the predictors list of arrays a predictor.
        # Done so that the dot product ccan work. Every predictor multiply one beta.
        predictors = np.vstack([*predictors]).T

        with pm.Model(coords=coords) as model:
            # Data
            x_data = pm.Data("x_data", predictors, dims=("obs_id", "predictors"))

            # Priors
            alpha = pm.Normal("alpha", 0, 0.2)
            beta = pm.Normal("beta", mu=0, sigma=0.5, dims="predictors")
            sigma = pm.Exponential("sigma", lam=1)

            # Linear Model: mu = alpha + X * beta
            # PyMC handles the matrix multiplication perfectly here
            mu = alpha + pm.math.dot(x_data, beta)

            # Likelihood
            obs = pm.Normal(
                outcome_name, mu=mu, sigma=sigma, observed=outcome, dims="obs_id"
            )
            if prior_predictive:
                idata = pm.sample_prior_predictive(draws=draws, random_seed=rng)
            else:
                idata = pm.sample(draws=draws, random_seed=rng)

        return idata

    return (run_linear_model,)


@app.cell
def _(az, np, plt, waffle_data):
    def plot_simple_regression_on_original_scale(
        idata, predictor_col, outcome_col="divorce", data=waffle_data
    ):
        """
        Plots the MAP regression line and 89% HDI for a given model,
        transforming standardized predictions back to the original scale.

        Args:
            idata: ArViz InferenceData object containing posterior samples.
            predictor_col: The string name of the original scale predictor column (e.g., 'medianAgeMarriage').
            outcome_col: The string name of the original scale outcome column. Defaults to 'divorce'.
            data: Polars or Pandas DataFrame containing the raw data.
        """
        _a_samples = idata["posterior"]["alpha"].to_numpy().flatten()
        _b_samples = idata["posterior"]["beta"].to_numpy().flatten()

        _raw_x = data[predictor_col].to_numpy()
        _std_x = data[f"{predictor_col}_std"].to_numpy()
        _raw_y = data[outcome_col].to_numpy()

        # Scaling parameters
        _y_mean = data[outcome_col].mean()
        _y_std = data[outcome_col].std()

        # Raw data
        plt.scatter(
            _raw_x,
            _raw_y,
            c="red",
            alpha=0.6,
            label=f"raw_{outcome_col}_data",
        )

        # MAP - convert back to original scale
        # Result: (n_points,)
        _map_line_std = _a_samples.mean() + _b_samples.mean() * _std_x
        _map_line = _map_line_std * _y_std + _y_mean

        # Sort for plotting lines properly
        _sort_idx = np.argsort(_raw_x)
        plt.plot(
            _raw_x[_sort_idx],
            _map_line[_sort_idx],
            c="green",
            lw=3,
            label="MAP regression line",
        )

        # Calculate mu for each posterior sample in original scale
        # Broadcasting: (n_samples, 1) + (n_samples, 1) * (n_points,)
        _mu_std = _a_samples[:, np.newaxis] + _b_samples[:, np.newaxis] * _std_x
        _mu = _mu_std * _y_std + _y_mean

        # 89% HDI mean
        _mu_hdi = az.hdi(_mu, hdi_prob=0.89)
        az.plot_hdi(
            x=_raw_x,
            hdi_data=_mu_hdi,
            color="green",
        )

        plt.xlabel(predictor_col)
        plt.ylabel(outcome_col)
        plt.title(f"Regression of {outcome_col} on {predictor_col}")
        plt.legend()
        return plt.gca()

    return (plot_simple_regression_on_original_scale,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Section 5.1
    """)
    return


@app.cell
def _(waffle_data):
    WAFFLE_OUTCOME = waffle_data["divorce_std"].to_numpy()
    return (WAFFLE_OUTCOME,)


@app.cell
def _(plot_divorce_marriage_age):
    plot_divorce_marriage_age()
    return


@app.cell(hide_code=True)
def _(EdgeDraw, mo):
    divorce_age_model_latex = mo.md(r"""
                        \begin{align*}
                        D_i &\sim \mathcal{N}(\mu_i, \sigma)\\
                        \mu_i &= \alpha + \beta_A A_i\\
                        \alpha &\sim \mathcal{N}(0,\,0.2)\\
                        \beta_A &\sim \mathcal{N}(0,\,0.5)\\
                        \sigma &\sim \text{Exponential}(1)
                        \end{align*}
                        """)
    divorce_age_graph = EdgeDraw(names=["D", "A"], directed=True)

    mo.hstack(
        [
            mo.vstack([mo.md(" "), divorce_age_model_latex], justify="center"),
            divorce_age_graph,
        ],
    )
    return


@app.cell
def _(WAFFLE_OUTCOME, np, plt, run_linear_model, waffle_data):
    m5_1_prior = run_linear_model(
        predictors=[waffle_data["medianAgeMarriage_std"].to_numpy()],
        predictors_names=["median_age_std"],
        outcome=WAFFLE_OUTCOME,
        outcome_name="Divorce_std",
        prior_predictive=True,
    )

    m5_1_x = np.linspace(
        waffle_data["medianAgeMarriage_std"].min(),
        waffle_data["medianAgeMarriage_std"].max(),
    )

    alpha_plot = m5_1_prior["prior"]["alpha"].to_numpy().flatten()
    beta_plot = m5_1_prior["prior"]["beta"].to_numpy().flatten()
    mu_plot = alpha_plot[:, None] + beta_plot[:, None] * m5_1_x

    plt.plot(m5_1_x, mu_plot.T, c="g", alpha=0.4)
    return


@app.cell
def _(waffle_data):
    waffle_data.select(["medianAgeMarriage", "divorce", "marriage"]).corr()
    return


@app.cell
def _(WAFFLE_OUTCOME, run_linear_model, waffle_data):
    m5_1_idata = run_linear_model(
        predictors=[waffle_data["medianAgeMarriage_std"].to_numpy()],
        predictors_names=["median_age_std"],
        outcome=WAFFLE_OUTCOME,
        outcome_name="Divorce_std",
        prior_predictive=False,
        draws=1000,
    )
    return (m5_1_idata,)


@app.cell
def _(m5_1_idata, plot_simple_regression_on_original_scale):
    plot_simple_regression_on_original_scale(m5_1_idata, "medianAgeMarriage")
    return


@app.cell
def _(az, m5_1_idata):
    az.summary(m5_1_idata, kind="stats")
    return


@app.cell(hide_code=True)
def _(EdgeDraw, mo):
    divorce_marriage_model_latex = mo.md(r"""
                        \begin{align*}
                        D_i &\sim \mathcal{N}(\mu_i, \sigma)\\
                        \mu_i &= \alpha + \beta_M M_i\\
                        \alpha &\sim \mathcal{N}(0,\,0.2)\\
                        \beta_A &\sim \mathcal{N}(0,\,0.5)\\
                        \sigma &\sim \text{Exponential}(1)
                        \end{align*}
                        """)
    divorce_marriage_graph = EdgeDraw(names=["Divorce", "Marriage"], directed=True)

    mo.hstack(
        [
            mo.vstack(
                [mo.md(" "), divorce_marriage_model_latex], justify="center"
            ),
            divorce_marriage_graph,
        ],
    )
    return


@app.cell
def _(WAFFLE_OUTCOME, run_linear_model, waffle_data):
    m5_2_idata = run_linear_model(
        predictors=[waffle_data["marriage_std"].to_numpy()],
        predictors_names=["marriage_std"],
        outcome=WAFFLE_OUTCOME,
        outcome_name="Divorce_std",
        prior_predictive=False,
        draws=1000,
    )
    return (m5_2_idata,)


@app.cell
def _(m5_2_idata, plot_simple_regression_on_original_scale):
    plot_simple_regression_on_original_scale(m5_2_idata, "marriage")
    return


@app.cell
def _(az, m5_2_idata):
    az.summary(m5_2_idata, kind="stats")
    return


@app.cell(hide_code=True)
def _(EdgeDraw):
    divorce_age_marriage_graph = EdgeDraw(
        names=["Divorce", "Age", "Marriage"], directed=True
    )
    divorce_age_marriage_graph
    return


@app.cell
def _(WAFFLE_OUTCOME, run_linear_model, waffle_data):
    m5_3_idata = run_linear_model(
        predictors=[
            waffle_data["marriage_std"].to_numpy(),
            waffle_data["medianAgeMarriage_std"].to_numpy(),
        ],
        predictors_names=["marriage_std", "median_age_marriage_std"],
        outcome=WAFFLE_OUTCOME,
        outcome_name="Divorce_std",
        prior_predictive=False,
        draws=1000,
    )
    return (m5_3_idata,)


@app.cell
def _(az, m5_3_idata):
    az.summary(m5_3_idata, kind="stats")
    return


@app.cell
def _(az, m5_1_idata, m5_2_idata, m5_3_idata):
    az.plot_forest(
        [
            m5_3_idata,
            m5_2_idata,
            m5_1_idata,
        ],
        model_names=["both", "marriage", "age"],
        var_names=["beta"],
        combined=True,
        figsize=(10, 5),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Section 5.2
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
