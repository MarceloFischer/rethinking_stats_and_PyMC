# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "arviz==0.23.4",
#     "marimo>=0.22.4",
#     "matplotlib==3.10.8",
#     "numpy==2.4.4",
#     "polars==1.39.3",
#     "pymc==5.28.3",
#     "scipy==1.17.1",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="columns")

with app.setup:
    import numpy as np
    import polars as pl
    import pymc as pm
    import scipy.stats as stats
    import arviz as az
    import matplotlib.pyplot as plt

    RANDOM_SEED = 1523
    rng = np.random.default_rng(RANDOM_SEED)


@app.function
def remove_period_col_name(dataf: pl.DataFrame) -> pl.DataFrame:
    return dataf.rename(
        {col: col.replace(".", "_") for col in dataf.columns}
    )


@app.function
def cols_to_lowercase(dataf: pl.DataFrame) -> pl.DataFrame:
    return dataf.rename(
        {col: col[0].lower() + col[1:] for col in dataf.columns}
    )


@app.function
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


@app.function
def std_log_mass(dataf: pl.DataFrame) -> pl.DataFrame:
    # add standardised columns
    return dataf.with_columns(log_mass=pl.col("mass").log()).with_columns(
        log_mass_std=(pl.col("log_mass") - pl.col("log_mass").mean())
        / pl.col("log_mass").std()
    )


@app.function
def set_dtypes_float64(dataf: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return dataf.with_columns(
        [pl.col(col).cast(pl.Float64, strict=False) for col in cols]
    )


@app.function
def plot_simple_regression_on_chosen_scale(
    idata,
    data,
    predictor_col,
    outcome_col="divorce",
    use_std=False,
):
    """
    Plots the MAP regression line and 89% HDI for a given model.
    Optionally transforms standardized predictions back to the original scale.

    Args:
        idata: ArViz InferenceData object containing posterior samples.
        predictor_col: The string name of the original scale predictor column (e.g., 'medianAgeMarriage').
        outcome_col: The string name of the original scale outcome column. Defaults to 'divorce'.
        data: Polars or Pandas DataFrame containing the raw data.
        use_std: If True, plots on the standardized scale. If False, plots on the original scale.
    """
    alpha_samples = idata["posterior"]["alpha"].to_numpy().flatten()
    beta_samples = idata["posterior"]["beta"].to_numpy().flatten()

    std_x = data[f"{predictor_col}_std"].to_numpy()
    std_y = data[f"{outcome_col}_std"].to_numpy()

    raw_x = data[predictor_col].to_numpy()
    raw_y = data[outcome_col].to_numpy()

    # Determine which scale to use for plotting
    plot_x = std_x if use_std else raw_x
    plot_y = std_y if use_std else raw_y
    x_label = f"{predictor_col}_std" if use_std else predictor_col
    y_label = f"{outcome_col}_std" if use_std else outcome_col

    # Scaling parameters for the outcome
    y_mean = data[outcome_col].mean()
    y_std = data[outcome_col].std()

    # Raw data points
    plt.scatter(
        plot_x,
        plot_y,
        c="red",
        s=50,
        alpha=0.6,
        label=f"raw {outcome_col} data",
    )

    # MAP calculation in standardized scale first
    map_line_std = alpha_samples.mean() + beta_samples.mean() * std_x

    # Calculate mu for each posterior sample in standardized scale
    # Broadcasting: (n_samples, 1) + (n_samples, 1) * (n_points,)
    mu_std = alpha_samples[:, np.newaxis] + beta_samples[:, np.newaxis] * std_x

    if use_std:
        map_line = map_line_std
        mu = mu_std
    else:
        # Transform back to original scale
        map_line = map_line_std * y_std + y_mean
        mu = mu_std * y_std + y_mean

    # Sort for plotting lines properly
    sort_idx = np.argsort(plot_x)
    plt.plot(
        plot_x[sort_idx],
        map_line[sort_idx],
        c="green",
        lw=3,
        label="MAP regression line",
    )

    # 89% HDI
    mu_hdi = az.hdi(mu, hdi_prob=0.89)
    az.plot_hdi(
        x=plot_x,
        hdi_data=mu_hdi,
        color="green",
    )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(
        f"Regression of {outcome_col} on {predictor_col} ({'Standardized' if use_std else 'Original'} Scale)"
    )
    plt.legend()
    return plt.gca()


@app.function
def plot_linear_regression_prior_predictive(
    idata: az.InferenceData,
    predictor_col: str,
    outcome_col: str,
    data: pl.DataFrame,
) -> plt.Axes:
    x_vals = np.linspace(-2, 2)

    alpha_plot = idata["prior"]["alpha"].to_numpy().flatten()
    beta_plot = idata["prior"]["beta"].to_numpy().flatten()
    mu_plot = alpha_plot[:, None] + beta_plot[:, None] * x_vals

    plt.plot(x_vals, mu_plot.T, c="g", alpha=0.4)

    # Add context lines for standard Normal scale if applicable
    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(0, color="gray", linestyle="--", alpha=0.5)

    plt.ylim(-2, 2)
    plt.xlabel(predictor_col)
    plt.ylabel(outcome_col)

    return plt.gca()


@app.function
def plot_counterfactual(
    post_trace: az.InferenceData,
    predictor_name: str = "neocortex.perc_std",
    control_name: str = "log_mass_std",
) -> plt.Axes:
    # 1. Create counterfactual data: vary predictor from -2 to 2, hold control at 0
    seq = np.linspace(-2, 2, 50)

    # 2. Extract posterior samples
    post = post_trace.posterior
    alpha = post["alpha"].values  # shape (chains, draws)
    beta_pred = (
        post["beta"].sel(predictors=predictor_name).values
    )  # shape (chains, draws)
    # beta_control is extracted but term drops out because control value is 0

    # 3. Calculate mu for each point in seq across all samples
    # Broadcast to (chains, draws, 50)
    mu_pred = alpha[:, :, np.newaxis] + beta_pred[:, :, np.newaxis] * seq

    # 4. Summarize the predictions
    mu_mean = mu_pred.mean(axis=(0, 1))  # avg chain/draws
    mu_samples = mu_pred.reshape(
        -1, 50
    )  # flatten chain/draws but keep the 50 x_values (total_draws, 50)
    mu_hdi = az.hdi(mu_samples, hdi_prob=0.89)

    # 5. Plot
    plt.plot(seq, mu_mean, label=f"Counterfactual: {predictor_name}")
    az.plot_hdi(seq, mu_hdi.T)

    plt.xlabel(f"{predictor_name} (std)")
    plt.ylabel("Predicted kcal per g (std)")
    plt.title(f"Effect of {predictor_name} (holding {control_name} constant)")
    plt.legend()

    return plt.gca()


@app.function
def run_linear_model(
    predictors: list[np.array],
    predictors_names: list[str],
    outcome: np.array,
    outcome_name: str,
    prior_predictive: bool = False,
    draws: int = 100,
    alpha: float = 0.2,
    beta: float = 0.5,
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
        alpha: Standard deviation for the intercept prior.
        beta: Standard deviation for the predictor coefficients prior.

    Returns:
        An ArViz InferenceData object containing the model samples.
    """
    coords = {
        "predictors": predictors_names,
        "obs_id": np.arange(len(outcome)),
    }
    # Make every column of the predictors list of arrays a predictor.
    # Done so that the dot product can work. Every predictor multiply one beta.
    predictors = np.vstack([*predictors]).T

    with pm.Model(coords=coords) as model:
        # Data
        x_data = pm.Data("x_data", predictors, dims=("obs_id", "predictors"))

        # Priors
        alpha = pm.Normal("alpha", 0, alpha)
        beta = pm.Normal("beta", mu=0, sigma=beta, dims="predictors")
        sigma = pm.Exponential("sigma", lam=1)

        # Linear Model: mu = alpha + X * beta
        mu = alpha + pm.math.dot(x_data, beta)

        # Likelihood
        obs = pm.Normal(
            outcome_name,
            mu=mu,
            sigma=sigma,
            observed=outcome,
            dims="obs_id",
        )
        if prior_predictive:
            idata = pm.sample_prior_predictive(draws=draws, random_seed=rng)
        else:
            idata = pm.sample(draws=draws, random_seed=rng)

    return idata


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


if __name__ == "__main__":
    app.run()
