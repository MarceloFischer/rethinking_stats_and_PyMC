import marimo

__generated_with = "0.17.8"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports and Constants
    """)
    return


@app.cell
def _():
    import altair as alt
    import arviz as az
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import pymc as pm
    import scipy.stats as stats

    RANDOM_SEED = 42
    rng = np.random.default_rng(RANDOM_SEED)
    az.style.use("arviz-darkgrid")
    az.rcParams["stats.ci_prob"] = 0.89  # sets default credible interval used by arviz
    return az, mo, np, pl, plt, pm, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Cleaning
    """)
    return


@app.cell
def _(pl):
    raw_data = pl.read_csv(
        "/home/marcelo/git/rethinking_stats_and_PyMC/data/Howell1.csv", separator=";"
    )

    def filter_data(df: pl.DataFrame):
        return df.filter(pl.col("age") >= 18)

    data = raw_data.pipe(filter_data)

    MEAN_W = data.select("weight").mean().item()

    data
    return MEAN_W, data


@app.cell
def _(np, plt, stats):
    _x = np.linspace(100, 250, 100)
    plt.plot(_x, stats.norm.pdf(_x, 178, 20))

    _x = np.linspace(-10, 60, 100)
    plt.plot(_x, stats.uniform.pdf(_x, 0, 50))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Our Model

    $h_i \sim \text{Normal}(\mu, \sigma)$

    $\mu \sim \text{Normal}(178, 20)$

    $\sigma \sim \text{Uniform}(0, 50)$
    """)
    return


@app.cell
def _(az, stats):
    # We can use our model to check if priors make sense before actually feeding any data in.
    # note that the code is written in opposite order as the mathematical definitions

    _n = 1000
    _mu = stats.norm.rvs(loc=178, scale=20, size=_n)
    _sigma = stats.uniform.rvs(loc=0, scale=1, size=_n)
    _prior_h = stats.norm.rvs(loc=_mu, scale=_sigma, size=_n)

    az.plot_kde(_prior_h)
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quadratic Approximation
    """)
    return


@app.cell
def _(data, pm):
    with pm.Model() as m4_1:
        # initival can be given to tell the algo where to start looking for the peak
        _mu = pm.Normal(
            "mu", mu=178, sigma=20, initval=data.select("height").mean().item()
        )
        _sigma = pm.Uniform(
            "sigma", lower=0, upper=50, initval=data.select("height").std().item()
        )
        _height = pm.Normal(
            "height", mu=_mu, sigma=_sigma, observed=data.select("height")
        )
        idata_4_1 = pm.sample(1_000, tune=1_000)
    return (idata_4_1,)


@app.cell
def _(az, idata_4_1):
    az.plot_trace(idata_4_1), az.summary(idata_4_1, round_to=2, kind="stats")
    return


@app.cell
def _(az, idata_4_1):
    _idata_df = az.extract(idata_4_1).to_dataframe()
    _idata_df, _idata_df.cov(), _idata_df.corr()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Linear Prediction
    """)
    return


@app.cell
def _(data, plt):
    plt.plot(data.select("height"), data.select("weight"), ".")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    \begin{align*}
    h_i &\sim \text{Normal}(\mu_i, \sigma) & [\text{likelihood}] \\
    \mu_i &= \alpha + \beta w_i \quad  (\beta(x_i - \bar{x})) & [\text{linear model}] \\
    \alpha &\sim \text{Normal}(178, 20) & [\alpha \text{ prior}] \\
    \beta &\sim \text{Normal}(0, 10) & [\beta \text{ prior}] \\
    \sigma &\sim \text{Uniform}(0, 50) & [\sigma \text{ prior}]
    \end{align*}

    The golem is assuming (by using a Gaussian) that the relationship to construct the mean is additive. We do grown a bit every year/day/minute and the weight, on avg, increases a bit as well.

    As the image below show, this does not make a good model as it allows or negative slopes and also slopes that are clearly not resoanble.

    Let's restrict beta to only positive values as:

    \begin{align*}
    h_i &\sim \text{Normal}(\mu_i, \sigma) & [\text{likelihood}] \\
    \mu_i &= \alpha + \beta w_i & [\text{linear model}] \\
    \alpha &\sim \text{Normal}(178, 20) & [\alpha \text{ prior}] \\
    \beta &\sim \text{Log-Normal}(0, 10) & [\beta \text{ prior}] \\
    \sigma &\sim \text{Uniform}(0, 50) & [\sigma \text{ prior}]
    \end{align*}

    Another way to is to expect that the slope would be between 0 and 1. Weight is in kg and height is in cm. Usually, weight is less than height.

    \begin{align*}
    h_i &\sim \text{Normal}(\mu_i, \sigma) & [\text{likelihood}] \\
    \mu_i &= \alpha + \beta w_i & [\text{linear model}] \\
    \alpha &\sim \text{Normal}(178, 20) & [\alpha \text{ prior}] \\
    \beta &\sim \text{Uniform}(0, 1) & [\beta \text{ prior}] \\
    \sigma &\sim \text{Uniform}(0, 50) & [\sigma \text{ prior}]
    \end{align*}
    """)
    return


@app.cell
def _(MEAN_W, data, np, plt, stats):
    # to see the effects of beta (prior) and understand what it entails, we need to simulate several heights using beta

    _N = 100  # 100 lines
    _alpha = stats.norm.rvs(loc=178, scale=20, size=_N)
    _, _ax = plt.subplots(1, 3, sharey=True, figsize=(10, 5))
    _x = np.linspace(
        data.select("weight").min().item(), data.select("weight").max().item(), _N
    )

    _beta = stats.norm.rvs(loc=0, scale=10, size=_N)
    for _i in range(_N):
        _ax[0].plot(_x, _alpha[_i] + _beta[_i] * (_x - MEAN_W), "k", alpha=0.2)
        _ax[0].set_xlim(
            data.select("weight").min().item(), data.select("weight").max().item()
        )
        _ax[0].set_ylim(-100, 400)
        _ax[0].axhline(0, c="k", ls="--")
        _ax[0].axhline(272, c="k")
        _ax[0].set_xlabel("weight")
        _ax[0].set_ylabel("height")

    _beta = stats.lognorm.rvs(s=1, scale=1, size=_N)
    for _i in range(_N):
        _ax[1].plot(_x, _alpha[_i] + _beta[_i] * (_x - MEAN_W), "k", alpha=0.2)
        _ax[1].set_xlim(
            data.select("weight").min().item(), data.select("weight").max().item()
        )
        _ax[1].set_ylim(-100, 400)
        _ax[1].axhline(0, c="k", ls="--", label="embryo")
        _ax[1].axhline(272, c="k")
        _ax[1].set_xlabel("weight")
        _ax[1].text(x=35, y=282, s="World's tallest person (272cm)")
        _ax[1].text(x=35, y=-25, s="Embryo")

    _beta = stats.uniform.rvs(loc=0, scale=1, size=_N)
    for _i in range(_N):
        _ax[2].plot(_x, _alpha[_i] + _beta[_i] * (_x - MEAN_W), "k", alpha=0.2)
        _ax[2].set_xlim(
            data.select("weight").min().item(), data.select("weight").max().item()
        )
        _ax[2].set_ylim(-100, 400)
        _ax[2].axhline(0, c="k", ls="--", label="embryo")
        _ax[2].axhline(272, c="k")
        _ax[2].set_xlabel("weight")

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    \begin{align*}
    h_i &\sim \text{Normal}(\mu_i, \sigma) & [\text{likelihood}] \\
    \mu_i &= \alpha + \beta(x_i - \bar{x}) & [\text{linear model}] \\
    \alpha &\sim \text{Normal}(178, 20) & [\alpha \text{ prior}] \\
    \beta &\sim \text{Log-Normal}(0, 1) & [\beta \text{ prior}] \\
    \sigma &\sim \text{Uniform}(0, 50) & [\sigma \text{ prior}]
    \end{align*}

    The log-normal prior for beta ensures that the slopes are positive and within a reasonable range.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Using pm.Deterministic
    """)
    return


@app.cell
def _(MEAN_W, az, data, np, plt, pm):
    _x = np.linspace(
        data.select("weight").min().item(), data.select("weight").max().item(), 100
    )

    with pm.Model() as model:
        # Priors
        _alpha = pm.Normal("alpha", mu=178, sigma=20)
        # _beta = pm.Uniform("beta", 0, 1)
        _beta = pm.LogNormal("beta", mu=0, sigma=1)
        _sigma = pm.Uniform("sigma", lower=0, upper=50)
    
        # Linear model
        # We use a Deterministic to track the values of mu
        _mu = pm.Deterministic("mu", _alpha + _beta * (_x - MEAN_W))
    
        # Likelihood
        _h = pm.Normal("h", mu=_mu, sigma=_sigma, shape=_x.shape)
    
        # Sample prior predictive
        _idata = pm.sample_prior_predictive(samples=100)

    # Plotting
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 5))

    # Extract prior samples for mu
    # Shape is usually (chains, draws, dim). We stack chains and draws.
    _prior_mu = _idata.prior["mu"].stack(sample=("chain", "draw")).values 
    _ax[0].plot(_x, _prior_mu, c="k", alpha=0.4)
    _ax[0].set_title("Prior Regression Lines")

    # Extract prior predictive samples for h
    _prior_h = _idata["prior"]["h"].stack(sample=("chain", "draw")).values.flatten()
    az.plot_kde(_prior_h, ax=_ax[1])
    _ax[1].set_title("Prior Predictive Distribution of h")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plotting priors by hand
    """)
    return


@app.cell
def _(MEAN_W, az, data, np, plt, pm):
    _x = np.linspace(
        data.select("weight").min().item(), data.select("weight").max().item(), 100
    )

    with pm.Model() as _model:
        # Priors
        _alpha = pm.Normal("alpha", mu=178, sigma=20)
        _beta = pm.LogNormal("beta", mu=0, sigma=1)
        _sigma = pm.Uniform("sigma", lower=0, upper=50)
    
        # Linear model
        # If not using pm.Deterministic, the values of mu are not saved in the inference_data obj at the end.
        # So it needs to be calculated manually. Saves a lot of memory if simulating a lot of samples.
        _mu = _alpha + _beta * (_x - MEAN_W)
    
        # Likelihood
        _h = pm.Normal("h", mu=_mu, sigma=_sigma, shape=_x.shape)
    
        # Sample prior predictive
        _idata = pm.sample_prior_predictive(samples=200)

    # 1. Extract the samples (flatten chains and draws)
    # These will be 1D arrays of length (chains * draws)
    _alpha_plot = _idata["prior"]["alpha"].to_numpy().flatten()
    _beta_plot = _idata["prior"]["beta"].to_numpy().flatten()

    # 2. Manually calculate mu
    # We use numpy broadcasting: 
    # (N_samples, 1) + (N_samples, 1) * (N_x_points,)
    _mu_plot = _alpha_plot[:, None] + _beta_plot[:, None] * (_x - MEAN_W)

    # Plotting
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 5))

    # The "Golden Rule" of plt.plot(x, y). When y is a 2D array, plt interpret every column as a curve to plot.
    # In this case, every column is a line
    _ax[0].plot(_x, _mu_plot.T, c="k", alpha=0.4)
    _ax[0].set_title("Prior Regression Lines")

    # Extract prior predictive samples for h
    _prior_h = _idata["prior"]["h"].stack(sample=("chain", "draw")).values.flatten()
    az.plot_kde(_prior_h, ax=_ax[1])
    _ax[1].set_title("Prior Predictive Distribution of h")

    return


@app.cell
def _():
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md("""
    ### Model 4_3 - Linear Prediction
    """)
    return


@app.cell
def _(MEAN_W, az, data, pm):
    with pm.Model() as m4_3:
        _alpha = pm.Normal("alpha", mu=178, sigma=20)
        _beta = pm.Lognormal("beta", mu=0, sigma=1)
        _sigma = pm.Uniform("sigma", 0, 50)
        _mu = _alpha + _beta * (data.get_column("weight").to_numpy() - MEAN_W)
        # CANNOT USE data.select('height') AS IT RETURNS A DATAFRAME. NEEDS TO BE A SERIES. RESULTS ARE COMPLETELY DIFFERENT
        # try type(data.select("height")), type(data.get_column("height")) = type(data["height"])
        _height = pm.Normal(
            "height", mu=_mu, sigma=_sigma, observed=data.get_column("height")
        )
        idata_4_3 = pm.sample(1000, tune=1000)

    data_4_3 = az.extract(idata_4_3)
    return data_4_3, idata_4_3, m4_3


@app.cell
def _(az, idata_4_3):
    data_4_3_df = az.extract(idata_4_3).to_dataframe()

    (
        az.summary(idata_4_3, kind="stats"),
        data_4_3_df,
        data_4_3_df.cov().round(3),
        data_4_3_df.corr().round(3),
    )
    return


@app.cell
def _(MEAN_W, data, idata_4_3, plt):
    # we can plot the line using the mean of the posterior (this would just be one of all the possible lines)
    plt.plot(data["weight"], data["height"], ".")

    plt.plot(
        data["weight"],
        idata_4_3.posterior["alpha"].mean().item()
        + idata_4_3.posterior["beta"].mean().item() * (data["weight"] - MEAN_W),
    )

    plt.xlabel("Weight")
    plt.ylabel("Height")
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    So now let’s display a bunch of these lines, so you can see the scatter. This lesson will be
    easier to appreciate, if we use only some of the data to begin. Then you can see how adding
    in more data changes the scatter of the lines. So we’ll begin with just the first 10 cases in data.
    The following code extracts the first 10 cases and re-estimates the model:
    """)
    return


@app.cell
def _(data, mo):
    n_slider = mo.ui.slider(10, data.shape[0], 10, show_value=True)
    n_slider
    return (n_slider,)


@app.cell
def _(az, data, n_slider, plt, pm):
    _n = n_slider.value
    _data_n = data.head(_n)
    mean_n = _data_n.select("weight").mean().item()

    with pm.Model() as m_N:
        _alpha = pm.Normal("alpha", mu=178, sigma=20)
        _beta = pm.Lognormal("beta", mu=0, sigma=1)
        _sigma = pm.Uniform("sigma", 0, 50)
        _mu = _alpha + _beta * (_data_n.get_column("weight").to_numpy() - mean_n)
        # CANNOT USE data.select('height') AS IT RETURNS A DATAFRAME. NEEDS TO BE A SERIES. RESULTS ARE COMPLETELY DIFFERENT
        # try type(data.select("height")), type(data.get_column("height")) = type(data["height"])
        _height = pm.Normal(
            "height", mu=_mu, sigma=_sigma, observed=_data_n.get_column("height")
        )
        idata_N = pm.sample(1000, tune=1000)

    data_N = az.extract(idata_N)

    # we can plot the line using the mean of the posterior (this would just be one of all the possible lines)
    plt.plot(_data_n["weight"], _data_n["height"], ".")

    _n_lines = 20
    for _i in range(_n_lines):
        plt.plot(
            _data_n["weight"],
            data_N["alpha"].item(_i)
            + data_N["beta"].item(_i) * (_data_n["weight"] - mean_n),
            alpha=0.2,
            color="orange",
        )

    plt.title(f"{_n_lines} lines for {_n} Data Points")
    plt.xlabel("Weight")
    plt.ylabel("Height")
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The value of $x_i$ in this case is `centre_at`. `mu_at_centre` is a vector of predicted means, one for each random sample from the posterior. Since joint `alpha` and `beta` went into computing each, the variation across those means incorporates the uncertainty in and correlation between both parameters. It might be helpful at this point to actually plot the density for this vector of means:
    """)
    return


@app.cell
def _(MEAN_W, az, data_4_3):
    centre_at = 50
    mu_at_centre = data_4_3["alpha"] + data_4_3["beta"] * (centre_at - MEAN_W)
    az.plot_kde(mu_at_centre.values), az.hdi(mu_at_centre.values, hdi_prob=0.89)
    return


@app.cell
def _(MEAN_W, data_4_3, np, plt):
    weight_seq_4_3 = np.arange(25, 71)
    # weight_seq_4_3 = data['weight'].sort()
    _n_samples = data_4_3.sizes["sample"]

    # only select every 10 sample from the posterior. Goes from 4000 to 400
    # can delete 10 and run for all data if wanted
    # data_4_3_thinned = data_4_3.isel(sample=range(0, _n_samples, 10))
    data_4_3_thinned = data_4_3.isel(sample=range(0, _n_samples))
    _n_samples_thinend = data_4_3_thinned.sizes["sample"]

    mu_pred_4_3 = np.zeros((len(weight_seq_4_3), _n_samples_thinend))
    for _i, w in enumerate(weight_seq_4_3):
        # mu_pred_4_3[_i] selects row _i from the mu_pred_4_3 array. So every row in this array is a distribution of heights for each weight in weight_seq_4_3
        mu_pred_4_3[_i] = data_4_3_thinned["alpha"] + data_4_3_thinned["beta"] * (
            w - MEAN_W
        )

    # to calculate the mean height for each weight, we can just use mu_pred_4_3.mean(axis=1). This would average each row of the mu_pred_4_3 array.

    # For every value in weight_seq_4_3, plots the corresponding row from mu_pred_4_3. In other words, for every weight, plot the correponding distribution of mu's.
    plt.plot(weight_seq_4_3, mu_pred_4_3, "C0.", alpha=0.1)
    plt.xlabel("weight")
    plt.ylabel("height")
    plt.gca()
    return mu_pred_4_3, weight_seq_4_3


@app.cell
def _(az, data, mu_pred_4_3, plt, weight_seq_4_3):
    # this calculates the highest density interval for each distribution of height for each weight.
    # mu_hdi_4_3 = az.hdi(mu_pred_4_3.T)

    az.plot_hdi(weight_seq_4_3, mu_pred_4_3.T)
    plt.scatter(data["weight"], data["height"])
    plt.plot(weight_seq_4_3, mu_pred_4_3.mean(axis=1), color="green", lw=3)
    plt.xlabel("weight")
    plt.ylabel("height")
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The above only shows the uncertainty for the mean of the posterior. We want to know how the whole posterior distribution behaves. It is given by

    \begin{align*}
    \mu_i &= \alpha + \beta w_i \quad  (\beta(x_i - \bar{x})) & [\text{linear model}]
    \end{align*}

    For every unique weight value, we sample from a Gaussian with the correct mean $\mu$ for that weight, using the correct value from $\sigma$ sampled from the same posterior. If we do this for every sample from the posterior, for every weight value of interest, we end up with a collection of simulated heights that embodies the uncertainty in the posterior as well as the uncertainty in the Gaussian distribution of heights.
    """)
    return


@app.cell
def _(az, data, idata_4_3, m4_3, mu_pred_4_3, plt, pm, weight_seq_4_3):
    with m4_3:
        # Generate predicted heights based on the posterior distribution
        height_pred_4_3 = pm.sample_posterior_predictive(idata_4_3)

    # calculate the hdi interval for each weight sample
    height_pred_4_3_hdi = az.hdi(height_pred_4_3["posterior_predictive"], hdi_prob=0.89)

    # plots the hdi for the mean and the line for the mean
    az.plot_hdi(weight_seq_4_3, mu_pred_4_3.T, hdi_prob=0.89, color="green")
    plt.plot(weight_seq_4_3, mu_pred_4_3.mean(axis=1), lw=3, color="green")
    # plots the hdi for the whole posterior distribution
    az.plot_hdi(
        data["weight"], height_pred_4_3["posterior_predictive"]["height"], hdi_prob=0.89
    )
    # plots the whole data
    plt.scatter(data["weight"], data["height"])

    plt.xlim(data["weight"].min(), data["weight"].max())

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The above can be done manually:
    """)
    return


@app.cell
def _(MEAN_W, az, data, data_4_3, mu_pred_4_3, np, plt, stats, weight_seq_4_3):
    post_samples_4_3 = []

    for _ in range(1000):  # number of samples from the posterior
        _i = np.random.randint(0, len(weight_seq_4_3))
        _mu_pred = data_4_3["alpha"][_i].item(0) + data_4_3["beta"][_i].item(0) * (
            weight_seq_4_3 - MEAN_W
        )
        _sigma_pred = data_4_3["sigma"][_i]
        post_samples_4_3.append(stats.norm.rvs(loc=_mu_pred, scale=_sigma_pred))

    # plots the hdi for the mean and the line for the mean
    az.plot_hdi(weight_seq_4_3, mu_pred_4_3.T, hdi_prob=0.89, color="green")
    plt.plot(weight_seq_4_3, mu_pred_4_3.mean(axis=1), lw=3, color="green")
    # plots the hdi for the whole posterior distribution
    az.plot_hdi(weight_seq_4_3, np.array(post_samples_4_3), hdi_prob=0.89)
    # plots the whole data
    plt.scatter(data["weight"], data["height"])

    plt.xlim(data["weight"].min(), data["weight"].max())

    plt.gca()
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
