# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.0.0",
#     "arviz==0.22.0",
#     "matplotlib==3.10.7",
#     "numpy==2.3.5",
#     "polars==1.35.2",
#     "pymc==5.26.1",
#     "scipy==1.16.3",
# ]
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App(width="columns")

with app.setup:
    # Initialization code that runs before all other cells
    pass


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
    ### Helper Functions
    """)
    return


@app.cell
def _(pl):
    raw_data = pl.read_csv(
        "/home/marcelo/git/rethinking_stats_and_PyMC/data/Howell1.csv", separator=";"
    )

    def filter_data_adults(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("age") >= 18)

    def filter_data_kids(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("age") < 18)

    data_adults = raw_data.pipe(filter_data_adults)
    data_kids = raw_data.pipe(filter_data_kids)

    MEAN_W_A = data_adults.select("weight").mean().item()
    MEAN_W_K = data_kids.select("weight").mean().item()
    MEAN_RAW = raw_data.select("weight").mean().item()

    data_adults, data_kids
    return MEAN_W_K, data_kids, raw_data


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ### Easy
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4E1. In the model definition below, which line is the likelihood?

    \begin{align*}
    y_i &\sim \text{Normal}(\mu, \sigma)\\
    \mu &\sim \text{Normal}(0, 10)\\
    \sigma &\sim \text{Exponential}(1)
    \end{align*}

    <span style="color:green;font-weight:bold">Answer:</span>

    \begin{align*}
    y_i &\sim \text{Normal}(\mu, \sigma) \end{align*}

    $\mu \sim \dots$ is the prior for the mean.

    $\sigma \sim \dots$ is the prior for the standard deviation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    4E2. In the model definition just above, how many parameters are in the posterior distribution?

    <span style="color:green;font-weight:bold">Answer:</span>

    2. The mean and the standard deviation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4E3. Using the model definition above, write down the appropriate form of Bayes’ theorem that includes the proper likelihood and priors.

    <span style="color:green;font-weight:bold">Answer:</span>

    \begin{align*}
    \text{Pr}(\mu, \sigma | y) = \frac
    {\text{Normal}(y|\mu, \sigma)	\text{Normal}(\mu|0, 10) \text{Exponential}(\sigma|1)}
    {\int \int \text{Normal}(y|\mu, \sigma)	\text{Normal}(\mu|0, 10) \text{Exponential}(\sigma|1) d\mu d\sigma}
    \end{align*}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4E4. In the model definition below, which line is the linear model?

    \begin{align*}
    y_i &\sim \mathrm{Normal}(\mu_i, \sigma)\\
    \mu_i &= \alpha + \beta x_i\\
    \alpha &\sim \mathrm{Normal}(0, 10)\\
    \beta &\sim \mathrm{Normal}(0, 1)\\
    \sigma &\sim \mathrm{Exponential}(2)
    \end{align*}

    <span style="color:green;font-weight:bold">Answer:</span>

    The linear model is defined by the equation:
    $\mu_i = \alpha + \beta x_i$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4E5. In the model definition just above, how many parameters are in the posterior distribution?

    <span style="color:green;font-weight:bold">Answer:</span>

    3. The parameters are $\alpha$, $\beta$, and $\sigma$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Medium
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4M1. For the model definition below, simulate observed y values from the prior (not the posterior).

    \begin{align*}
    y_i &\sim \mathrm{Normal}(\mu, \sigma)\\
    \mu &\sim \mathrm{Normal}(0, 10)\\
    \sigma &\sim \mathrm{Exponential}(1)
    \end{align*}
    """)
    return


@app.cell
def _(az, stats):
    _n = 1000
    _mu = stats.norm.rvs(loc=0, scale=10, size=_n)
    _sigma = stats.expon.rvs(scale=1, size=_n)
    _prior_h = stats.norm.rvs(loc=_mu, scale=_sigma, size=_n)

    az.plot_kde(_prior_h)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4M2. Translate the model just above into a quap formula (PyMC in this case).
    """)
    return


@app.cell
def _(az, pm):
    with pm.Model() as model_4M_2:
        _mu = pm.Normal("mu", mu=0, sigma=10)
        _sigma = pm.Exponential("sigma", lam=1)
        _h = pm.Normal("h", mu=_mu, sigma=_sigma)

        prior_4M_2 = pm.sample_prior_predictive(samples=1000)

    az.plot_kde(prior_4M_2["prior"]["h"].values, label="4M_2")
    return


@app.cell
def _():
    # az.plot_kde(stats.norm.rvs(loc=0, scale=20, size=1000), label="Simple")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4M3. Translate the quap model formula below (book pdf page 140) into a mathematical model definition.

    \begin{align*}
    y_i &\sim \mathrm{Normal}(\mu_i, \sigma)\\
    \mu_i &= \alpha + \beta x_i\\
    \alpha &\sim \mathrm{Normal}(0,\,10)\\
    \beta &\sim \mathrm{Uniform}(0,\,1)\\
    \sigma &\sim \mathrm{Exponential}(1)
    \end{align*}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4M4. A sample of students is measured for height each year for 3 years. After the third year, you want to fit a linear regression predicting height using year as a predictor. Write down the mathematical model definition for this regression, using any variable names and priors you choose. Be prepared to defend your choice of priors.

    <span style="color:green;font-weight:bold">Answer:</span>

    \begin{align*}
    h_i &\sim \mathrm{Normal}(\mu_i, \sigma)\\
    \mu_i &= \alpha + \beta\ \times (y - \bar{y})\\
    \alpha &\sim \mathrm{Normal}(150,\,30)\\
    \beta &\sim \mathrm{Uniform}(0,\,40)\\
    \sigma &\sim \mathrm{Normal}(0,\,30)
    \end{align*}

    The answer assume measurements are in $cm$.

    Since very little information is given about the students, non-informative priors will be needed.

    The model is centered so $\alpha$ is the average height. The standard deviation is large to account for the uncertainty.

    $\beta$ represents the average growth per year for each student. Without more information about the students, I will assume that a range between $0 cm$ and $40 cm$ accounts for the uncertainty.

    $\sigma$ represents how spread out the heihgts are. Again, without any information about the students, it must carry little information. That's why the normal with the large std.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4M5. Now suppose I remind you that every student got taller each year. Does this information lead you to change your choice of priors? How?

    <span style="color:green;font-weight:bold">Answer:</span>

    No, I have alredy taken that into account by forcing $\beta$ to be positive.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4M6. Now suppose I tell you that the variance among heights for students of the same age is never more than 64cm. How does this lead you to revise your priors?

    <span style="color:green;font-weight:bold">Answer:</span>

    Yes, to ensure that the variance never goes above $64$ cm, we can use a Uniform prior from $0$ to $8$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hard
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4H1. The weights listed below were recorded in the !Kung census, but heights were not recorded for these individuals. Provide predicted heights and 89% intervals for each of these individuals. That is, fill in the table below, using model-based predictions.

    <span style="color:green;font-weight:bold">Answer:</span>
    """)
    return


@app.cell
def _():
    # _weights = np.array([46.95, 43.72, 64.78, 32.59, 54.63])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4H2. Select out all the rows in the Howell1 data with ages below 18 years of age. If you do it right, you should end up with a new data frame with 192 rows in it.

    (a) Fit a linear regression to these data, using quap. Present and interpret the estimates. For every 10 units of increase in weight, how much taller does the model predict a child gets?

    (b) Plot the raw data, with height on the vertical axis and weight on the horizontal axis. Superimpose the MAP regression line and 89% interval for the mean. Also superimpose the 89% interval for predicted heights.

    (c) What aspects of the model fit concern you? Describe the kinds of assumptions you would change, if any, to improve the model. You don’t have to write any new code. Just explain what the model appears to be doing a bad job of, and what you hypothesize would be a better model.

    <span style="color:green;font-weight:bold">Answer:</span>

    (a) For every 10 units increase in weight, the model expects an average 27.2 cm increase in height (see below)

    (b) See below

    (c) The relationship is clearly not linear. The model overestimates for low and high weights and underestimates for middle weights. Getting away from the linear contraint and looking at other types of relationships is needed. The data looks like the log graph, so maybe a log-transform might help the linear equation.
    """)
    return


@app.cell
def _(MEAN_W_K, az, data_kids, pm):
    with pm.Model() as linear_kids:
        _alpha = pm.Normal("alpha", mu=110, sigma=30)
        _beta = pm.Uniform("beta", lower=0, upper=5)
        # _beta = pm.LogNormal("beta", mu=0, sigma=1)
        _sigma = pm.Uniform("sigma", lower=0, upper=50)

        _mu = _alpha + _beta * (data_kids.get_column("weight").to_numpy() - MEAN_W_K)

        _h = pm.Normal("height", mu=_mu, sigma=_sigma, observed=data_kids.get_column("height"))

        idata_kids_4h2 = pm.sample(1_000)

        pred_kids = pm.sample_posterior_predictive(idata_kids_4h2)

    az.summary(idata_kids_4h2, hdi_prob=0.89, kind="stats")
    return idata_kids_4h2, pred_kids


@app.cell
def _(az, idata_kids_4h2):
    az.extract(idata_kids_4h2).to_dataframe(), idata_kids_4h2["posterior"]["alpha"].to_numpy().flatten()[:10]
    return


@app.cell
def _(MEAN_W_K, az, data_kids, idata_kids_4h2, np, plt, pred_kids):
    _alpha_samples = idata_kids_4h2["posterior"]["alpha"].to_numpy().flatten() # shape is (n_samples,)
    _beta_samples = idata_kids_4h2["posterior"]["beta"].to_numpy().flatten() # shape is (n_samples,)
    _weight_kids = data_kids.get_column("weight").to_numpy() # shape is (lenght_data,)

    # Raw data
    plt.scatter(data_kids["weight"], data_kids["height"], c="red", label="raw data")

    # MAP
    plt.plot(
        data_kids["weight"],
        _alpha_samples.mean() + _beta_samples.mean() * (_weight_kids - MEAN_W_K),
        c="green",
        lw=3,
        label="MAP regression line"
    )

    # 2. Manually calculate mu for each posterior sample
    # Broadcasting: (n_samples, 1) + (n_samples, 1) * (n_points,)
    # Result: (n_samples, n_points)
    _mu_kids = _alpha_samples[:, np.newaxis] + _beta_samples[:, np.newaxis] * (_weight_kids - MEAN_W_K) # shape is (n_samples, lenght_data)

    # 89% HDI mean
    _mu_hdi = az.hdi(_mu_kids, hdi_prob=0.89)
    # Both below plot the same thing
    # az.plot_hdi(x=data_kids["weight"], y=_mu_kids, hdi_prob=0.89, color="green")
    az.plot_hdi(x=data_kids["weight"], hdi_data=_mu_hdi, color="green")

    # 89% HDI predictions
    az.plot_hdi(data_kids["weight"], pred_kids["posterior_predictive"]["height"], hdi_prob=0.89)

    plt.xlabel("Weight (kg)")
    plt.ylabel("Height (cm)")
    plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4H3. Suppose a colleague of yours, who works on allometry, glances at the practice problems just above. Your colleague exclaims, “That’s silly. Everyone knows that it’s only the logarithm of body weight that scales with height!” Let’s take your colleague’s advice and see what happens.

    (a) Model the relationship between height (cm) and the natural logarithm of weight (log-kg). Use the entire Howell1 data frame, all 544 rows, adults and non-adults. Fit this model, using quadratic approximation:

    \begin{align*}
    h_i &\sim \mathrm{Normal}(\mu_i, \sigma)\\
    \mu_i &= \alpha + \beta \times \log(w_i)\\
    \alpha &\sim \mathrm{Normal}(178, 20)\\
    \beta &\sim \mathrm{Log-Normal}(0, 1)\\
    \sigma &\sim \mathrm{Uniform}(0, 50)
    \end{align*}

    where $h_i$ is the height of individual $i$ and $w_i$ is the weight (in kg) of individual $i$. Can you interpret the resulting estimates?

    (b) Then use samples from the quadratic approximate posterior of the model in (a) to superimpose on the plot: (1) the predicted mean height as a function of weight, (2) the 97% interval for the mean, and (3) the 97% interval for predicted heights.

    <span style="color:green;font-weight:bold">Answer:</span>

    (a) $\alpha=-22.85$ tells us that when the $\ln(w)=0 \rightarrow w=1 kg$ the average height is -22.85. This is non-informative and doesn't make sense.

    $\beta=46.81$ tells us that every unit increase in the $\ln(w)$ gives us an increase in the mean height of 46.81. So for every 1 ln-kg we expect someone to be, on average, 46.81 cm taller.

    (b)
    """)
    return


@app.cell
def _(az, pm, raw_data):
    with pm.Model() as log_model:
        _alpha = pm.Normal("alpha", 178, 20)
        _beta = pm.LogNormal("beta", 0, 1)
        _sigma = pm.Uniform("sigma", 0, 50)
        _mu = _alpha + _beta * raw_data.get_column("weight").log().to_numpy()
        _h = pm.Normal("height", mu=_mu, sigma=_sigma, observed=raw_data.get_column("height"))

        idata_raw = pm.sample(1_000)

        pred_raw = pm.sample_posterior_predictive(idata_raw)

    az.summary(idata_raw)
    return idata_raw, pred_raw


@app.cell
def _(az, idata_raw, np, plt, pred_raw, raw_data):
    _alpha_samples = idata_raw["posterior"]["alpha"].to_numpy().flatten()
    _beta_samples = idata_raw["posterior"]["beta"].to_numpy().flatten()

    _, _axs = plt.subplots(1, 2, figsize=(12, 4))

    _axs[0].scatter(raw_data["weight"], raw_data["height"], c="red", label="raw_data")

    # MAP line
    # !!! Sort the values so that the plot makes sense !!!
    _sort_idxs = raw_data["weight"].arg_sort()
    _axs[0].plot(
        raw_data["weight"][_sort_idxs],
        _alpha_samples.mean() + _beta_samples.mean() * raw_data["weight"].log()[_sort_idxs],
        c="green",
        lw=3,
        label="MAP regression line"
    )

    _axs[1].scatter(raw_data["weight"].log(), raw_data["height"], c="blue", label="log_data")

    # MAP line
    _axs[1].plot(
        raw_data["weight"].log(),
        _alpha_samples.mean() + _beta_samples.mean() * raw_data["weight"].log(),
        c="green",
        lw=3,
        label="MAP regression line"
    )

    # MAP 97% HDI
    _mu = _alpha_samples[:, np.newaxis] + _beta_samples[:, np.newaxis] * raw_data["weight"].log().to_numpy()
    _mu_hdi = az.hdi(_mu, hdi_prob=0.97)
    az.plot_hdi(ax=_axs[1], x=raw_data["weight"].log(), hdi_data=_mu_hdi, color="green")

    # 97% HDI predictions
    az.plot_hdi(x=raw_data["weight"].log(), y=pred_raw["posterior_predictive"]["height"], hdi_prob=0.97, ax=_axs[1])

    plt.legend()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
