import marimo

__generated_with = "0.17.8"
app = marimo.App(width="columns")


@app.cell(column=0)
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
    return az, mo, pl, pm, stats


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

    data_adults, data_kids
    return MEAN_W_K, data_kids


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

    <span style="color:green;font-weight:bold">Answer:</span>

    For every 10 units increase in weight, the model expects an average 27.1 cm increase in height
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

    az.summary(idata_kids_4h2, hdi_prob=0.89, kind="stats")
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
