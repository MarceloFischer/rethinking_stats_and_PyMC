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

__generated_with = "0.17.8"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    ### Imports
    """)
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import scipy as sp
    from scipy.stats import binom, beta
    import arviz as az
    import pymc as pm
    import altair as alt
    import matplotlib.pyplot as plt
    return alt, az, binom, mo, np, pl, plt


@app.cell
def _(np):
    np.random.seed(42)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Fitting
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Here's the goal:

    Toss a globe and check if the index finger landed on land or water. Our goal is to estimate the proportion of water in the world given the data we collect. Observed data is W L W W W L W L W

    - p -> the proportion of water in the world (parameter)
    - N -> the number of tosses
    - W -> the number of times the index finger landed on water
    - L -> the number of times the index finger landed on land
    - $N=W+L$

    ### <span style="color:green;">Assumptions</span>

    - Every toss is independent
    - The probability of W in every toss is the same
        - It follows a binomial distribution
    $$P(W, L \mid p) = \frac{(W + L)!}{W! \, L!} \; p^W (1 - p)^L$$
    """)
    return


@app.cell
def _(binom):
    # get the likelihood of the data for several different values of p
    binom.pmf(k=6, n=9, p=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The above means that
    $$W \sim \mathrm{Binomial}(N,\, p), \text{where } N=W+L$$
    and
    $$p \sim \mathrm{Uniform}(0, 1)$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$Posterior = \frac{Probability\_of\_the\_data \times Prior}{Marginal\_probability}$$

    $$Pr(p\mid W,L) = \frac{Pr(W,L \mid p) \times Pr(p)}{Pr(W,L)}$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Grid Approximation

    Awful for scalability, but a good starting point to get the theory behind the models
    """)
    return


@app.cell
def _(alt, binom, np, pl):
    def grid_approx_ch2(p_grid_size: int = 51, prior: str = "uniform", success: int = 6, tosses: int = 9):
        # define a grid. Possible values for p
        p_grid = np.linspace(0, 1, p_grid_size)
        # define a prior.
        priors = {
            "uniform": np.ones_like(p_grid),
            "step": np.where(p_grid < 0.5, 0, 1),
            "exponential": np.exp(-5 * np.abs(p_grid - 0.5)),
        }
        prior_values = priors[prior]
        # compute the likelihood of the data for each value of p and the posterior
        likelihood = binom.pmf(k=success, n=tosses, p=p_grid)
        unstd_posterior = likelihood * prior_values
        posterior = unstd_posterior / np.sum(unstd_posterior)

        # Create a Polars DataFrame with simplified column names
        df = pl.DataFrame(
            {
                "prob_water": p_grid,
                "prior": prior_values,
                "likelihood": likelihood,
                "unstd_posterior": unstd_posterior,
                "posterior_prob": posterior,
            }
        )

        # Create the Altair plot using the correct column names
        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X("prob_water:Q", title="Probability of Water (p)"),
                y=alt.Y("posterior_prob:Q", title="Posterior Probability"),
                tooltip=["prob_water", "posterior_prob", "prior", "likelihood"],
            )
            .properties(title=f"Posterior with {len(p_grid)} Points for {tosses} Trials")
            .interactive()
        )
        return df, chart
    return (grid_approx_ch2,)


@app.cell
def _(post_df):
    post_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Quadratic Approximation

    This approximation essentially represents any log-posterior with a parabola. This is because under quite general conditions, the region near the peak of the posterior distribution will be nearly Gaussian in shape.
    """)
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chapter 2
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 1 - Medium
    """)
    return


@app.cell
def _(grid_approx_ch2, mo):
    post_df, globe_tosses_chart = grid_approx_ch2(5, success=2, tosses=3)

    mo.ui.altair_chart(globe_tosses_chart)
    return (post_df,)


@app.cell
def _(grid_approx_ch2, mo):
    # W, W, W, L
    ex_m1_2_df, chart_ex_m1_2 = grid_approx_ch2(p_grid_size=100, success=3, tosses=4)

    mo.ui.altair_chart(chart_ex_m1_2)
    return


@app.cell
def _(grid_approx_ch2, mo):
    # L, W, W, L, W, W, W
    ex_m1_3_df, chart_ex_m1_3 = grid_approx_ch2(p_grid_size=100, success=5, tosses=7)

    mo.ui.altair_chart(chart_ex_m1_3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 2 - Medium
    """)
    return


@app.cell
def _(grid_approx_ch2, mo):
    # W, W, W
    ex_m2_df, chart_ex_m2 = grid_approx_ch2(p_grid_size=100, prior="step", success=3, tosses=3)

    mo.ui.altair_chart(chart_ex_m2)
    return


@app.cell
def _(grid_approx_ch2, mo):
    # W, W, W, L
    ex_m2_2_df, chart_ex_m2_2 = grid_approx_ch2(p_grid_size=100, prior="step", success=3, tosses=4)

    mo.ui.altair_chart(chart_ex_m2_2)
    return


@app.cell
def _(grid_approx_ch2, mo):
    # L, W, W, L, W, W, W
    ex_m2_3_df, chart_ex_m2_3 = grid_approx_ch2(p_grid_size=100, prior="step", success=5, tosses=7)

    mo.ui.altair_chart(chart_ex_m2_3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 3 - Medium
    """)
    return


@app.cell
def _():
    p_w_earth = 0.7
    p_l_earth = 1 - p_w_earth
    p_w_mars = 0
    p_l_mars = 1 - p_w_mars
    p_earth, p_mars = 0.5, 0.5

    prob_earth_given_land = (p_earth * p_l_earth) / (p_earth * p_l_earth + p_mars * p_l_mars)
    prob_earth_given_land
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 4 - Medium
    """)
    return


@app.cell
def _(np):
    card_4 = np.array(["BB", "BW", "WW"])
    ways_4 = np.array([2, 1, 0])
    p_4 = ways_4 / ways_4.sum()
    # Sum probabilities for BB cards
    bb_probability_4 = p_4[card_4 == "BB"].sum()
    bb_probability_4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 5 - Medium
    """)
    return


@app.cell
def _(np):
    card_5 = np.array(["BB", "BW", "WW", "BB"])
    ways_5 = np.array([2, 1, 0, 2])
    p_5 = ways_5 / ways_5.sum()
    # Sum probabilities for BB cards
    bb_probability_5 = p_5[card_5 == "BB"].sum()
    bb_probability_5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 6 - Medium
    """)
    return


@app.cell
def _(np, pl):
    card_6 = np.array(["BB", "BW", "WW"])
    ways_6 = np.array([2, 1, 0])
    prior_6 = np.array([1, 2, 3])  # what I believe is true before any data
    likelihood_6 = ways_6 * prior_6
    posterior = likelihood_6 / likelihood_6.sum()
    # Sum probabilities for BB cards
    bb_probability_6 = posterior[card_6 == "BB"].sum()

    ex_6_df = pl.DataFrame({"card": card_6, "ways": ways_6, "prior": prior_6, "likelihood": likelihood_6, "prob_bb": posterior})
    ex_6_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 7 - Medium
    """)
    return


@app.cell
def _(np):
    card_7 = np.array(["BB", "BW", "WW"])
    ways_7 = np.array([2, 1, 0])
    p_7 = ways_7 / ways_7.sum()
    # Sum probabilities for BB cards
    bb_probability_7 = p_7[card_7 == "BB"].sum()
    bb_probability_7
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chapter 3
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise - Easy
    """)
    return


@app.cell
def _(grid_approx_ch2, np, pl):
    ch3_post_df, ch3_post_chart = grid_approx_ch2(1_001, "uniform", success=6, tosses=9)

    # samples used for a lot of the problems
    samples_easy = np.random.choice(
        ch3_post_df.select(pl.col("prob_water")).to_numpy().ravel(),
        10_000,
        replace=True,
        p=ch3_post_df.select(pl.col("posterior_prob")).to_numpy().ravel(),
    )

    # ch3_post_chart
    return (samples_easy,)


@app.cell
def _(np, samples_easy):
    # 3E1. How much posterior probability lies below p = 0.2?
    np.sum(samples_easy < 0.2) / samples_easy.size
    return


@app.cell
def _(np, samples_easy):
    # 3E2. How much posterior probability lies above p = 0.8?
    np.sum(samples_easy > 0.8) / samples_easy.size
    return


@app.cell
def _(np, samples_easy):
    # 3E3. How much posterior probability lies between p = 0.2 and p = 0.8?
    np.sum((0.2 < samples_easy) & (samples_easy < 0.8)) / samples_easy.size
    return


@app.cell
def _(np, samples_easy):
    # 3E4. 20% of the posterior probability lies below which value of p?
    np.quantile(samples_easy, 0.2)
    return


@app.cell
def _(np, samples_easy):
    # 3E5. 20% of the posterior probability lies above which value of p?
    np.quantile(samples_easy, 0.8)
    return


@app.cell
def _(az, samples_easy):
    # 3E6. Which values of p contain the narrowest interval equal to 66% of the posterior probability?

    # The question is asking for the highest density posterior interval (HDPI or HDI)
    az.hdi(samples_easy, hdi_prob=0.66)
    return


@app.cell
def _(np, samples_easy):
    # 3E7. Which values of p contain 66% of the posterior probability, assuming equal posterior probability both below and above the interval?

    middle_pct = 0.66
    pct_tail = (1 - middle_pct) / 2

    np.quantile(samples_easy, (pct_tail, 1 - pct_tail))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise - Medium
    """)
    return


@app.cell
def _(grid_approx_ch2, np, pl):
    # 3M1. Suppose the globe tossing data had turned out to be 8 water in 15 tosses. Construct the posterior distribution, using grid approximation. Use the same flat prior as before.


    ch3_m1_to_4_df, chart_ch3_m1_to_4 = grid_approx_ch2(p_grid_size=1_001, prior="uniform", success=8, tosses=15)

    # samples used for a lot of the problems
    samples_medium = np.random.choice(
        ch3_m1_to_4_df.select(pl.col("prob_water")).to_numpy().ravel(),
        10_000,
        replace=True,
        p=ch3_m1_to_4_df.select(pl.col("posterior_prob")).to_numpy().ravel(),
    )

    chart_ch3_m1_to_4
    return (samples_medium,)


@app.cell
def _(az, samples_medium):
    # 3M2. Draw 10,000 samples from the grid approximation from above. Then use the samples to calculate the 90% HPDI for p.
    az.hdi(ary=samples_medium, hdi_prob=0.9)
    return


@app.cell
def _(binom, np, plt, samples_medium):
    # 3M3. Construct a posterior predictive check for this model and data. This means simulate the distribution of samples, averaging over the posterior uncertainty in p. What is the probability of observing 8 water in 15 tosses?

    # posterior predictive distribution
    ppd = np.random.binomial(n=15, p=samples_medium)

    plt.hist(ppd, bins=15, label="Bayesian with step prior")

    # we know that around 70% of Earth is water, so the below is the true distribution
    plt.plot(
        np.linspace(0, 15, 16),
        samples_medium.size * binom.pmf(np.linspace(0, 15, 16), n=15, p=0.7),
        label="True underlying model",
    )
    plt.title("Posterior predictive distribution")
    plt.legend(loc=(1.05, 0.9))
    plt.show()
    return (ppd,)


@app.cell
def _(ppd):
    # 3M4. Using the posterior distribution constructed from the new (8/15) data, now calculate the probability of observing 6 water in 15 tosses.
    (ppd == 6).mean()
    return


@app.cell
def _():
    # 3M6. Suppose you want to estimate the Earth's proportion of water very precisely. Specifically, you want the 99% percentile interval of the posterior distribution of p to be only 0.05 wide. This means the distance between the upper and lower bound of the interval should be 0.05. How many times will you have to toss the globe to do this?

    # DON'T KNOW
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise - Hard
    """)
    return


@app.cell
def _(np):
    # The practice problems here all use the data below. These data indicate the gender (male=1, female=0) of officially reported first and second born children in 100 two-child families.
    birth1 = np.array(
        [
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
        ]
    )

    birth2 = np.array(
        [
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
        ]
    )

    # So for example, the first family in the data reported a boy (1) and then a girl (0). The second family reported a girl (0) and then a boy (1). The third family reported two girls.
    return birth1, birth2


@app.cell
def _(az, binom, birth1, birth2, grid_approx_ch2, np):
    # 3H1. Using grid approximation, compute the posterior distribution for the probability of a birth being a boy. Assume a uniform prior probability. Which parameter value maximizes the posterior probability?

    _tosses = birth1.size + birth2.size
    _success = np.concat((birth1, birth2)).sum()

    # using the function made for chapter 2
    ch3_h1_df, chart_ch3_h1 = grid_approx_ch2(p_grid_size=101, prior="uniform", success=_success, tosses=_tosses)
    # 3h1 = ch3_h1_df.filter(pl.col("posterior_prob") == pl.col("posterior_prob").max()).select("prob_water")

    _p_grid = np.linspace(0, 1, 101)
    _prior = np.ones_like(_p_grid)
    _likelihood = binom.pmf(k=_success, n=_tosses, p=_p_grid)
    _posterior = (_likelihood * _prior) / (_likelihood * _prior).sum()

    _3h1 = _p_grid[_posterior.argmax()]

    # 3H2. Using the sample function, draw 10,000 random parameter values from the posterior distribution you calculated above. Use these samples to estimate the 50%, 89%, and 97% highest posterior density intervals.
    _samples = np.random.choice(
        _p_grid,
        10_000,
        replace=True,
        p=_posterior,
    )

    _50_hdi = az.hdi(ary=_samples, hdi_prob=0.5)
    _89_hdi = az.hdi(ary=_samples, hdi_prob=0.89)
    _97_hdi = az.hdi(ary=_samples, hdi_prob=0.97)

    # 3H3. Use rbinom to simulate 10,000 replicates of 200 births. You should end up with 10,000 numbers, each one a count of boys out of 200 births. Compare the distribution of predicted numbers of boys to the actual count in the data (111 boys out of 200 births). There are many good ways to visualize the simulations, but the dens command (part of the rethinking package) is probably the easiest way in this case. Does it look like the model fits the data well? That is, does the distribution of predictions include the actual observation as a central, likely outcome?

    _ppd = np.random.binomial(n=200, p=_samples)

    # plt.hist(_ppd)
    # plt.axvline(_success, c='r')
    # plt.show()

    # 3H4. Now compare 10,000 counts of boys from 100 simulated first borns only to the number of boys in the first births, birth1. How does the model look in this light?

    # _ppd100 = np.random.binomial(n=100, p=_samples)
    # plt.axvline(np.sum(birth1), c='r')
    # plt.hist(_ppd100)
    # plt.show()

    # 3H5. The model assumes that sex of first and second births are independent. To check this assumption, focus now on second births that followed female first borns. Compare 10,000 simulated counts of boys to only those second births that followed girls. To do this correctly, you need to count the number of first borns who were girls and simulate that many births, 10,000 times. Compare the counts of boys in your simulations to the actual observed count of boys following girls. How does the model look in this light? Any guesses what is going on in these data?

    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
