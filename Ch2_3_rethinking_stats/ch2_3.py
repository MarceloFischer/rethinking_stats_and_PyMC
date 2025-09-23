import marimo

__generated_with = "0.16.1"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""### Imports""")
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import scipy as sp
    from scipy.stats import binom
    import pymc as pm
    import altair as alt
    return alt, binom, mo, np, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Model Fitting""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell
def _(binom):
    # get the likelihood of the data for several different values of p
    binom.pmf(k=6, n=9, p=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The above means that
    $$W \sim \mathrm{Binomial}(N,\, p), \text{where } N=W+L$$
    and 
    $$p \sim \mathrm{Uniform}(0, 1)$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$Posterior = \frac{Probability\_of\_the\_data \times Prior}{Marginal\_probability}$$

    $$Pr(p\mid W,L) = \frac{Pr(W,L \mid p) \times Pr(p)}{Pr(W,L)}$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Grid Approximation

    Awful for scalability, but a good starting point to get the theory behind the models
    """
    )
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Quadratic Approximation

    This approximation essentially represents any log-posterior with a parabola. This is because under quite general conditions, the region near the peak of the posterior distribution will be nearly Gaussian in shape.
    """
    )
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""### Exercise 1 - Medium""")
    return


@app.cell
def _(grid_approx_ch2, mo):
    post_df, globe_tosses_chart = grid_approx_ch2(5, success=2, tosses=3)

    mo.ui.altair_chart(globe_tosses_chart)
    return


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
    mo.md(r"""### Exercise 2 - Medium""")
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
    mo.md(r"""### Exercise 3 - Medium""")
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
    mo.md(r"""### Exercise 4 - Medium""")
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
    mo.md(r"""### Exercise 5 - Medium""")
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
    mo.md(r"""### Exercise 6 - Medium""")
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
    mo.md(r"""### Exercise 7 - Medium""")
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


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(column=2)
def _():
    return


if __name__ == "__main__":
    app.run()
