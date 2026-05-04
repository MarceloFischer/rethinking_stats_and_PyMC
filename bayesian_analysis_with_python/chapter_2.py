import marimo

__generated_with = "0.23.4"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    import marimo as mo
    from itertools import combinations
    import altair as alt
    import arviz as az
    import preliz as pz
    import matplotlib.pyplot as plt
    import marimo as moe
    import numpy as np
    import polars as pl
    import pymc as pm
    import scipy.stats as stats
    from pathlib import Path
    from wigglystuff import EdgeDraw

    RANDOM_SEED = 1523
    rng = np.random.default_rng(RANDOM_SEED)

    plt.style.use("fivethirtyeight")
    # Set default figure size to 10 inches wide by 6 inches tall
    plt.rcParams["figure.figsize"] = (14, 7)
    # Make the layout "tight" by default so labels don't overlap
    plt.rcParams["figure.autolayout"] = True
    # sets default credible interval used by arviz
    az.rcParams["stats.ci_prob"] = 0.89
    return (
        Path,
        RANDOM_SEED,
        alt,
        az,
        combinations,
        mo,
        np,
        pl,
        plt,
        pm,
        pz,
        rng,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Constants
    """)
    return


@app.cell
def _(Path, pl):
    TIPS_PATH = Path(__file__).parent.parent / "data" / "tips.csv"

    tips = pl.read_csv(TIPS_PATH)

    tips
    return (tips,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Distribution Overview
    """)
    return


@app.cell
def _(mo):
    mu_slider = mo.ui.slider(0, 10, step=0.1, show_value=True, label="mu")
    sigma_slider = mo.ui.slider(1, 10, step=0.1, show_value=True, label="sigma")
    nu_slider = mo.ui.slider(0.1, 20, step=0.5, show_value=True, label="nu")
    return mu_slider, nu_slider, sigma_slider


@app.cell
def _(mo, mu_slider, nu_slider, pz, sigma_slider):
    mean, var, skewness, kurtosis = pz.StudentT(
        mu=mu_slider.value, sigma=sigma_slider.value, nu=nu_slider.value
    ).moments()

    mo.vstack(
        [
            mo.hstack([mu_slider, sigma_slider, nu_slider]),
            mo.md(
                f"**Mean:** {mean:.2f} | **Variance:** {var:.2f} | **Skewness:** {skewness:.2f} | **Kurtosis:** {kurtosis:.2f}"
            ),
            pz.StudentT(
                mu=mu_slider.value, sigma=sigma_slider.value, nu=nu_slider.value
            ).plot_pdf(),
            # pz.Normal(0, 1).plot_pdf()
        ],
        align="stretch",
        gap=1,
    )
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    # Tips Example
    """)
    return


@app.cell
def _(az, pl, plt, tips):
    _tips_dict = {
        day: tips.filter(pl.col("day") == day)["tip"].to_numpy()
        for day in tips["day"].unique().to_list()
    }

    az.plot_forest(
        _tips_dict,
        kind="ridgeplot",
        ridgeplot_truncate=False,
        ridgeplot_quantiles=[0.25, 0.5, 0.75],
        ridgeplot_overlap=2.2,
    )

    plt.title("Distribution of Tips per Weekday")
    plt.xlabel("Tip Amount")
    plt.gca()
    return


@app.cell
def _(alt, tips):
    ridge_plot = (
        alt.Chart(tips)
        .transform_density("tip", as_=["tip", "density"], groupby=["day"])
        .mark_area(
            interpolate="monotone", fillOpacity=0.8, stroke="lightgray", strokeWidth=0.5
        )
        .encode(
            alt.X("tip:Q", title="Tip Amount ($)"),
            alt.Y("density:Q", stack=None, title=None, axis=None),
            alt.Fill("day:N", legend=None, scale=alt.Scale(scheme="viridis")),
            tooltip=[
                alt.Tooltip("day:N", title="Day"),
                alt.Tooltip("tip:Q", title="Tip", format="$.2f"),
                alt.Tooltip("density:Q", title="Density", format=".4f"),
            ],
        )
        .properties(height=90, width="container")
        .facet(
            row=alt.Row(
                "day:N",
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="right"),
                sort=["Thur", "Fri", "Sat", "Sun"],
            )
        )
        .configure_facet(spacing=0)
        .configure_view(stroke=None)
    )

    ridge_plot
    return


@app.cell
def _(pl, tips):
    def create_group_vars_tips(dataf: pl.DataFrame = tips):
        """
        Creates a mapping of unique days and their corresponding integer indices from the tips dataset.

        Args:
            dataf: A polars DataFrame containing a 'day' column. Defaults to the global 'tips' dataframe.

        Returns:
            tuple: A tuple containing (days, day_idx) where 'days' is an array of unique day names
                   and 'day_idx' is an array of integer codes mapping each row to a day.
        """
        days = dataf["day"].unique().to_numpy()
        day_idx = dataf["day"].cast(pl.Enum(days)).to_physical().to_numpy()

        return days, day_idx


    days, day_idx = create_group_vars_tips()
    return day_idx, days


@app.cell
def _(day_idx, days, pm, rng, tips):
    def tips_model():
        """
        Constructs and samples from a Bayesian model to estimate tips per weekday.

        The model assumes a Normal likelihood for tip amounts, with separate
        mu and sigma parameters for each day of the week.

        Returns:
            tuple: (model, idata) where 'model' is the PyMC model object
                   and 'idata' is the InferenceData object containing samples.
        """
        coords = {"days": days, "days_flat": days[day_idx]}
        with pm.Model(coords=coords) as model:
            # priors
            mu = pm.HalfNormal("mu", sigma=5, dims="days", rng=rng)
            sigma = pm.HalfNormal("sigma", sigma=3, dims="days", rng=rng)

            # likelihood
            obs = pm.Normal(
                "obs",
                mu=mu[day_idx],
                sigma=sigma[day_idx],
                observed=tips["tip"],
                dims="days_flat",
                rng=rng,
            )

            idata = pm.sample(random_seed=rng)

        return model, idata


    day_tip_model, day_tip_idata = tips_model()
    return day_tip_idata, day_tip_model


@app.cell
def _(day_tip_model):
    day_tip_model
    return


@app.cell
def _(az, day_tip_idata):
    az.plot_posterior(day_tip_idata, figsize=(14, 7))
    return


@app.cell
def _(RANDOM_SEED, az, day_tip_idata, day_tip_model, days, plt, pm):
    with day_tip_model:
        day_tip_idata.extend(pm.sample_posterior_predictive(day_tip_idata))

    _, axes = plt.subplots(2, 2)
    az.plot_ppc(
        day_tip_idata,
        num_pp_samples=100,
        coords={"days_flat": days},
        flatten=[],
        ax=axes,
        random_seed=RANDOM_SEED,
    )
    return


@app.cell
def _(diffs):
    diffs
    return


@app.cell
def _(az, combinations, day_tip_idata, days):
    diffs = {}
    for day_1, day_2 in combinations(days, 2):
        diffs[f"{day_1}-{day_2}"] = day_tip_idata.posterior["mu"].sel(
            days=day_1
        ) - day_tip_idata.posterior["mu"].sel(days=day_2)

    az.plot_posterior(diffs, ref_val=0, figsize=(14, 7))
    return (diffs,)


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercises
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercice 1
    """)
    return


@app.cell
def _(pm, pz, rng):
    def ex_1(alpha: float, beta: float):
        trials = 4
        theta_real = 0.35  # unknown value in a real experiment
        data = pz.Binomial(n=1, p=theta_real).rvs(trials)

        with pm.Model() as our_first_model:
            # prior
            theta = pm.Beta("theta", alpha=alpha, beta=beta, rng=rng)
            # likelihood
            obs = pm.Bernoulli("obs", p=theta, observed=data, rng=rng)

            idata = pm.sample(random_seed=rng)

        return our_first_model, idata


    ex_1_beta_params = [(1, 1), (20, 20), (1, 4)]
    models_dict_ex1 = {}
    # for idx, (para_alpha, para_beta) in enumerate(ex_1_beta_params):
    #     model, idata = ex_1(alpha=para_alpha, beta=para_beta)
    #     models_dict_ex1[f"model_{idx}"] = {"model": model, "idata": idata}


    # def _():
    #     # Create a figure with enough subplots
    #     num_models = len(models_dict_ex1)
    #     fig, axes = plt.subplots(num_models, 1, figsize=(10, 4 * num_models))

    #     for i, model_name in enumerate(models_dict_ex1):
    #         az.plot_posterior(
    #             models_dict_ex1[model_name]["idata"], ax=axes[i], ref_val=0.35
    #         )
    #         axes[i].set_title(f"Posterior for {model_name}")

    #     # The last expression returns the figure/axis to be displayed
    #     return plt.gca()


    # _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Com $\alpha=1, \beta=1$ a distribuicao a priori e igual a uniforme entre 0 e 1. Como o parametro esta nesse intervalo, a posteriori consegue estimar o parametro e visitar todos os valores do intervalo sem nenhuma preferencia. Isso facilita Azer o fit dos dados de forma melhor.

    Com $\alpha=20, \beta=20$ a distribuicao a priori concentra muito da densidade de probabilidade em torno de $0.5$ e fax com que valores longe da media sejam pouco provaveis. Isso cria um vies no modelo que acredita que a moeda tem mais chance se der honesta do que nao.

    Com $\alpha=1, \beta=4$ a distribuicao a priori da mais imporatancia para probabilidades baixas para o parametro, mas deixa o modelo livre para explorar varios valores altos tambem.

    O primeiro e o ultimo casos fazem com que a distribuicao posteriori sejam amis fieas a realidade, por nao imporem uma tendencia no modela da qual nao temos conhecimento (nao sabemos se a moeda eh honesta ou nao, o objetivo eh inferir isso). Portanto, o melhor seria nao forcar o modelo a ficar proximo de $0.5$.
    """)
    return


@app.cell
def _(pz):
    pz.Beta(alpha=1, beta=4).plot_pdf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercice 2
    """)
    return


@app.cell
def _(pm, pz, rng):
    def ex_2(lower: float, upper: float):
        trials = 4
        theta_real = 0.35  # unknown value in a real experiment
        data = pz.Binomial(n=1, p=theta_real).rvs(trials)

        with pm.Model() as our_first_model:
            # prior
            theta = pm.Uniform("theta", lower=lower, upper=upper, rng=rng)
            # likelihood
            obs = pm.Bernoulli("obs", p=theta, observed=data, rng=rng)

            idata = pm.sample(random_seed=rng)

        return our_first_model, idata


    # model_ex2, idata_ex2 = ex_2(lower=0, upper=1)

    # az.plot_posterior(idata_ex2, ref_val=0.35)
    return


@app.cell
def _():
    # model_ex2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercice 4
    """)
    return


@app.cell
def _(np, pl, plt):
    # fmt: off
    disaster_data = pl.Series(
        [4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
        3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
        2, 2, 3, 4, 2, 1, 3, np.nan, 2, 1, 1, 1, 1, 3, 0, 0,
        1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
        0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
        3, 3, 1, np.nan, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1], strict=False
    )
    # fmt: on
    years = np.arange(1851, 1962)

    plt.plot(years, disaster_data, "o", markersize=8, alpha=0.4)
    plt.ylabel("Disaster count")
    plt.xlabel("Year")
    return disaster_data, years


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    \begin{split}
    \begin{aligned}
      D_t &\sim \text{Pois}(r_t), r_t= \begin{cases}
       e, & \text{if } t \le s \\
       l, & \text{if } t \gt s
       \end{cases} \\
      s &\sim \text{Unif}(t_l, t_h)\\
      e &\sim \text{exp}(1)\\
      l &\sim \text{exp}(1)
    \end{aligned}
    \end{split}

    <p>the parameters are defined as follows:</p>
    <ul class="simple">
    <li><p><span class="math notranslate nohighlight"><mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" overflow="overflow" style="font-size: 119.5%;" tabindex="0" ctxtmenu_counter="34" data-semantic-locale="en" data-semantic-domain="clearspeak" data-semantic-style="default" data-semantic-domain2style="mathspeak:default,clearspeak:default" data-semantic-collapsible="collapsible" data-semantic-expandable="expandable" data-semantic-level="Level" data-semantic-speech-none="D sub t" data-semantic-speech="&lt;prosody pitch=&quot;+30%&quot;&gt; &lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;D&lt;/say-as&gt; &lt;/prosody&gt; &lt;mark name=&quot;1&quot;/&gt; sub &lt;say-as interpret-as=&quot;character&quot;&gt;t&lt;/say-as&gt;" data-speech-attached="true" data-semantic-braille="⠠⠙⠰⠞" data-braille-attached="true" has-speech="true"><mjx-math data-latex="D_t" data-semantic-structure="(2 0 1)" class="NCM-N" aria-hidden="true"><mjx-msub data-latex="D_t" data-semantic-type="subscript" data-semantic-role="latinletter" data-semantic-annotation="depth:1" data-semantic-id="2" data-semantic-children="0,1" data-semantic-attributes="latex:D_t" data-semantic-owns="0 1" data-semantic-level-number="0" data-speech-node="true" data-semantic-speech-none="D sub t" data-semantic-summary-none="subscript" data-semantic-speech="&lt;prosody pitch=&quot;+30%&quot;&gt; &lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;D&lt;/say-as&gt; &lt;/prosody&gt; &lt;mark name=&quot;1&quot;/&gt; sub &lt;say-as interpret-as=&quot;character&quot;&gt;t&lt;/say-as&gt;" data-semantic-summary="&lt;mark name=&quot;2&quot;/&gt; subscript" data-semantic-braille="⠠⠙⠰⠞"><mjx-mi data-latex="D" data-semantic-type="identifier" data-semantic-role="latinletter" data-semantic-font="italic" data-semantic-annotation="clearspeak:simple;nemeth:number;depth:2" data-semantic-id="0" data-semantic-parent="2" data-semantic-attributes="latex:D" data-semantic-level-number="1" data-speech-node="true" data-semantic-speech-none="D" data-semantic-prefix-none="Base" data-semantic-summary-none="identifier" data-semantic-speech="&lt;prosody pitch=&quot;+30%&quot;&gt; &lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;D&lt;/say-as&gt; &lt;/prosody&gt;" data-semantic-prefix="&lt;mark name=&quot;0&quot;/&gt; Base" data-semantic-summary="&lt;mark name=&quot;0&quot;/&gt; identifier" data-semantic-braille="⠠⠙"><mjx-c class="mjx-c1D437">𝐷</mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em;"><mjx-mi size="s" data-latex="t" data-semantic-type="identifier" data-semantic-role="latinletter" data-semantic-font="italic" data-semantic-annotation="clearspeak:simple;depth:2" data-semantic-id="1" data-semantic-parent="2" data-semantic-attributes="latex:t" data-semantic-level-number="1" data-speech-node="true" data-semantic-speech-none="t" data-semantic-prefix-none="Subscript" data-semantic-summary-none="identifier" data-semantic-speech="&lt;mark name=&quot;1&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;t&lt;/say-as&gt;" data-semantic-prefix="&lt;mark name=&quot;1&quot;/&gt; Subscript" data-semantic-summary="&lt;mark name=&quot;1&quot;/&gt; identifier" data-semantic-braille="⠞"><mjx-c class="mjx-c1D461">𝑡</mjx-c></mjx-mi></mjx-script></mjx-msub></mjx-math><mjx-speech aria-label="D sub t, math" role="img" aria-roledescription="" aria-braillelabel="⠠⠙⠰⠞" aria-brailleroledescription="⠀"></mjx-speech></mjx-container></span>: The number of disasters in year <span class="math notranslate nohighlight"><mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" overflow="overflow" style="font-size: 119.5%;" tabindex="0" ctxtmenu_counter="35" data-semantic-locale="en" data-semantic-domain="clearspeak" data-semantic-style="default" data-semantic-domain2style="mathspeak:default,clearspeak:default" data-semantic-collapsible="collapsible" data-semantic-expandable="expandable" data-semantic-level="Level" data-semantic-speech-none="t" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;t&lt;/say-as&gt;" data-speech-attached="true" data-semantic-braille="⠞" data-braille-attached="true" has-speech="true"><mjx-math data-latex="t" data-semantic-structure="0" class="NCM-N" aria-hidden="true"><mjx-mi data-latex="t" data-semantic-type="identifier" data-semantic-role="latinletter" data-semantic-font="italic" data-semantic-annotation="clearspeak:simple;nemeth:number;depth:1" data-semantic-id="0" data-semantic-attributes="latex:t" data-semantic-level-number="0" data-speech-node="true" data-semantic-speech-none="t" data-semantic-summary-none="identifier" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;t&lt;/say-as&gt;" data-semantic-summary="&lt;mark name=&quot;0&quot;/&gt; identifier" data-semantic-braille="⠞"><mjx-c class="mjx-c1D461">𝑡</mjx-c></mjx-mi></mjx-math><mjx-speech aria-label="t, math" role="img" aria-roledescription="" aria-braillelabel="⠞" aria-brailleroledescription="⠀"></mjx-speech></mjx-container></span></p></li>
    <li><p><span class="math notranslate nohighlight"><mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" overflow="overflow" style="font-size: 119.5%;" tabindex="0" ctxtmenu_counter="36" data-semantic-locale="en" data-semantic-domain="clearspeak" data-semantic-style="default" data-semantic-domain2style="mathspeak:default,clearspeak:default" data-semantic-collapsible="collapsible" data-semantic-expandable="expandable" data-semantic-level="Level" data-semantic-speech-none="r sub t" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;r&lt;/say-as&gt; &lt;mark name=&quot;1&quot;/&gt; sub &lt;say-as interpret-as=&quot;character&quot;&gt;t&lt;/say-as&gt;" data-speech-attached="true" data-semantic-braille="⠗⠰⠞" data-braille-attached="true" has-speech="true"><mjx-math data-latex="r_t" data-semantic-structure="(2 0 1)" class="NCM-N" aria-hidden="true"><mjx-msub data-latex="r_t" data-semantic-type="subscript" data-semantic-role="latinletter" data-semantic-annotation="depth:1" data-semantic-id="2" data-semantic-children="0,1" data-semantic-attributes="latex:r_t" data-semantic-owns="0 1" data-semantic-level-number="0" data-speech-node="true" data-semantic-speech-none="r sub t" data-semantic-summary-none="subscript" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;r&lt;/say-as&gt; &lt;mark name=&quot;1&quot;/&gt; sub &lt;say-as interpret-as=&quot;character&quot;&gt;t&lt;/say-as&gt;" data-semantic-summary="&lt;mark name=&quot;2&quot;/&gt; subscript" data-semantic-braille="⠗⠰⠞"><mjx-mi data-latex="r" data-semantic-type="identifier" data-semantic-role="latinletter" data-semantic-font="italic" data-semantic-annotation="clearspeak:simple;nemeth:number;depth:2" data-semantic-id="0" data-semantic-parent="2" data-semantic-attributes="latex:r" data-semantic-level-number="1" data-speech-node="true" data-semantic-speech-none="r" data-semantic-prefix-none="Base" data-semantic-summary-none="identifier" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;r&lt;/say-as&gt;" data-semantic-prefix="&lt;mark name=&quot;0&quot;/&gt; Base" data-semantic-summary="&lt;mark name=&quot;0&quot;/&gt; identifier" data-semantic-braille="⠗"><mjx-c class="mjx-c1D45F">𝑟</mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em;"><mjx-mi size="s" data-latex="t" data-semantic-type="identifier" data-semantic-role="latinletter" data-semantic-font="italic" data-semantic-annotation="clearspeak:simple;depth:2" data-semantic-id="1" data-semantic-parent="2" data-semantic-attributes="latex:t" data-semantic-level-number="1" data-speech-node="true" data-semantic-speech-none="t" data-semantic-prefix-none="Subscript" data-semantic-summary-none="identifier" data-semantic-speech="&lt;mark name=&quot;1&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;t&lt;/say-as&gt;" data-semantic-prefix="&lt;mark name=&quot;1&quot;/&gt; Subscript" data-semantic-summary="&lt;mark name=&quot;1&quot;/&gt; identifier" data-semantic-braille="⠞"><mjx-c class="mjx-c1D461">𝑡</mjx-c></mjx-mi></mjx-script></mjx-msub></mjx-math><mjx-speech aria-label="r sub t, math" role="img" aria-roledescription="" aria-braillelabel="⠗⠰⠞" aria-brailleroledescription="⠀"></mjx-speech></mjx-container></span>: The rate parameter of the Poisson distribution of disasters in year <span class="math notranslate nohighlight"><mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" overflow="overflow" style="font-size: 119.5%;" tabindex="0" ctxtmenu_counter="37" data-semantic-locale="en" data-semantic-domain="clearspeak" data-semantic-style="default" data-semantic-domain2style="mathspeak:default,clearspeak:default" data-semantic-collapsible="collapsible" data-semantic-expandable="expandable" data-semantic-level="Level" data-semantic-speech-none="t" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;t&lt;/say-as&gt;" data-speech-attached="true" data-semantic-braille="⠞" data-braille-attached="true" has-speech="true"><mjx-math data-latex="t" data-semantic-structure="0" class="NCM-N" aria-hidden="true"><mjx-mi data-latex="t" data-semantic-type="identifier" data-semantic-role="latinletter" data-semantic-font="italic" data-semantic-annotation="clearspeak:simple;nemeth:number;depth:1" data-semantic-id="0" data-semantic-attributes="latex:t" data-semantic-level-number="0" data-speech-node="true" data-semantic-speech-none="t" data-semantic-summary-none="identifier" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;t&lt;/say-as&gt;" data-semantic-summary="&lt;mark name=&quot;0&quot;/&gt; identifier" data-semantic-braille="⠞"><mjx-c class="mjx-c1D461">𝑡</mjx-c></mjx-mi></mjx-math><mjx-speech aria-label="t, math" role="img" aria-roledescription="" aria-braillelabel="⠞" aria-brailleroledescription="⠀"></mjx-speech></mjx-container></span>.</p></li>
    <li><p><span class="math notranslate nohighlight"><mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" overflow="overflow" style="font-size: 119.5%;" tabindex="0" ctxtmenu_counter="38" data-semantic-locale="en" data-semantic-domain="clearspeak" data-semantic-style="default" data-semantic-domain2style="mathspeak:default,clearspeak:default" data-semantic-collapsible="collapsible" data-semantic-expandable="expandable" data-semantic-level="Level" data-semantic-speech-none="s" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;s&lt;/say-as&gt;" data-speech-attached="true" data-semantic-braille="⠎" data-braille-attached="true" has-speech="true"><mjx-math data-latex="s" data-semantic-structure="0" class="NCM-N" aria-hidden="true"><mjx-mi data-latex="s" data-semantic-type="identifier" data-semantic-role="latinletter" data-semantic-font="italic" data-semantic-annotation="clearspeak:simple;nemeth:number;depth:1" data-semantic-id="0" data-semantic-attributes="latex:s" data-semantic-level-number="0" data-speech-node="true" data-semantic-speech-none="s" data-semantic-summary-none="identifier" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;s&lt;/say-as&gt;" data-semantic-summary="&lt;mark name=&quot;0&quot;/&gt; identifier" data-semantic-braille="⠎"><mjx-c class="mjx-c1D460">𝑠</mjx-c></mjx-mi></mjx-math><mjx-speech aria-label="s, math" role="img" aria-roledescription="" aria-braillelabel="⠎" aria-brailleroledescription="⠀"></mjx-speech></mjx-container></span>: The year in which the rate parameter changes (the switchpoint).</p></li>
    <li><p><span class="math notranslate nohighlight"><mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" overflow="overflow" style="font-size: 119.5%;" tabindex="0" ctxtmenu_counter="39" data-semantic-locale="en" data-semantic-domain="clearspeak" data-semantic-style="default" data-semantic-domain2style="mathspeak:default,clearspeak:default" data-semantic-collapsible="collapsible" data-semantic-expandable="expandable" data-semantic-level="Level" data-semantic-speech-none="e" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;e&lt;/say-as&gt;" data-speech-attached="true" data-semantic-braille="⠑" data-braille-attached="true" has-speech="true"><mjx-math data-latex="e" data-semantic-structure="0" class="NCM-N" aria-hidden="true"><mjx-mi data-latex="e" data-semantic-type="identifier" data-semantic-role="latinletter" data-semantic-font="italic" data-semantic-annotation="clearspeak:simple;nemeth:number;depth:1" data-semantic-id="0" data-semantic-attributes="latex:e" data-semantic-level-number="0" data-speech-node="true" data-semantic-speech-none="e" data-semantic-summary-none="identifier" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;e&lt;/say-as&gt;" data-semantic-summary="&lt;mark name=&quot;0&quot;/&gt; identifier" data-semantic-braille="⠑"><mjx-c class="mjx-c1D452">𝑒</mjx-c></mjx-mi></mjx-math><mjx-speech aria-label="e, math" role="img" aria-roledescription="" aria-braillelabel="⠑" aria-brailleroledescription="⠀"></mjx-speech></mjx-container></span>: The rate parameter before the switchpoint <span class="math notranslate nohighlight"><mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" overflow="overflow" style="font-size: 119.5%;" tabindex="0" ctxtmenu_counter="40" data-semantic-locale="en" data-semantic-domain="clearspeak" data-semantic-style="default" data-semantic-domain2style="mathspeak:default,clearspeak:default" data-semantic-collapsible="collapsible" data-semantic-expandable="expandable" data-semantic-level="Level" data-semantic-speech-none="s" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;s&lt;/say-as&gt;" data-speech-attached="true" data-semantic-braille="⠎" data-braille-attached="true" has-speech="true"><mjx-math data-latex="s" data-semantic-structure="0" class="NCM-N" aria-hidden="true"><mjx-mi data-latex="s" data-semantic-type="identifier" data-semantic-role="latinletter" data-semantic-font="italic" data-semantic-annotation="clearspeak:simple;nemeth:number;depth:1" data-semantic-id="0" data-semantic-attributes="latex:s" data-semantic-level-number="0" data-speech-node="true" data-semantic-speech-none="s" data-semantic-summary-none="identifier" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;s&lt;/say-as&gt;" data-semantic-summary="&lt;mark name=&quot;0&quot;/&gt; identifier" data-semantic-braille="⠎"><mjx-c class="mjx-c1D460">𝑠</mjx-c></mjx-mi></mjx-math><mjx-speech aria-label="s, math" role="img" aria-roledescription="" aria-braillelabel="⠎" aria-brailleroledescription="⠀"></mjx-speech></mjx-container></span>.</p></li>
    <li><p><span class="math notranslate nohighlight"><mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" overflow="overflow" style="font-size: 119.5%;" tabindex="0" ctxtmenu_counter="41" data-semantic-locale="en" data-semantic-domain="clearspeak" data-semantic-style="default" data-semantic-domain2style="mathspeak:default,clearspeak:default" data-semantic-collapsible="collapsible" data-semantic-expandable="expandable" data-semantic-level="Level" data-semantic-speech-none="l" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;l&lt;/say-as&gt;" data-speech-attached="true" data-semantic-braille="⠇" data-braille-attached="true" has-speech="true"><mjx-math data-latex="l" data-semantic-structure="0" class="NCM-N" aria-hidden="true"><mjx-mi data-latex="l" data-semantic-type="identifier" data-semantic-role="latinletter" data-semantic-font="italic" data-semantic-annotation="clearspeak:simple;nemeth:number;depth:1" data-semantic-id="0" data-semantic-attributes="latex:l" data-semantic-level-number="0" data-speech-node="true" data-semantic-speech-none="l" data-semantic-summary-none="identifier" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;l&lt;/say-as&gt;" data-semantic-summary="&lt;mark name=&quot;0&quot;/&gt; identifier" data-semantic-braille="⠇"><mjx-c class="mjx-c1D459">𝑙</mjx-c></mjx-mi></mjx-math><mjx-speech aria-label="l, math" role="img" aria-roledescription="" aria-braillelabel="⠇" aria-brailleroledescription="⠀"></mjx-speech></mjx-container></span>: The rate parameter after the switchpoint <span class="math notranslate nohighlight"><mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" overflow="overflow" style="font-size: 119.5%;" tabindex="0" ctxtmenu_counter="42" data-semantic-locale="en" data-semantic-domain="clearspeak" data-semantic-style="default" data-semantic-domain2style="mathspeak:default,clearspeak:default" data-semantic-collapsible="collapsible" data-semantic-expandable="expandable" data-semantic-level="Level" data-semantic-speech-none="s" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;s&lt;/say-as&gt;" data-speech-attached="true" data-semantic-braille="⠎" data-braille-attached="true" has-speech="true"><mjx-math data-latex="s" data-semantic-structure="0" class="NCM-N" aria-hidden="true"><mjx-mi data-latex="s" data-semantic-type="identifier" data-semantic-role="latinletter" data-semantic-font="italic" data-semantic-annotation="clearspeak:simple;nemeth:number;depth:1" data-semantic-id="0" data-semantic-attributes="latex:s" data-semantic-level-number="0" data-speech-node="true" data-semantic-speech-none="s" data-semantic-summary-none="identifier" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;s&lt;/say-as&gt;" data-semantic-summary="&lt;mark name=&quot;0&quot;/&gt; identifier" data-semantic-braille="⠎"><mjx-c class="mjx-c1D460">𝑠</mjx-c></mjx-mi></mjx-math><mjx-speech aria-label="s, math" role="img" aria-roledescription="" aria-braillelabel="⠎" aria-brailleroledescription="⠀"></mjx-speech></mjx-container></span>.</p></li>
    <li><p><span class="math notranslate nohighlight"><mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" overflow="overflow" style="font-size: 119.5%;" tabindex="0" ctxtmenu_counter="43" data-semantic-locale="en" data-semantic-domain="clearspeak" data-semantic-style="default" data-semantic-domain2style="mathspeak:default,clearspeak:default" data-semantic-collapsible="collapsible" data-semantic-expandable="expandable" data-semantic-level="Level" data-semantic-speech-none="t sub l" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;t&lt;/say-as&gt; &lt;mark name=&quot;1&quot;/&gt; sub &lt;say-as interpret-as=&quot;character&quot;&gt;l&lt;/say-as&gt;" data-speech-attached="true" data-semantic-braille="⠞⠰⠇" data-braille-attached="true" has-speech="true"><mjx-math data-latex="t_l" data-semantic-structure="(2 0 1)" class="NCM-N" aria-hidden="true"><mjx-msub data-latex="t_l" data-semantic-type="subscript" data-semantic-role="latinletter" data-semantic-annotation="depth:1" data-semantic-id="2" data-semantic-children="0,1" data-semantic-attributes="latex:t_l" data-semantic-owns="0 1" data-semantic-level-number="0" data-speech-node="true" data-semantic-speech-none="t sub l" data-semantic-summary-none="subscript" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;t&lt;/say-as&gt; &lt;mark name=&quot;1&quot;/&gt; sub &lt;say-as interpret-as=&quot;character&quot;&gt;l&lt;/say-as&gt;" data-semantic-summary="&lt;mark name=&quot;2&quot;/&gt; subscript" data-semantic-braille="⠞⠰⠇"><mjx-mi data-latex="t" data-semantic-type="identifier" data-semantic-role="latinletter" data-semantic-font="italic" data-semantic-annotation="clearspeak:simple;nemeth:number;depth:2" data-semantic-id="0" data-semantic-parent="2" data-semantic-attributes="latex:t" data-semantic-level-number="1" data-speech-node="true" data-semantic-speech-none="t" data-semantic-prefix-none="Base" data-semantic-summary-none="identifier" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;t&lt;/say-as&gt;" data-semantic-prefix="&lt;mark name=&quot;0&quot;/&gt; Base" data-semantic-summary="&lt;mark name=&quot;0&quot;/&gt; identifier" data-semantic-braille="⠞"><mjx-c class="mjx-c1D461">𝑡</mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em;"><mjx-mi size="s" data-latex="l" data-semantic-type="identifier" data-semantic-role="latinletter" data-semantic-font="italic" data-semantic-annotation="clearspeak:simple;depth:2" data-semantic-id="1" data-semantic-parent="2" data-semantic-attributes="latex:l" data-semantic-level-number="1" data-speech-node="true" data-semantic-speech-none="l" data-semantic-prefix-none="Subscript" data-semantic-summary-none="identifier" data-semantic-speech="&lt;mark name=&quot;1&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;l&lt;/say-as&gt;" data-semantic-prefix="&lt;mark name=&quot;1&quot;/&gt; Subscript" data-semantic-summary="&lt;mark name=&quot;1&quot;/&gt; identifier" data-semantic-braille="⠇"><mjx-c class="mjx-c1D459">𝑙</mjx-c></mjx-mi></mjx-script></mjx-msub></mjx-math><mjx-speech aria-label="t sub l, math" role="img" aria-roledescription="" aria-braillelabel="⠞⠰⠇" aria-brailleroledescription="⠀"></mjx-speech></mjx-container></span>, <span class="math notranslate nohighlight"><mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" overflow="overflow" style="font-size: 119.5%;" tabindex="0" ctxtmenu_counter="44" data-semantic-locale="en" data-semantic-domain="clearspeak" data-semantic-style="default" data-semantic-domain2style="mathspeak:default,clearspeak:default" data-semantic-collapsible="collapsible" data-semantic-expandable="expandable" data-semantic-level="Level" data-semantic-speech-none="t sub h" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;t&lt;/say-as&gt; &lt;mark name=&quot;1&quot;/&gt; sub &lt;say-as interpret-as=&quot;character&quot;&gt;h&lt;/say-as&gt;" data-speech-attached="true" data-semantic-braille="⠞⠰⠓" data-braille-attached="true" has-speech="true"><mjx-math data-latex="t_h" data-semantic-structure="(2 0 1)" class="NCM-N" aria-hidden="true"><mjx-msub data-latex="t_h" data-semantic-type="subscript" data-semantic-role="latinletter" data-semantic-annotation="depth:1" data-semantic-id="2" data-semantic-children="0,1" data-semantic-attributes="latex:t_h" data-semantic-owns="0 1" data-semantic-level-number="0" data-speech-node="true" data-semantic-speech-none="t sub h" data-semantic-summary-none="subscript" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;t&lt;/say-as&gt; &lt;mark name=&quot;1&quot;/&gt; sub &lt;say-as interpret-as=&quot;character&quot;&gt;h&lt;/say-as&gt;" data-semantic-summary="&lt;mark name=&quot;2&quot;/&gt; subscript" data-semantic-braille="⠞⠰⠓"><mjx-mi data-latex="t" data-semantic-type="identifier" data-semantic-role="latinletter" data-semantic-font="italic" data-semantic-annotation="clearspeak:simple;nemeth:number;depth:2" data-semantic-id="0" data-semantic-parent="2" data-semantic-attributes="latex:t" data-semantic-level-number="1" data-speech-node="true" data-semantic-speech-none="t" data-semantic-prefix-none="Base" data-semantic-summary-none="identifier" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;t&lt;/say-as&gt;" data-semantic-prefix="&lt;mark name=&quot;0&quot;/&gt; Base" data-semantic-summary="&lt;mark name=&quot;0&quot;/&gt; identifier" data-semantic-braille="⠞"><mjx-c class="mjx-c1D461">𝑡</mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em;"><mjx-mi size="s" data-latex="h" data-semantic-type="identifier" data-semantic-role="simple function" data-semantic-font="italic" data-semantic-annotation="clearspeak:simple;depth:2" data-semantic-id="1" data-semantic-parent="2" data-semantic-attributes="latex:h" data-semantic-level-number="1" data-speech-node="true" data-semantic-speech-none="h" data-semantic-prefix-none="Subscript" data-semantic-summary-none="identifier" data-semantic-speech="&lt;mark name=&quot;1&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;h&lt;/say-as&gt;" data-semantic-prefix="&lt;mark name=&quot;1&quot;/&gt; Subscript" data-semantic-summary="&lt;mark name=&quot;1&quot;/&gt; identifier" data-semantic-braille="⠓"><mjx-c class="mjx-c210E">ℎ</mjx-c></mjx-mi></mjx-script></mjx-msub></mjx-math><mjx-speech aria-label="t sub h, math" role="img" aria-roledescription="" aria-braillelabel="⠞⠰⠓" aria-brailleroledescription="⠀"></mjx-speech></mjx-container></span>: The lower and upper boundaries of year <span class="math notranslate nohighlight"><mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" overflow="overflow" style="font-size: 119.5%;" tabindex="0" ctxtmenu_counter="45" data-semantic-locale="en" data-semantic-domain="clearspeak" data-semantic-style="default" data-semantic-domain2style="mathspeak:default,clearspeak:default" data-semantic-collapsible="collapsible" data-semantic-expandable="expandable" data-semantic-level="Level" data-semantic-speech-none="t" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;t&lt;/say-as&gt;" data-speech-attached="true" data-semantic-braille="⠞" data-braille-attached="true" has-speech="true"><mjx-math data-latex="t" data-semantic-structure="0" class="NCM-N" aria-hidden="true"><mjx-mi data-latex="t" data-semantic-type="identifier" data-semantic-role="latinletter" data-semantic-font="italic" data-semantic-annotation="clearspeak:simple;nemeth:number;depth:1" data-semantic-id="0" data-semantic-attributes="latex:t" data-semantic-level-number="0" data-speech-node="true" data-semantic-speech-none="t" data-semantic-summary-none="identifier" data-semantic-speech="&lt;mark name=&quot;0&quot;/&gt; &lt;say-as interpret-as=&quot;character&quot;&gt;t&lt;/say-as&gt;" data-semantic-summary="&lt;mark name=&quot;0&quot;/&gt; identifier" data-semantic-braille="⠞"><mjx-c class="mjx-c1D461">𝑡</mjx-c></mjx-mi></mjx-math><mjx-speech aria-label="t, math" role="img" aria-roledescription="" aria-braillelabel="⠞" aria-brailleroledescription="⠀"></mjx-speech></mjx-container></span>.</p></li>
    </ul>
    """)
    return


@app.cell
def _(disaster_data, pm, years):
    with pm.Model() as disaster_model:
        switchpoint = pm.DiscreteUniform("switchpoint", lower=years.min(), upper=years.max())
        # priors
        early_rate = pm.Exponential("early_rate", 1.0)
        late_rate = pm.Exponential("late_rate", 1.0)
        # Alocate appropriate Poisson rates to years before and after current
        rate = pm.math.switch(switchpoint >= years, early_rate, late_rate)

        disasters = pm.Poisson("disasters", rate, observed=disaster_data)

        disaster_idata = pm.sample(10000)
    return (disaster_idata,)


@app.cell
def _(az, disaster_idata, plt):
    _axes_arr = az.plot_trace(disaster_idata)
    for _ax in _axes_arr.flatten():
        if _ax.get_title() == "switchpoint":
            _labels = [_label.get_text() for _label in _ax.get_xticklabels()]
            _ax.set_xticklabels(_labels, rotation=45, ha="right")
            break
    plt.gca()
    return


@app.cell
def _(az, disaster_data, disaster_idata, np, plt, years):
    plt.figure(figsize=(10, 8))
    plt.plot(years, disaster_data, ".", alpha=0.6)
    plt.ylabel("Number of accidents", fontsize=16)
    plt.xlabel("Year", fontsize=16)

    trace = disaster_idata.posterior.stack(draws=("chain", "draw"))

    plt.vlines(trace["switchpoint"].mean(), disaster_data.min(), disaster_data.max(), color="C1")
    average_disasters = np.zeros_like(disaster_data, dtype="float")
    for i, year in enumerate(years):
        idx = year < trace["switchpoint"]
        average_disasters[i] = np.mean(np.where(idx, trace["early_rate"], trace["late_rate"]))

    sp_hpd = az.hdi(disaster_idata, var_names=["switchpoint"])["switchpoint"].values
    plt.fill_betweenx(
        y=[disaster_data.min(), disaster_data.max()],
        x1=sp_hpd[0],
        x2=sp_hpd[1],
        alpha=0.5,
        color="C1",
    )
    plt.plot(years, average_disasters, "k--", lw=2)
    return


if __name__ == "__main__":
    app.run()
