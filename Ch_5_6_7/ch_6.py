# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.0.0",
#     "arviz==0.23.4",
#     "hvplot==0.12.2",
#     "marimo>=0.22.4",
#     "matplotlib==3.10.8",
#     "numpy==2.4.4",
#     "polars==1.39.3",
#     "pyarrow==23.0.1",
#     "pymc==5.28.3",
#     "scipy==1.17.1",
#     "seaborn==0.13.2",
#     "wigglystuff==0.3.1",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    import altair as alt
    import arviz as az
    import preliz as pz
    import hvplot.polars
    import seaborn as sns
    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import polars as pl
    import pymc as pm
    import scipy.stats as stats
    from pathlib import Path
    from wigglystuff import EdgeDraw

    from pgmpy.base import DAG
    from pgmpy.inference import CausalInference
    from pgmpy.identification import Adjustment
    import networkx as nx

    from linear_models_funcs import (
        remove_period_col_name,
        cols_to_lowercase,
        std_cols_of_interest,
        std_log_mass,
        set_dtypes_float64,
        plot_simple_regression_on_chosen_scale,
        plot_linear_regression_prior_predictive,
        plot_counterfactual,
        run_linear_model,
    )

    RANDOM_SEED = 1523
    rng = np.random.default_rng(RANDOM_SEED)
    return (
        Adjustment,
        CausalInference,
        DAG,
        Path,
        alt,
        az,
        mo,
        np,
        nx,
        pl,
        plt,
        pm,
        pz,
        remove_period_col_name,
        rng,
        run_linear_model,
        set_dtypes_float64,
        sns,
        std_cols_of_interest,
        std_log_mass,
    )


@app.cell
def _(alt, az, plt):
    alt.theme.enable('fivethirtyeight')
    plt.style.use("fivethirtyeight")

    # Set default figure size to 14 inches wide by 5 inches tall
    plt.rcParams["figure.figsize"] = (14, 5)
    # You can also set the DPI (dots per inch) for crisper images
    plt.rcParams["figure.dpi"] = 100
    # Make the layout "tight" by default so labels don't overlap
    plt.rcParams["figure.autolayout"] = True
    # sets default credible interval used by arviz
    az.rcParams["stats.ci_prob"] = 0.89
    return


@app.cell
def _(nx, plt):
    def draw_dag(dag: "DAG"):
        # pgmpy's DAG *is* a networkx DiGraph, so no need to extract a sub-attribute
        _nx_graph = dag

        # Try to calculate a topological layout (layered by "generations")
        # This ensures a top-to-bottom flow which is standard for DAGs
        for _layer, _nodes in enumerate(nx.topological_generations(_nx_graph)):
            for _node in _nodes:
                _nx_graph.nodes[_node]["layer"] = _layer

        _pos = nx.multipartite_layout(_nx_graph, subset_key="layer")

        # Create the figure
        _fig, _ax = plt.subplots(figsize=(10, 6))

        nx.draw(
            _nx_graph,
            pos=_pos,
            with_labels=True,
            node_color="#4C72B0",
            node_size=1000,
            font_color="white",
            font_size=14,
            font_weight="bold",
            arrowsize=25,
            edge_color="#666666",
            width=2,
            connectionstyle="arc3,rad=0.1", # Slightly curved arrows look cleaner
            ax=_ax
        )

        _ax.set_title("Causal Graphical Model", fontsize=16, pad=20)

        return plt.gca()

    return (draw_dag,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Raw Data
    """)
    return


@app.cell
def _(Path, pl):
    #############
    # Constants #
    #############
    MILK_PATH = Path(__file__).parent.parent / "data" / "milk.csv"

    raw_milk_data = pl.read_csv(MILK_PATH, separator=";")
    return (raw_milk_data,)


@app.cell
def _(
    raw_milk_data,
    remove_period_col_name,
    set_dtypes_float64,
    std_cols_of_interest,
    std_log_mass,
):
    # fmt: off
    milk_data = (
        raw_milk_data
        .pipe(remove_period_col_name)
        .pipe(set_dtypes_float64, ["kcal_per_g", "neocortex_perc"])
        .pipe(std_cols_of_interest, ["kcal_per_g", "neocortex_perc", "perc_fat", "perc_lactose"])
        .pipe(std_log_mass)
    )
    # fmt: on
    return (milk_data,)


@app.cell
def _(milk_data):
    milk_data
    return


@app.cell
def _(alt, milk_data, mo):
    _chart = (
        alt.Chart(milk_data)
        .mark_circle(size=100, opacity=0.7)
        .encode(
            x=alt.X(
                "perc_lactose:Q",
                title="Percentage Lactose",
                axis=alt.Axis(labelFontSize=12, titleFontSize=14, gridOpacity=0.3),
            ),
            y=alt.Y(
                "perc_fat:Q",
                title="Percentage Fat",
                axis=alt.Axis(labelFontSize=12, titleFontSize=14, gridOpacity=0.3),
            ),
            color=alt.Color(
                "clade:N",
                title="Clade",
                scale=alt.Scale(scheme="tableau10"),
                legend=alt.Legend(
                    titleFontSize=13, labelFontSize=11, symbolSize=80, orient="right"
                ),
            ),
        )
        .properties(
            width="container",
            height=400,
            title={
                "text": "Relationship between Lactose and Fat by Clade",
                "fontSize": 16,
                "fontWeight": "bold",
                "subtitle": "Each point represents a mammal species",
                "subtitleFontSize": 12,
                "subtitleColor": "#666666",
                "offset": 10,
            },
        )
        .configure_view(strokeWidth=0)
        .interactive()
    )

    mo.ui.altair_chart(_chart)
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    # Chater Notes
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multicollinearity
    """)
    return


@app.cell
def _(milk_data, sns):
    sns.pairplot(
        data=milk_data.select(
            ["perc_fat_std", "perc_lactose_std", "kcal_per_g_std", "clade"]
        ).to_pandas(),
        # hue="clade",
        diag_kind="kde",
        kind="reg",
        height=2,
        aspect=1.7,
    )
    return


@app.cell
def _(milk_data, run_linear_model):
    fat_model = run_linear_model(
        predictors=[milk_data["perc_fat_std"]],
        predictors_names=["perc_fat_std"],
        outcome=milk_data["kcal_per_g_std"],
        outcome_name="kcal_per_g_std",
        prior_predictive=False,
        draws=1000,
        alpha=0.2,
        beta=0.5,
    )

    lactose_model = run_linear_model(
        predictors=[milk_data["perc_lactose_std"]],
        predictors_names=["perc_lactose_std"],
        outcome=milk_data["kcal_per_g_std"],
        outcome_name="kcal_per_g_std",
        prior_predictive=False,
        draws=1000,
        alpha=0.2,
        beta=0.5,
    )

    lactose_fat_model = run_linear_model(
        predictors=[milk_data["perc_fat_std"], milk_data["perc_lactose_std"]],
        predictors_names=["perc_fat_std", "perc_lactose_std"],
        outcome=milk_data["kcal_per_g_std"],
        outcome_name="kcal_per_g_std",
        prior_predictive=False,
        draws=1000,
        alpha=0.2,
        beta=0.5,
    )
    return fat_model, lactose_fat_model, lactose_model


@app.cell
def _(az, fat_model, lactose_fat_model, lactose_model):
    # az.summary(fat_model, kind="stats"), az.summary(lactose_model, kind="stats"), az.summary(lactose_fat_model, kind="stats")

    az.plot_forest(
        {'fat': fat_model, 'lactose': lactose_model, 'both': lactose_fat_model},
        var_names=["beta"],
        combined=True,
        figure_kwargs={'figsize':(10, 4)},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The posterior distributions for perc.fat and perc.lactose are essentially mirror images of one another. The posterior mean of perc.fat is as positive as the mean of perc.lactose is negative. Both are narrow posterior distributions that lie almost entirely on one side or the other of zero. Given the strong association of each predictor with the outcome, we might conclude that both variables are reliable predictors of total energy in milk, across species. The more fat, the more kilocalories in the milk. The more lactose, the fewer kilocalories in milk.

    Now the posterior means of both perc.fat and perc.lactose are closer to zero. And the standard deviations for both parameters are twice as large as in the bivariate models. This is the same statistical phenomenon as in the leg length example. What has happened is that the variables perc.fat and perc.lactose contain much of the same information. They are almost substitutes for one another. As a result, when you include both in a regression, the posterior distribution ends up describing a long ridge of combinations of perc.fat and perc.lactose that are equally plausible.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Post-treatment Bias
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fungus example page 175
    """)
    return


@app.cell
def _(np, pl, rng):
    _n = 100
    _h0 = rng.normal(10, 2, size=_n)
    _treatment = np.repeat([0, 1], _n / 2)
    _fungus = rng.binomial(n=1, p=0.5 - _treatment * 0.4, size=_n)
    _h1 = _h0 + rng.normal(5 - 3 * _fungus, size=_n)

    _d = pl.DataFrame({"h0": _h0, "h1": _h1, "treatment": _treatment, "funfus": _fungus})

    _d
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We know that the plants at time $t = 1$ should be taller than at time $t = 0$, whatever scale they are measured on. So if we put the parameters on a scale of proportion of height at time $t = 0$, rather than on the absolute scale of the data, we can set the priors more easily. To make this simpler, let’s focus right now only on the height variables, ignoring the predictor variables. We might have a linear model like:

    $$ h_{1,i} \sim \text{Normal}(\mu_i, \sigma) $$
    $$ \mu_i = h_{0,i} \times p $$

    where h0,i is plant i’s height at time $t = 0$, h1,i is its height at time $t = 1$, and p is a parameter measuring the proportion of h0,i that h1,i is. More precisely, $p=\frac{h_{1,i}}{h_{0,i}}$ . If p = 1, the plant hasn’t changed at all from time $t = 0$ to time $t = 1$. If p = 2, it has doubled in height. So if we center our prior for p on 1, that implies an expectation of no change in height. That is less than we know. But we should allow p to be less than 1, in case the experiment goes horribly wrong and we kill all the plants. We also have to ensure that p > 0, because it is a proportion. A Log-Normal distribution, because it is always positive. If we use p ∼ Log-Normal(0, 0.25) (draw garph to see what it looks like).
    """)
    return


@app.cell
def _(az, pz, rng):
    (
        az.summary(rng.lognormal(0, 0.25, (1, 1000)), kind="stats"),
        pz.LogNormal(mu=0, sigma=0.25).plot_pdf(pointinterval=True, levels=[0.91])
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So this prior expects anything from 40% shrinkage up to 40% growth. Now to include the treatment and fungus variables. We’ll include both of them, following the notion that we’d like to measure the impact of both the treatment and the fungus itself. The parameters for these variables will also be on the pro- portion scale. They will be changes in proportion growth. So we’re going to make a linear model of p:

    \begin{aligned}
    h_{1,i} &\sim \mathcal{N}(\mu_i,\sigma)\\
    \mu_i &= h_{0,i} \times p\\
    p &= \alpha + \beta_T T_i + \beta_F F_i\\
    \alpha &\sim \mathrm{Log\text{-}Normal}(0,0.25)\\
    \beta_T &\sim \mathcal{N}(0,0.5)\\
    \beta_F &\sim \mathcal{N}(0,0.5)\\
    \sigma &\sim \mathrm{Exponential}(1)
    \end{aligned}

    The proportion of growth p is now a function of the predictor variables. The priors on the slopes are almost certainly too flat. They place 95% of the prior mass between −1 (100% reduction) and +1 (100% increase) and two-thirds of the prior mass between −0.5 and +0.5.
    """)
    return


@app.cell
def _(az, np, pm, rng):
    def run_fungus_model() -> az.InferenceData:
        n = 1000
        h0 = rng.normal(10, 2, size=n)
        treatment = np.repeat([0, 1], n / 2)
        fungus = rng.binomial(n=1, p=0.5 - treatment * 0.4, size=n)
        h1 = h0 + rng.normal(5 - 3 * fungus, size=n)

        with pm.Model() as model:
            a = pm.Normal("a", 0, 0.2)
            bt = pm.Normal("treatment", 0, 0.5)
            bf = pm.Normal("fungus", 0, 0.5)
            sigma = pm.Exponential("sigma", 1)

            p = a + bt*treatment + bf*fungus
            mu = h0*p

            h1_inference = pm.Normal("h1", mu, sigma, observed=h1)

            idata = pm.sample()

        return idata

    return (run_fungus_model,)


@app.cell
def _(run_fungus_model):
    fungus_model_wrong = run_fungus_model()
    return (fungus_model_wrong,)


@app.cell
def _(az, fungus_model_wrong):
    az.summary(fungus_model_wrong, kind="stats")
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


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercises
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6H1

    Use the Waffle House data, data(WaffleDivorce), to find the total causal influence of number of Waffle Houses on divorce rate. Justify your model or models with a causal graph
    """)
    return


app._unparsable_cell(
    r"""
    WAFFLE_PATH = Path(__file__).parent.parent / "data" / "WaffleDivorce.csv"

    raw_waffle_data = pl.read_csv(WAFFLE_PATH, separator=";")

    def select_cols(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
        return df.select(cols)

    def create_south_cat_col():
    

    cols_to_keep_waffle = [
      # "location",
      # "loc",
      # "population",
      "medianAgeMarriage",
      "marriage",
      # "marriage SE",
      "divorce",
      # "divorce SE",
      "waffleHouses",
      "south",
      # "slaves1860",
      # "population1860",
      # "propSlaves1860",
      "divorce_std",
      "medianAgeMarriage_std",
      "waffleHouses_std",
      "marriage_std"
    ]

    waffle_data = (
        raw_waffle_data
        .pipe(cols_to_lowercase)
        .pipe(std_cols_of_interest, ["divorce", "medianAgeMarriage", "marriage", "waffleHouses"])
        .pipe(select_cols, cols_to_keep_waffle)
    )

    # region_cat = ['not_south','south']
    # region_idx = waffle_data['south'].cast(pl.Enum(region)).to_physical().to_numpy()
    WAFFLE_OUTCOME = waffle_data["divorce_std"].to_numpy()
    """,
    name="_"
)


@app.cell
def _(region, region_idx):
    region, region_idx
    return


@app.cell
def _(DAG):
    dag_6h1 = DAG(
        [
            ("A", "D"),
            ("A", "M"),
            ("M", "D"),
            ("S", "A"),
            ("S", "M"),
            ("S", "W"),
            ("W", "D"),
        ],
        roles={"exposures": "W", "outcomes": "D"}
    )
    return (dag_6h1,)


@app.cell
def _(dag_6h1, draw_dag):
    draw_dag(dag_6h1)
    return


@app.cell
def _(Adjustment, CausalInference, dag_6h1):
    # easier, old version
    inference = CausalInference(dag_6h1)

    # new way, seems to be giving different answers. This shows more results than the above
    identified_all, ok_all = Adjustment(variant="all").identify(dag_6h1)

    # variant="all" returns a list of graphs (one per valid adjustment set),
    # so we need to extract the "adjustment" role from each graph individually
    if isinstance(identified_all, list):
        all_adjustment_sets = [_graph.get_role("adjustment") for _graph in identified_all]
    else:
        all_adjustment_sets = identified_all.get_role("adjustment")

    # Adjustment sets for exposure=W, outcome=D, Implied conditional independencies
    # all_adjustment_sets, inference.get_all_backdoor_adjustment_sets("W", "D"), dag_6h1.get_independencies()

    inference.get_all_backdoor_adjustment_sets("W", "D"), dag_6h1.get_independencies()
    return


@app.cell
def _(np, pm, rng, waffle_data):
    with pm.Model(coords={'obs': np.arange(waffle_data.shape[0]), 'region': ['not_south','south']}) as model_region:
        # Data
        _south= pm.Data("south", waffle_data["south"].to_numpy(), dims="obs")
        _w = pm.Data('W', waffle_data['waffleHouses_std'].to_numpy(), dims="obs")
        # priors
        _α = pm.Normal('α', 0, 0.5, dims='region')
        _β = pm.Normal('β', 0, 0.8, dims='region')
        _σ = pm.Exponential('σ', 1)
        # Mean
        _μ = pm.Deterministic('μ', _α[_south] + _β[_south] * _w, dims="obs")
        # Likelihood
        _div = pm.Normal('D', mu=_μ, sigma=_σ, observed=waffle_data['divorce_std'], dims="obs")

        region_idata = pm.sample(1000, random_seed=rng)
        pm.sample_posterior_predictive(region_idata, extend_inferencedata=True, random_seed=rng)

    with pm.Model(coords={'obs': np.arange(waffle_data.shape[0])}) as model_additive:
        # Data
        _south= pm.Data("south", waffle_data["south"].to_numpy(), dims="obs")
        _w = pm.Data('W', waffle_data['waffleHouses_std'].to_numpy(), dims="obs")
        # priors
        _α = pm.Normal('α', 0, 0.5)
        _β_w = pm.Normal('β_w', 0, 0.5)
        _β_s = pm.Normal('β_s', 0, 0.5)
        _σ = pm.Exponential('σ', 1)
        # Mean
        _μ = pm.Deterministic('μ', _α + _β_w*_w + _β_s*_south, dims="obs")
        # Likelihood
        _div = pm.Normal('D', mu=_μ, sigma=_σ, observed=waffle_data['divorce_std'], dims="obs")

        region_idata_add = pm.sample(1000, random_seed=rng)
        pm.sample_posterior_predictive(region_idata_add, extend_inferencedata=True, random_seed=rng)
    return region_idata, region_idata_add


@app.cell
def _(az, region_idata, region_idata_add):
    az.summary(region_idata, var_names=['~μ'], kind='stats'), az.summary(region_idata_add, var_names=['~μ'], kind='stats')
    return


@app.cell
def _(az, region_idata):
    az.plot_trace_dist(region_idata, var_names=['~μ'], figure_kwargs={'figsize':(10, 6)})
    return


@app.cell
def _(az, region_idata, region_idata_add):
    az.plot_forest(
        {'additive (shared slope)': region_idata_add, "region-stratified": region_idata},
        var_names=["β_w", "β"],
        combined=True,
        # hdi_prob=0.94,
        figure_kwargs={'figsize':(10, 4)},
    )
    return


@app.cell
def _(az, region_idata_add):
    az.plot_dist(region_idata_add, var_names=['β_w'])
    return


@app.cell
def _(az, region_idata):
    az.plot_dist(region_idata, var_names=['β'], col_wrap=1)
    return


if __name__ == "__main__":
    app.run()
