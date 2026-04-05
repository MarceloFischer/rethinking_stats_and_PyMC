import marimo

__generated_with = "0.19.4"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import math

    TOTAL_PEOPLE = 10
    TOTAL_WINS = 8
    return TOTAL_PEOPLE, TOTAL_WINS, math, mo


@app.cell(hide_code=True)
def _(TOTAL_PEOPLE, TOTAL_WINS, mo):
    mo.md(rf"""
    Out of {TOTAL_PEOPLE} people we observed {TOTAL_WINS} heads and {TOTAL_PEOPLE - TOTAL_WINS} tails.

    The sample is {", ".join(["1"] * TOTAL_WINS + ["0"] * (TOTAL_PEOPLE - TOTAL_WINS))}

    To calculate in how many different ways this could have happened if we have $d$ dishonest people:

    How many different ways can $d = {TOTAL_PEOPLE} - h$ dishonest people be arranged among the {TOTAL_WINS} observed heads:

    $$\binom{{{TOTAL_WINS}}}{{{TOTAL_PEOPLE} - d}} = \binom{{{TOTAL_WINS}}}{{h}}$$

    From the $h = {TOTAL_PEOPLE} - d$ honest people, in how many different ways can we get the {TOTAL_PEOPLE - TOTAL_WINS} tails:

    $$\binom{{{TOTAL_PEOPLE} - d}}{{{TOTAL_PEOPLE} - 2 - d}} = \binom{{h}}{{{TOTAL_WINS} - {TOTAL_PEOPLE} + h}} = \binom{{h}}{{h-{TOTAL_PEOPLE - TOTAL_WINS}}}$$
    """)
    return


@app.cell
def _(TOTAL_PEOPLE, TOTAL_WINS, math):
    def calculate_ways_that_honest_get_2_fails(h: int) -> int:
        return math.comb(h, h - 2)


    def calculate_ways_to_arrange_d_dishonest_among_8_observed_heads(
        h: int,
    ) -> int:
        return math.comb(TOTAL_WINS, TOTAL_PEOPLE - h)


    def calculate_total_ways_to_observe_sample(h: int):
        # print(f"{h} honest people | {10-h} dishosnet")
        # print(f"math.comb({h}, {h-2}) = {calculate_ways_that_honest_get_2_fails(h)}")
        # print(f"math.comb({TOTAL_WINS}, {TOTAL_PEOPLE - h}) = {calculate_ways_to_arrange_d_dishonest_among_8_observed_heads(h)}")
        total_ways = calculate_ways_that_honest_get_2_fails(
            h
        ) * calculate_ways_to_arrange_d_dishonest_among_8_observed_heads(h)
        return total_ways
    return (calculate_total_ways_to_observe_sample,)


@app.cell
def _(calculate_total_ways_to_observe_sample):
    [
        f"total ways for {h_people} honest people: {calculate_total_ways_to_observe_sample(h=h_people)}"
        for h_people in range(2, 9)
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(f"""
    The number os honest people that maximises the ways to realise the sample is 7.
    """)
    return


if __name__ == "__main__":
    app.run()
