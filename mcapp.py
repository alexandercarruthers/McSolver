import gradio as gr
import cvxpy as cp
import pandas as pd
import numpy as np


def load_data():
    df = pd.read_csv("mcdata/mcdonalds_dataset_uk.csv")
    df = df.dropna()
    df["Calories"] = df["product_calories"].str.replace("kcal: ", "").astype(int)
    return df


def solve(
    prefer_salt_slider,
    prefer_fat_slider,
    prefer_carbs_slider,
    prefer_protein_slider,
    prefer_sugar_slider,
):
    df = load_data()
    foods = df["product_name"].to_numpy()
    calories = df["Calories"].to_numpy()
    salt_g = df["Salt"].to_numpy()
    fat_g = df["Fat"].to_numpy()
    carbs_g = df["Carbs"].to_numpy()
    protein_g = df["Protein"].to_numpy()
    sugar_g = df["Sugars"].to_numpy()

    n_foods = len(foods)

    max_calories = 2500

    x = cp.Variable(n_foods, boolean=True)

    constraints = [
        x @ calories <= max_calories,  # Maximum caloric intake
    ]

    salt_normalised = salt_g / np.max(salt_g)
    fat_g_normalised = fat_g / np.max(fat_g)
    carbs_g_normalised = carbs_g / np.max(carbs_g)
    protein_g_normalised = protein_g / np.max(protein_g)
    sugar_g_normalised = sugar_g / np.max(sugar_g)

    salt_expr = x @ salt_normalised
    fat_expr = x @ fat_g_normalised
    carbs_expr = x @ carbs_g_normalised
    protein_expr = x @ protein_g_normalised
    sugar_expr = x @ sugar_g_normalised

    combined_objective = cp.Maximize(
        prefer_salt_slider * salt_expr
        + prefer_fat_slider * fat_expr
        + prefer_carbs_slider * carbs_expr
        + prefer_protein_slider * protein_expr
        + prefer_sugar_slider * sugar_expr
    )

    problem = cp.Problem(combined_objective, constraints)

    problem.solve(verbose=False)

    x_value = np.round(x.value).astype(int)

    if problem.status == cp.OPTIMAL:
        output_items = []
        output_calories = []
        output_salt = []
        output_fat = []
        output_carbs = []
        output_protein = []
        output_sugar = []
        for i in range(len(foods)):
            if x_value[i] > 0:
                output_items.append(foods[i])
                output_calories.append(calories[i])
                output_salt.append(salt_g[i])
                output_fat.append(fat_g[i])
                output_carbs.append(carbs_g[i])
                output_protein.append(protein_g[i])
                output_sugar.append(sugar_g[i])

        output_df = pd.DataFrame(
            data={
                "Item": output_items,
                "Calories (kcal)": output_calories,
                "Salt (g)": output_salt,
                "Fat (g)": output_fat,
                "Carbs (g)": output_carbs,
                "Protein (g)": output_protein,
                "Sugar (g)": output_sugar,
            },
        )
        output_df = output_df.sort_values(
            by="Calories (kcal)", inplace=False, ascending=False
        )
        totals_df = pd.DataFrame(
            data={
                "Totals": [max([len(item) for item in output_items]) * "_"],
                "Calories (kcal)": sum(output_calories),
                "Salt (g)": round(sum(output_salt), 2),
                "Fat (g)": round(sum(output_fat), 1),
                "Carbs (g)": round(sum(output_carbs)),
                "Protein (g)": round(sum(output_protein), 2),
                "Sugar (g)": round(sum(output_sugar), 1),
            }
        )
        return output_df, totals_df

    else:
        print("No optimal solution found.")


with gr.Blocks() as demo:
    gr.Markdown(
        """
    # McSolver - I'm Solvin' It
    
    ![Ronald Mcdonald sitting on a bench](file/ronald.png)
    
    ## Welcome to McSolver, the ultimate tool for balancing macronutrients subject to a daily calorie intake at McDonald's. 
    
    We aim to maximize the total calories consumed
    
    $$
    \\text{maximize} \\quad \\sum_{i=1}^{n} \\text{calories}_i \\cdot x_i
    $$
    
    Where we do not exceed the 2500 kcal recommended calorie intake for males between 19 - 64.

    $$
    \\text{subject to} \\quad \\sum_{i=1}^{n} \\text{calories}_i \\cdot x_i \\leq 2500
    $$
    
    And we have a boolean decision variable indicating whether we should order that menu item.     

    $$
    x_i \\in \\{0, 1\\} \\quad \\forall i \\in \\{1, 2, \\ldots, n\\}
    $$
    
    Our objective function is defined as:
    
    where all normalisations are:
    
    $$
    \\text{Salt Normalised} = \\frac{\\text{salt}}{\\max(\\text{salt})}
    $$
    
    $$
    \\text{Objective} = \\text{maximize } 
    \\left( 
    \\left(\\text{PreferSalt} \\cdot SaltNormalised_i\\right) 
    +
    \\left(\\text{PreferFat} \\cdot FatNormalised_i\\right)
    +
    \\left(\\text{PreferCarbs} \\cdot CarbsNormalised_i\\right)
    +
    \\left(\\text{PreferProtein} \\cdot ProteinNormalised_i\\right)
    +
    \\left(\\text{PreferSugar} \\cdot SugarNormalised_i\\right)
    \\right)
    $$
    

    
    ## Use the sliders to adjust your preferences.
    """
    )
    with gr.Row():
        prefer_salt_slider = gr.Slider(
            minimum=0.001, maximum=1, value=0.5, label="Prefer Salt"
        )
        prefer_fat_slider = gr.Slider(
            minimum=0.001, maximum=1, value=0.5, label="Prefer Fat"
        )
        prefer_carbs_slider = gr.Slider(
            minimum=0.001, maximum=1, value=0.5, label="Prefer Carbs"
        )
        prefer_protein_slider = gr.Slider(
            minimum=0.001, maximum=1, value=0.5, label="Prefer Protein"
        )
        prefer_sugar_slider = gr.Slider(
            minimum=0.001, maximum=1, value=0.5, label="Prefer Sugar"
        )

    meal_plan_textbox = gr.Dataframe(
        headers=[
            "Item",
            "Calories (kcal)",
            "Salt (g)",
            "Fat (g)",
            "Carbs (g)",
            "Protein (g)",
            "Sugar (g)",
        ]
    )
    summary_dataframe = gr.Dataframe(
        headers=[
            "Total Items",
            "Total Calories (kcal)",
            "Total Salt (g)",
            "Total Fat (g)",
            "Total Carbs (g)",
            "Total Protein (g)",
            "Total Sugar (g)",
        ]
    )
    slider_inputs = [
        prefer_salt_slider,
        prefer_fat_slider,
        prefer_carbs_slider,
        prefer_protein_slider,
        prefer_sugar_slider,
    ]

    prefer_salt_slider.release(
        fn=solve, inputs=slider_inputs, outputs=[meal_plan_textbox, summary_dataframe]
    )
    prefer_fat_slider.release(
        fn=solve, inputs=slider_inputs, outputs=[meal_plan_textbox, summary_dataframe]
    )
    prefer_carbs_slider.release(
        fn=solve, inputs=slider_inputs, outputs=[meal_plan_textbox, summary_dataframe]
    )
    prefer_protein_slider.release(
        fn=solve, inputs=slider_inputs, outputs=[meal_plan_textbox, summary_dataframe]
    )
    prefer_sugar_slider.release(
        fn=solve, inputs=slider_inputs, outputs=[meal_plan_textbox, summary_dataframe]
    )

    gr.Markdown(
        """
    ## References    
    - Steven Diamond and Stephen Boyd. [CVXPY](https://github.com/cvxpy/cvxpy): A Python-embedded modeling language for convex optimization. Journal of Machine Learning Research, 17(83):1-5, 2016.
    - [Government recommendations for energy and nutrients for males and females aged 1 – 18 years and 19+ years.](https://assets.publishing.service.gov.uk/media/5a749fece5274a44083b82d8/government_dietary_recommendations.pdf)
    - [McDonald's UK Menu Dataset](https://www.kaggle.com/datasets/danilchurko/mcdonalds-uk-menu-dataset)
    """
    )

if __name__ == "__main__":
    demo.launch(allowed_paths=["/"])
