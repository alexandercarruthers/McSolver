# McSolver - I'm Solvin' It

![Ronald Mcdonald sitting on a bench](ronald.png)

## Welcome to McSolver, the ultimate tool for balancing macronutrients subject to a daily calorie intake at McDonald's. 

McSolver aims to solve the optimisation problem that is balancing macronutrients at McDonald's whilst staying
within the recommended daily calorie intake of 2500 kcal.

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

![](mc_readme_screenshot.png)

## How to Run

1. Install the dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
gradio mcapp.py
```
