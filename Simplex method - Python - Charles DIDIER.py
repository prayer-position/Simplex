import numpy as np
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
import pandas as pd
import seaborn as sns

# Part 1: Read CSV and Parse Data
def parse_csv(filename):
    # Define column names and read CSV
    df = pd.read_csv(filename, names=['x', 'y', 'type', 'value'])

    # Extract objective coefficients and max/min type from the first row
    objective_coeffs = [float(df.iloc[0]['x']), float(df.iloc[0]['y'])]
    maximize = df.iloc[0]['type'] == 'max'

    # Extract constraints and bounds from the remaining rows
    constraints = df.iloc[1:].apply(lambda row: [float(row['x']), float(row['y'])], axis=1).tolist()
    bounds = df.iloc[1:]['value'].astype(float).tolist()

    return objective_coeffs, constraints, bounds, maximize

# Part 2: Solve using OR-Tools
def solve_lp_with_ortools(objective_coeffs, constraints, bounds, maximize=True):
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        print("GLOP solver not available.")
        return None

    # Define variables
    variables = [solver.NumVar(0, solver.infinity(), f'x{i+1}') for i in range(len(objective_coeffs))]

    # Define objective
    objective = solver.Objective()
    for var, coeff in zip(variables, objective_coeffs):
        objective.SetCoefficient(var, coeff)
    objective.SetMaximization() if maximize else objective.SetMinimization()

    # Define constraints
    for i, (constraint_coeffs, bound) in enumerate(zip(constraints, bounds)):
        constraint = solver.RowConstraint(-solver.infinity(), bound, f'constraint_{i+1}')
        for var, coeff in zip(variables, constraint_coeffs):
            constraint.SetCoefficient(var, coeff)

    # Solve and plot if optimal solution is found
    if solver.Solve() == pywraplp.Solver.OPTIMAL:
        solution = [var.solution_value() for var in variables]
        print("Optimal solution found:", solution)
        plot_3d_graph(objective_coeffs, constraints, bounds, solution)
        return solution
    else:
        print("No optimal solution found.")
        return None

def plot_3d_graph(objective_coeffs, constraints, bounds, solution):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.linspace(0, max(bounds), 100), np.linspace(0, max(bounds), 100)
    X, Y = np.meshgrid(x, y)

    for constraint, bound in zip(constraints, bounds):
        Z = (bound - constraint[0] * X - constraint[1] * Y) if len(constraint) == 2 else np.zeros_like(X)
        ax.plot_surface(X, Y, Z, alpha=0.3, color='blue')

    ax.scatter(*solution, color="red", s=100, label="Optimal Solution")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Objective")
    plt.legend()
    plt.show()

# Part 3: Custom Simplex Method
def simplex(c, A, b):
    num_vars = len(c)
    tableau = np.zeros((len(b) + 1, len(c) + len(b) + 1))
    tableau[:-1, :num_vars] = A
    tableau[:-1, num_vars:num_vars + len(b)] = np.eye(len(b))
    tableau[:-1, -1] = b
    tableau[-1, :num_vars] = -c

    # Basic variable setup
    basic_vars = [f's{i+1}' for i in range(len(b))] + ['Objective']
    row_names = [f'Basic {var}' for var in basic_vars]
    col_names = ["Basic Variable"] + [f'x{i+1}' for i in range(num_vars)] + [f's{i+1}' for i in range(len(b))] + ['RHS']

    def display_tableau(tableau, row_names, col_names, basic_vars):
        # Prepare DataFrame for display without graphical output
        tableau_df = pd.DataFrame(tableau, index=row_names, columns=col_names[1:])
        tableau_df.insert(0, "Basic Variable", basic_vars)
        print("Current Tableau:")
        print(tableau_df)

    # Simplex algorithm
    while True:
        display_tableau(tableau, row_names, col_names, basic_vars)
        
        # Determine pivot column (most negative indicator in objective row)
        pivot_col = np.argmin(tableau[-1, :-1])
        if tableau[-1, pivot_col] >= 0:
            break

        # Determine pivot row (smallest positive ratio), excluding division by zero
        ratios = np.divide(
            tableau[:-1, -1], tableau[:-1, pivot_col], 
            out=np.full_like(tableau[:-1, -1], np.inf), 
            where=tableau[:-1, pivot_col] > 0
        )
        pivot_row = np.argmin(ratios)

        # Update basic variable name
        basic_vars[pivot_row] = col_names[pivot_col + 1]

        # Perform pivot
        tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    display_tableau(tableau, row_names, col_names, basic_vars)
    solution = tableau[:-1, -1]
    print("Optimal Solution:", solution)
    return solution


# Load data from CSV and run both methods
filename = r"system.csv"
objective_coeffs, constraints, bounds, maximize = parse_csv(filename)

print("Solution using OR-Tools:")
solve_lp_with_ortools(objective_coeffs, constraints, bounds, maximize)

print("\nSolution using Simplex Method:")
simplex(np.array(objective_coeffs), np.array(constraints), np.array(bounds))
        
