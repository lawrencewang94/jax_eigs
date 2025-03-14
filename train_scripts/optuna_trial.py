import optuna

# # Define the objective function (for reference, not used in `ask()` step)
# def objective(trial):
#     x = trial.suggest_float("x", -10, 10)
#     return (x - 2) ** 2  # A simple quadratic function

# Create a study
study = optuna.create_study(direction="minimize")

# Get the next suggested point **without running the trial**
trial = study.ask()  # Get the next point

# Extract the suggested values
next_params = {"x": trial.suggest_float("x", -10, 10)}
print("Next hyperparameter to try:", next_params)

# After evaluating the objective, manually record the result
study.tell(trial, (next_params["x"] - 2) ** 2)

# Repeat `ask()` and `tell()` for more manual control
