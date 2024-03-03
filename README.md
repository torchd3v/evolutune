# Evolutune
A Genetic Algorithm-based hyperparameter tuner for machine learning models.

## Introduction
Evolutune, implements a hyperparameter tuner based on the principles of a genetic algorithm. The genetic algorithm evolves a population of hyperparameter sets over several generations, aiming to find the set that optimizes a given scoring metric. This tuner is designed to work with various machine learning models.

## Dependencies
Make sure you have the following dependencies installed:

- ```numpy```
- ```joblib.Parallel``` and ```joblib.delayed```
- ```sklearn.metrics.get_scorer```

## Installation
```sh
pip install evolutune
```

## Usage
```python3
from evolutune import GeneticTuner

# Define your machine learning model
# model = ...

# Define the hyperparameter search space
param_grid = {
    'param1': [value1, value2, ...],
    'param2': [value3, value4, ...],
    # Add more hyperparameters as needed
}

# Define the scoring metric to optimize
scoring_metric = 'accuracy'  # Replace with your preferred metric

# Instantiate the GeneticTuner
genetic_tuner = GeneticTuner(
    model=model,
    param_grid=param_grid,
    scoring=scoring_metric,
    population_size=10,
    generations=100,
    mutation_rate=0.1,
    random_state=None,
    n_jobs=None
)
```

## Fitting the Tuner
```python3
# Define your training and evaluation sets
train_set = [X_train, y_train]
eval_set = [X_eval, y_eval]  # Set to None to use the training set for evaluation

# Specify the optimization direction ('maximize' or 'minimize')
direction = 'maximize'

# Fit the tuner on the training set
genetic_tuner.fit(train_set, eval_set, direction)
```

## Accessing Results
```python3
# Access the best score and corresponding hyperparameters
best_score = genetic_tuner.best_score_
best_params = genetic_tuner.best_params_

print(f"Best Score: {best_score}")
print("Best Hyperparameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")
```

## Methods

| Method                                                                          | Description                                                           |
|---------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| `initialize_population(population_size: int) -> list`                           | Initialize a population of individuals with random hyperparameters.   |
| `crossover(parent1: dict, parent2: dict) -> tuple`                              | Perform crossover between two parents to generate two children.       |
| `mutate(individual: dict, mutation_rate: float) -> dict`                        | Introduce random mutations to an individual's hyperparameters.        |
| `calculate_fitness(train_set: list, eval_set: list, parameters: dict) -> float` | Evaluate the fitness (scoring metric) of a set of hyperparameters.    |
| `fit(train_set: list, eval_set: list = None, direction: str = "maximize")`      | Fit the GeneticTuner on the training set and optional evaluation set. |


## Example
An example script demonstrating the usage of the GeneticTuner class is provided in the example.py file.