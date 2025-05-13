# Evolutune
A hyperparameter tuning library for machine learning models based on evolutionary algorithms.

## Introduction
Evolutune implements hyperparameter tuners based on principles of evolutionary optimization algorithms. Currently, it offers two tuners:

1. **GeneticTuner**: Uses genetic algorithm principles to evolve a population of hyperparameter sets over several generations.
2. **PSOTuner**: Uses Particle Swarm Optimization to find optimal hyperparameters by simulating the movement of particles in the search space.

Both tuners are designed to work with various machine learning models to find hyperparameter sets that optimize a given scoring metric.

## Dependencies
Make sure you have the following dependencies installed:

- `numpy`
- `joblib`
- `scikit-learn`

## Installation
```sh
pip install evolutune
```

## GeneticTuner Usage
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
    cv=None,
    n_jobs=None
)
```

## PSOTuner Usage
```python3
from evolutune import PSOTuner

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

# Instantiate the PSOTuner
pso_tuner = PSOTuner(
    model=model,
    param_grid=param_grid,
    scoring=scoring_metric,
    n_particles=10,
    iterations=100,
    w=0.5,      # Inertia weight
    c1=1.5,     # Cognitive coefficient
    c2=1.5,     # Social coefficient
    cv=None,
    random_state=None,
    n_jobs=None
)
```

## Fitting the Tuners
```python3
# Define your training and evaluation sets
train_set = [X_train, y_train]
eval_set = [X_eval, y_eval]  # Set to None to use the training set for evaluation

# Specify the optimization direction ('maximize' or 'minimize')
direction = 'maximize'

# Fit the tuner on the training set
tuner.fit(train_set, eval_set, direction)
```

## Accessing Results
```python3
# Access the best score and corresponding hyperparameters
best_score = tuner.best_score_
best_params = tuner.best_params_

print(f"Best Score: {best_score}")
print("Best Hyperparameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")
```

## GeneticTuner Methods

| Method                                                                          | Description                                                           |
|---------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| `initialize_population(population_size: int) -> list`                           | Initialize a population of individuals with random hyperparameters.   |
| `crossover(parent1: dict, parent2: dict) -> tuple`                              | Perform crossover between two parents to generate two children.       |
| `mutate(individual: dict, mutation_rate: float) -> dict`                        | Introduce random mutations to an individual's hyperparameters.        |
| `calculate_fitness(train_set: list, eval_set: list, parameters: dict) -> float` | Evaluate the fitness (scoring metric) of a set of hyperparameters.    |
| `fit(train_set: list, eval_set: list = None, direction: str = "maximize")`      | Fit the GeneticTuner on the training set and optional evaluation set. |

## PSOTuner Methods

| Method                                                                          | Description                                                            |
|---------------------------------------------------------------------------------|------------------------------------------------------------------------|
| `initialize_swarm(n_particles: int) -> tuple`                                   | Initialize a swarm of particles with random parameters and velocities. |
| `update_velocity(position: dict, velocity: dict, personal_best: dict, global_best: dict) -> dict` | Update the velocity of a particle based on current position and bests. |
| `update_position(position: dict, velocity: dict) -> dict`                       | Update the position of a particle based on its velocity.               |
| `calculate_fitness(train_set: list, eval_set: list, parameters: dict) -> float` | Evaluate the fitness (scoring metric) of a set of hyperparameters.     |
| `fit(train_set: list, eval_set: list = None, direction: str = "maximize")`      | Fit the PSOTuner on the training set and optional evaluation set.      |

## Examples
Example scripts demonstrating the usage of both tuners are provided in the examples directory:

1. `examples/genetic_example.py` - Demonstrates the usage of the GeneticTuner
2. `examples/pso_example.py` - Demonstrates the usage of the PSOTuner

You can also find comparison scripts in the main directory that demonstrate how both tuners perform against each other on various datasets.