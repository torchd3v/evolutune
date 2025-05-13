from evolutune.tuner import GeneticTuner, PSOTuner
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
import time
import matplotlib.pyplot as plt
import numpy as np

def format_time(seconds):
    """Format time in a human-readable way"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds / 60)
    remaining_seconds = seconds % 60
    return f"{minutes}m {remaining_seconds:.2f}s"

# Define hyperparameter search space for Decision Tree Classifier
hyperparameter_space = {
    'criterion': ['gini', 'entropy'],  # Splitting criterion
    'splitter': ['best', 'random'],  # Strategy to choose the split at each node
    'max_depth': [None, 5, 10, 15, 20],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_features': ['sqrt', 'log2', None],  # Number of features to consider for the best split
    'max_leaf_nodes': [None, 10, 20, 30],  # Grow a tree with max_leaf_nodes in best-first fashion
    'min_impurity_decrease': [0.0, 0.1, 0.2], # A node will be split if this split induces a decrease of the impurity greater than or equal to this value
    'ccp_alpha': [0.0, 0.1, 0.2]  # Complexity parameter used for Minimal Cost-Complexity Pruning
}

print("Loading and preparing the Titanic dataset...")
# Load Titanic dataset from URL
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_df = pd.read_csv(url)

# Data preprocessing
titanic_df = titanic_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
titanic_df = titanic_df.dropna()
titanic_df = pd.get_dummies(titanic_df, columns=['Sex', 'Embarked'], drop_first=True)

# Define features (X) and target variable (y)
X = titanic_df.drop('Survived', axis=1)
y = titanic_df['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the Cross Validator
skf = StratifiedKFold(n_splits=3)

# Common parameters
common_params = {
    'model': DecisionTreeClassifier(random_state=42),
    'param_grid': hyperparameter_space,
    'scoring': "accuracy",
    'cv': skf,
    'n_jobs': -1,
    'random_state': 42
}

# Calculate total number of possible combinations for grid search
total_combinations = 1
for values in hyperparameter_space.values():
    total_combinations *= len(values)
print(f"\nTotal combinations in grid search: {total_combinations}")

# Set number of iterations for fair comparison
n_iterations = 100
population_size = 20

# Results dictionary
results = {
    'algorithm': [],
    'time_seconds': [],
    'best_score': [],
    'test_score': [],
    'default_score': [],
    'n_evaluations': []
}

# ------------------------------------------------
# Test with default parameters
# ------------------------------------------------
print("\nEvaluating model with default parameters...")
default_model = DecisionTreeClassifier(random_state=42)
default_model.fit(X_train, y_train)
default_score = default_model.score(X_test, y_test)
print(f"Test accuracy with default parameters: {default_score:.4f}")

# ------------------------------------------------
# Run Grid Search (Exhaustive)
# ------------------------------------------------
print("\n" + "="*50)
print("Running GridSearchCV hyperparameter tuning...")

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=hyperparameter_space,
    scoring="accuracy",
    cv=skf,
    n_jobs=-1,
    return_train_score=True
)

# Measure the time it takes to run Grid Search
start_time = time.time()
grid_search.fit(X_train, y_train)
grid_time = time.time() - start_time

# Evaluate model with best parameters
grid_model = DecisionTreeClassifier(random_state=42).set_params(**grid_search.best_params_)
grid_model.fit(X_train, y_train)
grid_test_score = grid_model.score(X_test, y_test)

print(f"Grid Search completed in {format_time(grid_time)}")
print(f"Best validation score: {grid_search.best_score_:.4f}")
print(f"Test accuracy: {grid_test_score:.4f}")
print(f"Improvement over default: {(grid_test_score - default_score) * 100:.2f}%")
print(f"Total evaluations: {total_combinations}")

# Store results
results['algorithm'].append('Grid Search')
results['time_seconds'].append(grid_time)
results['best_score'].append(grid_search.best_score_)
results['test_score'].append(grid_test_score)
results['default_score'].append(default_score)
results['n_evaluations'].append(total_combinations)

# ------------------------------------------------
# Run Random Search (with n_iterations evaluations)
# ------------------------------------------------
print("\n" + "="*50)
print(f"Running RandomizedSearchCV hyperparameter tuning with {n_iterations} iterations...")

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_distributions=hyperparameter_space,
    n_iter=n_iterations,
    scoring="accuracy",
    cv=skf,
    n_jobs=-1,
    random_state=42,
    return_train_score=True
)

# Measure the time it takes to run Random Search
start_time = time.time()
random_search.fit(X_train, y_train)
random_time = time.time() - start_time

# Evaluate model with best parameters
random_model = DecisionTreeClassifier(random_state=42).set_params(**random_search.best_params_)
random_model.fit(X_train, y_train)
random_test_score = random_model.score(X_test, y_test)

print(f"Random Search completed in {format_time(random_time)}")
print(f"Best validation score: {random_search.best_score_:.4f}")
print(f"Test accuracy: {random_test_score:.4f}")
print(f"Improvement over default: {(random_test_score - default_score) * 100:.2f}%")
print(f"Total evaluations: {n_iterations}")

# Store results
results['algorithm'].append('Random Search')
results['time_seconds'].append(random_time)
results['best_score'].append(random_search.best_score_)
results['test_score'].append(random_test_score)
results['default_score'].append(default_score)
results['n_evaluations'].append(n_iterations)

# ------------------------------------------------
# Run Genetic Algorithm (with same n_iterations)
# ------------------------------------------------
print("\n" + "="*50)
print(f"Running Genetic Algorithm with {n_iterations} generations, population size {population_size}...")

# Initialize the Genetic Tuner
genetic_tuner = GeneticTuner(
    **common_params,
    population_size=population_size,
    generations=n_iterations,
    mutation_rate=0.1
)

# Measure the time it takes to run Genetic Algorithm
start_time = time.time()
genetic_tuner.fit(train_set=[X_train, y_train], eval_set=[X_test, y_test], direction="maximize")
ga_time = time.time() - start_time

# Evaluate model with best parameters
ga_model = common_params['model'].set_params(**genetic_tuner.best_params_)
ga_model.fit(X_train, y_train)
ga_test_score = ga_model.score(X_test, y_test)

# Estimate total evaluations (generations * population size)
ga_evaluations = n_iterations * population_size

print(f"Genetic Algorithm completed in {format_time(ga_time)}")
print(f"Best validation score: {genetic_tuner.best_score_:.4f}")
print(f"Test accuracy: {ga_test_score:.4f}")
print(f"Improvement over default: {(ga_test_score - default_score) * 100:.2f}%")
print(f"Total evaluations: {ga_evaluations}")

# Store results
results['algorithm'].append('Genetic Algorithm')
results['time_seconds'].append(ga_time)
results['best_score'].append(genetic_tuner.best_score_)
results['test_score'].append(ga_test_score)
results['default_score'].append(default_score)
results['n_evaluations'].append(ga_evaluations)

# ------------------------------------------------
# Run Particle Swarm Optimization (with same n_iterations)
# ------------------------------------------------
print("\n" + "="*50)
print(f"Running Particle Swarm Optimization with {n_iterations} iterations, particles {population_size}...")

# Initialize the PSO Tuner with custom parameters
pso_tuner = PSOTuner(
    **common_params,
    n_particles=population_size,
    iterations=n_iterations,
    w=0.7,
    c1=1.5,
    c2=2.0
)

# Measure the time it takes to run PSO
start_time = time.time()
pso_tuner.fit(train_set=[X_train, y_train], eval_set=[X_test, y_test], direction="maximize")
pso_time = time.time() - start_time

# Evaluate model with best parameters
pso_model = common_params['model'].set_params(**pso_tuner.best_params_)
pso_model.fit(X_train, y_train)
pso_test_score = pso_model.score(X_test, y_test)

# Estimate total evaluations (iterations * particles)
pso_evaluations = n_iterations * population_size

print(f"PSO completed in {format_time(pso_time)}")
print(f"Best validation score: {pso_tuner.best_score_:.4f}")
print(f"Test accuracy: {pso_test_score:.4f}")
print(f"Improvement over default: {(pso_test_score - default_score) * 100:.2f}%")
print(f"Total evaluations: {pso_evaluations}")

# Store results
results['algorithm'].append('PSO')
results['time_seconds'].append(pso_time)
results['best_score'].append(pso_tuner.best_score_)
results['test_score'].append(pso_test_score)
results['default_score'].append(default_score)
results['n_evaluations'].append(pso_evaluations)

# ------------------------------------------------
# Comparison Summary
# ------------------------------------------------
print("\n" + "="*50)
print("COMPARISON SUMMARY")
print("="*50)

# Create a DataFrame for better display
comparison_df = pd.DataFrame(results)
comparison_df['improvement'] = (comparison_df['test_score'] - comparison_df['default_score']) * 100
comparison_df['time_formatted'] = comparison_df['time_seconds'].apply(format_time)

# Display the comparison
print(comparison_df[['algorithm', 'time_formatted', 'best_score', 'test_score', 'improvement', 'n_evaluations']])

# Determine the winner in each category
faster_algo = comparison_df.loc[comparison_df['time_seconds'].idxmin()]['algorithm']
better_validation = comparison_df.loc[comparison_df['best_score'].idxmax()]['algorithm']
better_test = comparison_df.loc[comparison_df['test_score'].idxmax()]['algorithm']

print("\nResults:")
print(f"Faster algorithm: {faster_algo}")
print(f"Better validation score: {better_validation}")
print(f"Better test score: {better_test}")

# Calculate efficiency (score per evaluation)
comparison_df['efficiency'] = comparison_df['best_score'] / comparison_df['n_evaluations']
most_efficient = comparison_df.loc[comparison_df['efficiency'].idxmax()]['algorithm']
print(f"Most efficient algorithm (score/evaluation): {most_efficient}")

# Calculate speedup
speedup = max(comparison_df['time_seconds']) / min(comparison_df['time_seconds'])
print(f"Speed difference: {speedup:.2f}x")

# ------------------------------------------------
# Visualize the results
# ------------------------------------------------
try:
    # Set up the figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot execution times
    ax1.bar(comparison_df['algorithm'], comparison_df['time_seconds'], color=['blue', 'green', 'orange', 'red'])
    ax1.set_title('Execution Time (lower is better)')
    ax1.set_ylabel('Time (seconds)')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot accuracy scores
    algorithms = comparison_df['algorithm'].tolist()
    x = np.arange(len(algorithms))
    width = 0.35
    
    ax2.bar(x - width/2, comparison_df['best_score'], width, label='Validation Score')
    ax2.bar(x + width/2, comparison_df['test_score'], width, label='Test Score')
    ax2.axhline(y=default_score, color='r', linestyle='-', label='Default Score')
    
    ax2.set_title('Accuracy (higher is better)')
    ax2.set_ylabel('Accuracy Score')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    ax2.legend()
    
    # Plot efficiency (score per evaluation)
    ax3.bar(comparison_df['algorithm'], comparison_df['efficiency'] * 1000, color=['purple', 'brown', 'cyan', 'magenta'])
    ax3.set_title('Efficiency (score per evaluation × 1000)')
    ax3.set_ylabel('Efficiency Score')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('tuner_comparison.png')
    print("\nVisualization saved as 'tuner_comparison.png'")
except Exception as e:
    print(f"\nCouldn't create visualization: {e}")

print("\nDone!") 