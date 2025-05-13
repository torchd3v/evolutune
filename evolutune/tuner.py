import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score
import random
import warnings


class GeneticTuner:
    def __init__(self, model, param_grid, scoring, population_size=10, generations=100, mutation_rate=0.1, cv=None,
                 random_state=None, n_jobs=None):
        """
        A Genetic Algorithm-based hyperparameter tuner for machine learning models.

        Parameters
        ----------
        model : object
            The machine learning model for which hyperparameters are to be tuned.
        param_grid : dict
            Dictionary specifying the hyperparameter search space.
        scoring : str or callable
            The scoring metric to optimize during tuning.
        population_size : int, optional, default: 10
            The size of the population in each generation.
        generations : int, optional, default: 100
            The number of generations in the genetic algorithm.
        mutation_rate : float, optional, default: 0.1
            The probability of a mutation occurring during crossover.
        random_state : int or None, optional, default: None
            Seed for reproducibility.
        cv : int or None, optional, default: None
            The number of cross validation during fitness evaluation.
        n_jobs : int or None, optional, default: None
            The number of parallel jobs to run during fitness evaluation.

        Attributes
        ----------
        best_score_ : float
            The best score achieved during the tuning process.
        best_params_ : dict
            The set of hyperparameters that produced the best score.

        Methods
        -------
        fit(train_set, eval_set=None, direction="maximize")
            Fit the GeneticTuner on the training set and optional evaluation set.

        initialize_population(population_size)
            Initialize a population of individuals with random hyperparameters.

        crossover(parent1, parent2)
            Perform crossover between two parents to generate two children.

        mutate(individual, mutation_rate)
            Introduce random mutations to an individual's hyperparameters.

        calculate_fitness(train_set, eval_set, parameters)
            Evaluate the fitness (scoring metric) of a set of hyperparameters.

        """
        # Set the random seed
        if random_state is not None:
            # Issue a warning about setting a random seed
            message = ("Setting a random seed can be beneficial for reproducibility, "
                       "but it goes against the algorithm's nature. Consider trying multiple seeds.")
            warnings.warn(message, UserWarning, stacklevel=2)
            np.random.seed(random_state)
            random.seed(random_state)
            Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(np.random.seed)(random_state + i) for i in range(population_size))

        # Initializations for genetic algorithm
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.cv = cv
        self.n_jobs = n_jobs

    # Genetic Algorithm functions
    def initialize_population(self, population_size: int) -> list:
        """
        Initialize a population of individuals with random hyperparameters.

        Parameters
        ----------
        population_size : int
            The size of the population.

        Returns
        -------
        list of individuals, each represented as a dictionary of hyperparameters.

        """
        # Calculate the total number of possible combinations
        total_combinations = 1
        for values in self.param_grid.values():
            total_combinations *= len(values)

        # Check if the population size exceeds the total number of combinations
        if population_size > total_combinations:
            warnings.warn("Warning: Population size exceeds the total number of possible combinations.")

        # Initialize the population
        return [{key: random.choice(values) for key, values in self.param_grid.items()} for _ in range(population_size)]

    def crossover(self, parent1: dict, parent2: dict) -> tuple:
        """
        Perform crossover between two parents to generate two children.

        Parameters
        ----------
        parent1 : dict
            Hyperparameters of the first parent.
        parent2 : dict
            Hyperparameters of the second parent.

        Returns
        -------
        tuple
            Two children generated through crossover.

        """
        crossover_point = np.random.randint(1, len(parent1))
        child1 = dict(list(parent1.items())[:crossover_point] + list(parent2.items())[crossover_point:])
        child2 = dict(list(parent2.items())[:crossover_point] + list(parent1.items())[crossover_point:])
        return child1, child2

    def mutate(self, individual: dict, mutation_rate: float) -> dict:
        """
        Introduce random mutations to an individual's hyperparameters.

        Parameters
        ----------
        individual : dict
            Hyperparameters of an individual.
        mutation_rate : float
            The probability of a mutation occurring for each hyperparameter.

        Returns
        -------
        dict
            Hyperparameters of the mutated individual.

        """
        mask = np.random.rand(len(individual)) < mutation_rate
        genes = [item for i, item in enumerate(individual.items()) if mask[i]]
        new_genes = [(param, random.choice(self.param_grid[param])) for param, value in genes]
        for param, value in new_genes:
            individual[param] = value
        return individual

    def calculate_fitness(self, train_set: list, eval_set: list, parameters: dict) -> float:
        """
        Evaluate the fitness (scoring metric) of a set of hyperparameters.

        Parameters
        ----------
        train_set : list
            Training dataset, represented as a list [X_train, y_train].
        eval_set : list or None
            Evaluation dataset, represented as a list [X_eval, y_eval]. If None, use the training set.
        parameters : dict
            Hyperparameters to be evaluated.

        Returns
        -------
        float
            The fitness score of the hyperparameters.

        """
        # Create a model
        f_model = self.model.set_params(**parameters)
        if self.cv is not None:
            # Evaluate the model
            return np.mean(cross_val_score(f_model, train_set[0], train_set[1], cv=self.cv, scoring=self.scoring))
        else:
            # Get scorer
            scorer = get_scorer(self.scoring)
            # Fit the model on the training data
            f_model.fit(train_set[0], train_set[1])
            # Evaluate the model
            if eval_set is None:
                return scorer(f_model, train_set[0], train_set[1])
            else:
                return scorer(f_model, eval_set[0], eval_set[1])

    def fit(self, train_set: list, eval_set: list = None, direction: str = "maximize"):
        """
        Fit the GeneticTuner on the training set and optional evaluation set.

        Parameters
        ----------
        train_set : list
            Training dataset, represented as a list [X_train, y_train].
        eval_set : list or None, optional, default: None
            Evaluation dataset, represented as a list [X_eval, y_eval]. If None, use the training set.
        direction : str, optional, default: "maximize"
            The optimization direction, either "maximize" or "minimize".

        """
        population = self.initialize_population(self.population_size)
        for generation in range(self.generations):
            fitness_scores = Parallel(n_jobs=self.n_jobs)(
                delayed(self.calculate_fitness)(train_set, eval_set, parameters) for parameters in population)

            if direction == "maximize":
                idx_best_2 = np.argsort(fitness_scores)[::-1][:2]
            elif direction == "minimize":
                idx_best_2 = np.argsort(fitness_scores)[:2]
            else:
                raise ValueError("Invalid direction. Use 'maximize' or 'minimize'.")

            new_population = [population[i] for i in idx_best_2]
            for _ in range(int((len(population) / 2) - 1)):
                parent1 = new_population[0]
                parent2 = new_population[1]
                child1, child2 = self.crossover(parent1, parent2)

                child1 = self.mutate(child1, self.mutation_rate)
                child2 = self.mutate(child2, self.mutation_rate)

                new_population.extend([child1, child2])

            population = np.array(new_population)
            fitness_scores = Parallel(n_jobs=self.n_jobs)(
                delayed(self.calculate_fitness)(train_set, eval_set, parameters) for parameters in population)

        self.best_score_ = max(fitness_scores) if direction == "maximize" else min(fitness_scores)
        self.best_params_ = population[
            np.argmax(fitness_scores) if direction == "maximize" else np.argmin(fitness_scores)]
        
class PSOTuner:
    def __init__(self, model, param_grid, scoring, n_particles=10, iterations=100, w=0.5, c1=1.5, c2=1.5, 
                 cv=None, random_state=None, n_jobs=None):
        """
        A Particle Swarm Optimization (PSO) based hyperparameter tuner for machine learning models.

        Parameters
        ----------
        model : object
            The machine learning model for which hyperparameters are to be tuned.
        param_grid : dict
            Dictionary specifying the hyperparameter search space.
        scoring : str or callable
            The scoring metric to optimize during tuning.
        n_particles : int, optional, default: 10
            The number of particles in the swarm.
        iterations : int, optional, default: 100
            The number of iterations in the PSO algorithm.
        w : float, optional, default: 0.5
            Inertia weight for the PSO algorithm.
        c1 : float, optional, default: 1.5
            Cognitive coefficient (personal best attraction).
        c2 : float, optional, default: 1.5
            Social coefficient (global best attraction).
        cv : int or None, optional, default: None
            The number of cross validation during fitness evaluation.
        random_state : int or None, optional, default: None
            Seed for reproducibility.
        n_jobs : int or None, optional, default: None
            The number of parallel jobs to run during fitness evaluation.

        Attributes
        ----------
        best_score_ : float
            The best score achieved during the tuning process.
        best_params_ : dict
            The set of hyperparameters that produced the best score.
        """
        # Set the random seed
        if random_state is not None:
            message = ("Setting a random seed can be beneficial for reproducibility, "
                       "but it might limit exploration in some cases.")
            warnings.warn(message, UserWarning, stacklevel=2)
            np.random.seed(random_state)
            random.seed(random_state)
            Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(np.random.seed)(random_state + i) for i in range(n_particles))

        # Initializations for PSO algorithm
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.n_particles = n_particles
        self.iterations = iterations
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.cv = cv
        self.n_jobs = n_jobs

    def initialize_swarm(self, n_particles):
        """
        Initialize a swarm of particles with random hyperparameters and velocities.

        Parameters
        ----------
        n_particles : int
            The number of particles in the swarm.

        Returns
        -------
        tuple
            (positions, velocities, personal_best_positions, personal_best_scores)
            where each element is a list of dictionaries or values for each particle.
        """
        # Calculate the total number of possible combinations
        total_combinations = 1
        for values in self.param_grid.values():
            total_combinations *= len(values)

        # Check if the number of particles exceeds the total number of combinations
        if n_particles > total_combinations:
            warnings.warn("Warning: Number of particles exceeds the total number of possible combinations.")

        # Initialize positions (hyperparameter sets)
        positions = [{key: random.choice(values) for key, values in self.param_grid.items()} 
                     for _ in range(n_particles)]
        
        # Initialize velocities (dictionaries of the same structure as positions, with zero values)
        velocities = [{key: 0 for key in self.param_grid.keys()} for _ in range(n_particles)]
        
        # Initialize personal best positions and scores
        personal_best_positions = positions.copy()
        personal_best_scores = [float('-inf') for _ in range(n_particles)]  # Initialize with worst possible score
        
        return positions, velocities, personal_best_positions, personal_best_scores

    def update_velocity(self, position, velocity, personal_best, global_best):
        """
        Update the velocity of a particle based on its current position, personal best, and global best.

        Parameters
        ----------
        position : dict
            Current position (hyperparameters) of the particle.
        velocity : dict
            Current velocity of the particle.
        personal_best : dict
            Personal best position of the particle.
        global_best : dict
            Global best position across all particles.

        Returns
        -------
        dict
            Updated velocity for the particle.
        """
        new_velocity = {}
        
        for param in velocity.keys():
            # Get indices of current, personal best, and global best values in param_grid
            param_values = self.param_grid[param]
            current_idx = param_values.index(position[param])
            pbest_idx = param_values.index(personal_best[param])
            gbest_idx = param_values.index(global_best[param])
            
            # Calculate velocity components
            inertia = self.w * velocity[param]
            cognitive = self.c1 * random.random() * (pbest_idx - current_idx)
            social = self.c2 * random.random() * (gbest_idx - current_idx)
            
            # Update velocity
            new_velocity[param] = inertia + cognitive + social
            
        return new_velocity

    def update_position(self, position, velocity):
        """
        Update the position of a particle based on its velocity.

        Parameters
        ----------
        position : dict
            Current position (hyperparameters) of the particle.
        velocity : dict
            Current velocity of the particle.

        Returns
        -------
        dict
            Updated position for the particle.
        """
        new_position = {}
        
        for param in position.keys():
            param_values = self.param_grid[param]
            current_idx = param_values.index(position[param])
            
            # Calculate new index with velocity and bound it within the parameter values
            new_idx = int(round(current_idx + velocity[param]))
            new_idx = max(0, min(new_idx, len(param_values) - 1))
            
            # Update position
            new_position[param] = param_values[new_idx]
            
        return new_position

    def calculate_fitness(self, train_set, eval_set, parameters):
        """
        Evaluate the fitness (scoring metric) of a set of hyperparameters.

        Parameters
        ----------
        train_set : list
            Training dataset, represented as a list [X_train, y_train].
        eval_set : list or None
            Evaluation dataset, represented as a list [X_eval, y_eval]. If None, use the training set.
        parameters : dict
            Hyperparameters to be evaluated.

        Returns
        -------
        float
            The fitness score of the hyperparameters.
        """
        # Create a model
        f_model = self.model.set_params(**parameters)
        if self.cv is not None:
            # Evaluate the model with cross-validation
            return np.mean(cross_val_score(f_model, train_set[0], train_set[1], cv=self.cv, scoring=self.scoring))
        else:
            # Get scorer
            scorer = get_scorer(self.scoring)
            # Fit the model on the training data
            f_model.fit(train_set[0], train_set[1])
            # Evaluate the model
            if eval_set is None:
                return scorer(f_model, train_set[0], train_set[1])
            else:
                return scorer(f_model, eval_set[0], eval_set[1])

    def fit(self, train_set, eval_set=None, direction="maximize"):
        """
        Fit the PSOTuner on the training set and optional evaluation set.

        Parameters
        ----------
        train_set : list
            Training dataset, represented as a list [X_train, y_train].
        eval_set : list or None, optional, default: None
            Evaluation dataset, represented as a list [X_eval, y_eval]. If None, use the training set.
        direction : str, optional, default: "maximize"
            The optimization direction, either "maximize" or "minimize".
        """
        # Initialize swarm
        positions, velocities, personal_best_positions, personal_best_scores = self.initialize_swarm(self.n_particles)
        
        # Initialize global best
        global_best_score = float('-inf') if direction == "maximize" else float('inf')
        global_best_position = None
        
        # Optimization loop
        for iteration in range(self.iterations):
            # Evaluate fitness of each particle
            fitness_scores = Parallel(n_jobs=self.n_jobs)(
                delayed(self.calculate_fitness)(train_set, eval_set, position) for position in positions)
            
            # Update personal and global bests
            for i, score in enumerate(fitness_scores):
                # Update personal best
                if ((direction == "maximize" and score > personal_best_scores[i]) or 
                    (direction == "minimize" and score < personal_best_scores[i])):
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()
                
                # Update global best
                if ((direction == "maximize" and score > global_best_score) or 
                    (direction == "minimize" and score < global_best_score)):
                    global_best_score = score
                    global_best_position = positions[i].copy()
            
            # Update velocities and positions
            for i in range(self.n_particles):
                velocities[i] = self.update_velocity(
                    positions[i], velocities[i], personal_best_positions[i], global_best_position
                )
                positions[i] = self.update_position(positions[i], velocities[i])
        
        # Set best results
        self.best_score_ = global_best_score
        self.best_params_ = global_best_position
