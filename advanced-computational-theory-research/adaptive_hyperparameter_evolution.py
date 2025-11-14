"""
Adaptive Hyperparameter Evolution (AHE)
Novel meta-learning algorithm for joint architecture and hyperparameter optimization
"""

import numpy as np
from typing import Callable, List, Tuple, Dict, Any
from dataclasses import dataclass
import copy


@dataclass
class Individual:
    """Represents a candidate model with architecture and hyperparameters"""
    architecture: Dict[str, Any]  # Network architecture parameters
    hyperparams: Dict[str, float]  # Learning rate, momentum, etc.
    fitness: float = -np.inf
    complexity: int = 0  # Model complexity measure


class AdaptiveHyperparameterEvolution:
    """
    Evolutionary algorithm that co-optimizes:
    1. Neural architecture (layers, units, connections)
    2. Training hyperparameters (learning rate, momentum, regularization)
    3. Optimization strategy (adaptive learning rate scheduling)
    """
    
    def __init__(
        self,
        loss_function: Callable,
        architecture_space: Dict[str, List],
        population_size: int = 50,
        elite_fraction: float = 0.2,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        max_generations: int = 100,
        complexity_weight: float = 0.01
    ):
        self.loss_function = loss_function
        self.architecture_space = architecture_space
        self.population_size = population_size
        self.elite_size = int(population_size * elite_fraction)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.complexity_weight = complexity_weight
        
        self.population: List[Individual] = []
        self.archive: List[Individual] = []  # Pareto-optimal solutions
        self.generation = 0
        
    def initialize_population(self) -> None:
        """Create initial random population from architecture space"""
        self.population = []
        for _ in range(self.population_size):
            individual = self._sample_individual()
            self.population.append(individual)
            
    def _sample_individual(self) -> Individual:
        """Sample random architecture and hyperparameters"""
        architecture = {}
        for param, values in self.architecture_space.items():
            if isinstance(values, list):
                architecture[param] = np.random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                # Continuous range (min, max)
                architecture[param] = np.random.uniform(values[0], values[1])
                
        hyperparams = {
            'learning_rate': 10 ** np.random.uniform(-5, -1),
            'momentum': np.random.uniform(0.5, 0.99),
            'weight_decay': 10 ** np.random.uniform(-6, -2),
            'batch_size': np.random.choice([16, 32, 64, 128, 256])
        }
        
        complexity = self._compute_complexity(architecture)
        
        return Individual(architecture, hyperparams, complexity=complexity)
    
    def _compute_complexity(self, architecture: Dict[str, Any]) -> int:
        """Estimate model complexity (parameters, FLOPs, etc.)"""
        complexity = 0
        
        # Example: count layers and units
        if 'num_layers' in architecture:
            complexity += architecture['num_layers'] * 1000
        if 'hidden_units' in architecture:
            complexity += sum(architecture['hidden_units']) if isinstance(
                architecture['hidden_units'], list
            ) else architecture['hidden_units']
            
        return complexity
    
    def evaluate_fitness(self, individual: Individual) -> float:
        """
        Evaluate individual fitness: minimize loss, penalize complexity
        Returns: fitness score (higher is better)
        """
        try:
            loss = self.loss_function(individual.architecture, individual.hyperparams)
            
            # Multi-objective: accuracy vs complexity
            fitness = -loss - self.complexity_weight * individual.complexity
            
            return fitness
        except Exception as e:
            # Invalid configuration
            return -np.inf
    
    def evaluate_population(self) -> None:
        """Evaluate fitness for all individuals in population"""
        for individual in self.population:
            if individual.fitness == -np.inf:  # Not yet evaluated
                individual.fitness = self.evaluate_fitness(individual)
    
    def selection(self) -> List[Individual]:
        """Tournament selection: select parents for reproduction"""
        parents = []
        
        # Elitism: always keep top performers
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        parents.extend(sorted_pop[:self.elite_size])
        
        # Tournament selection for remaining slots
        tournament_size = 3
        while len(parents) < self.population_size:
            tournament = np.random.choice(self.population, tournament_size, replace=False)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(copy.deepcopy(winner))
            
        return parents
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Uniform crossover: combine architectures and hyperparameters"""
        if np.random.random() > self.crossover_rate:
            return copy.deepcopy(parent1)
        
        child_arch = {}
        for key in parent1.architecture:
            child_arch[key] = (
                parent1.architecture[key] if np.random.random() < 0.5
                else parent2.architecture[key]
            )
        
        child_hyper = {}
        for key in parent1.hyperparams:
            # Arithmetic crossover for continuous hyperparameters
            alpha = np.random.random()
            child_hyper[key] = (
                alpha * parent1.hyperparams[key] +
                (1 - alpha) * parent2.hyperparams[key]
            )
        
        complexity = self._compute_complexity(child_arch)
        return Individual(child_arch, child_hyper, complexity=complexity)
    
    def mutate(self, individual: Individual) -> Individual:
        """Gaussian mutation on hyperparameters, discrete mutation on architecture"""
        mutated = copy.deepcopy(individual)
        
        # Mutate architecture
        for key, values in self.architecture_space.items():
            if np.random.random() < self.mutation_rate:
                if isinstance(values, list):
                    mutated.architecture[key] = np.random.choice(values)
                elif isinstance(values, tuple):
                    # Add Gaussian noise
                    current = mutated.architecture[key]
                    noise = np.random.normal(0, (values[1] - values[0]) * 0.1)
                    mutated.architecture[key] = np.clip(
                        current + noise, values[0], values[1]
                    )
        
        # Mutate hyperparameters with adaptive mutation strength
        for key in mutated.hyperparams:
            if np.random.random() < self.mutation_rate:
                # Log-space mutation for learning rate and weight decay
                if 'rate' in key or 'decay' in key:
                    log_val = np.log10(mutated.hyperparams[key])
                    log_val += np.random.normal(0, 0.3)
                    mutated.hyperparams[key] = 10 ** log_val
                else:
                    # Linear space for momentum, etc.
                    mutated.hyperparams[key] *= np.random.uniform(0.8, 1.2)
                    mutated.hyperparams[key] = np.clip(mutated.hyperparams[key], 0, 1)
        
        mutated.complexity = self._compute_complexity(mutated.architecture)
        mutated.fitness = -np.inf  # Needs re-evaluation
        
        return mutated
    
    def update_archive(self) -> None:
        """Maintain Pareto-optimal solutions (accuracy vs complexity trade-off)"""
        candidates = self.population + self.archive
        
        # Compute Pareto frontier
        pareto_front = []
        for candidate in candidates:
            dominated = False
            for other in candidates:
                if other.fitness > candidate.fitness and other.complexity <= candidate.complexity:
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(candidate)
        
        self.archive = pareto_front[:20]  # Keep top 20 Pareto-optimal solutions
    
    def evolve(self) -> Individual:
        """Run evolutionary optimization"""
        self.initialize_population()
        
        for generation in range(self.max_generations):
            self.generation = generation
            
            # Evaluate fitness
            self.evaluate_population()
            
            # Update archive of Pareto-optimal solutions
            self.update_archive()
            
            # Report progress
            best = max(self.population, key=lambda x: x.fitness)
            print(f"Generation {generation}: Best Fitness = {best.fitness:.4f}, "
                  f"Complexity = {best.complexity}")
            
            # Selection
            parents = self.selection()
            
            # Generate offspring
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                child1 = self.crossover(parents[i], parents[i+1])
                child2 = self.crossover(parents[i+1], parents[i])
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                offspring.extend([child1, child2])
            
            # Replace population
            self.population = offspring[:self.population_size]
        
        # Final evaluation
        self.evaluate_population()
        self.update_archive()
        
        # Return best individual
        best_individual = max(self.population, key=lambda x: x.fitness)
        return best_individual


def example_usage():
    """Demonstrate AHE on a toy problem"""
    
    # Define architecture search space
    architecture_space = {
        'num_layers': [2, 3, 4, 5],
        'hidden_units': [(32, 256)],  # Continuous range
        'activation': ['relu', 'tanh', 'gelu'],
        'dropout_rate': [(0.0, 0.5)]
    }
    
    # Mock loss function (replace with actual training)
    def mock_loss(architecture, hyperparams):
        """Simulate training and return validation loss"""
        # Simpler architectures generalize better in this toy example
        base_loss = 1.0
        base_loss += 0.01 * architecture['num_layers']
        base_loss += 0.001 * architecture.get('hidden_units', 128)
        base_loss -= np.log10(hyperparams['learning_rate']) * 0.1
        base_loss += np.random.normal(0, 0.05)  # Stochasticity
        return max(0, base_loss)
    
    # Run AHE
    ahe = AdaptiveHyperparameterEvolution(
        loss_function=mock_loss,
        architecture_space=architecture_space,
        population_size=30,
        max_generations=20
    )
    
    best = ahe.evolve()
    
    print("\n=== Best Solution ===")
    print(f"Architecture: {best.architecture}")
    print(f"Hyperparameters: {best.hyperparams}")
    print(f"Fitness: {best.fitness:.4f}")
    print(f"Complexity: {best.complexity}")
    
    print("\n=== Pareto Archive ===")
    for i, individual in enumerate(ahe.archive[:5]):
        print(f"{i+1}. Fitness: {individual.fitness:.4f}, Complexity: {individual.complexity}")


if __name__ == "__main__":
    example_usage()
