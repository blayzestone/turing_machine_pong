import numpy as np
import multiprocessing

from ai import AI
from config import GameConfig
from game import Game, simulate_match

def compute_fitness(ai):
    game = Game(ai=ai)
    fitness = simulate_match(game)
    return fitness, ai
    
def evolve_population(population_size=100, generations=100, config=None):
    config = config or GameConfig()
    population = [AI(num_states=config.NUM_STATES, input_size=config.INPUT_SIZE) 
                 for _ in range(population_size)]
    
    elite_scores = []
    with multiprocessing.Pool() as pool:
        for generation in range(generations):
            # Process population in parallel
            results = pool.map(compute_fitness, population)
            
            # Process results by original AI groups
            generation_scores = []
            for fitness, ai in results:
                generation_scores.append((fitness, ai))
                
            # Combine with elites and sort
            combined_scores = generation_scores + elite_scores
            combined_scores.sort(key=lambda x: x[0], reverse=True)

            # Update elites
            elite_scores = combined_scores[:population_size // 2]

            # Logging
            fitness_values = [score for score, _ in combined_scores]
            if fitness_values:
                print(f"Generation {generation + 1} "
                    f"Best Score: {max(fitness_values):.3f} "
                      f"Avg Score: {np.mean(fitness_values):.3f} "
                      f"Min Score: {min(fitness_values):.3f}")
                
            # Create next generation (without elites)
            new_population = []
            for fitness, ai in elite_scores:
                    child = AI(rules=ai.rules.copy(), 
                               input_size=ai.input_size, 
                               num_states=ai.num_states)
                    child.mutate_rules(mutation_rate=0.1)

                    new_population.append(child)
            
            population = new_population

    elites = [ai for _, ai in elite_scores]
    return elites