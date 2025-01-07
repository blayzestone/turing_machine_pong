from config import GameConfig
from game import play_game

from evolution import evolve_population

if __name__ == '__main__':
    population_size = 100
    generations = 50

    config = GameConfig()
    ais = evolve_population(population_size, generations=generations, config=config)

    # Get best performer for the game
    play_game(ais[0])

