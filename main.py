from dataclasses import dataclass
import numpy as np
import sys
import multiprocessing

@dataclass
class GameConfig:
    """Game configuration constants"""
    SCREEN_WIDTH: int = 1000
    SCREEN_HEIGHT: int = 600
    BLACK: tuple = (0, 0, 0)
    WHITE: tuple = (255, 255, 255)
    INPUT_SIZE: int = 6
    NUM_STATES: int = 32
    ball_radius: int = 4
    ball_speed: int = 6
    paddle_speed: int = 6
    paddle_width: int = 5
    paddle_height: int = 80
    paddle_offset: int = 10  # Distance from the screen edge

# Custom Rectangle Class
class Rect:
    def __init__(self, x, y, width, height):
        self.left = x
        self.top = y
        self.width = width
        self.height = height

    @property
    def x(self):
        return self.left

    @x.setter
    def x(self, value):
        self.left = value

    @property
    def y(self):
        return self.top

    @y.setter
    def y(self, value):
        self.top = value

    @property
    def right(self):
        return self.left + self.width

    @right.setter
    def right(self, value):
        self.left = value - self.width

    @property
    def bottom(self):
        return self.top + self.height

    @bottom.setter
    def bottom(self, value):
        self.top = value - self.height  # Setter to handle assignment to bottom

    @property
    def centerx(self):
        return self.left + self.width / 2

    @centerx.setter
    def centerx(self, value):
        self.left = value - self.width / 2

    @property
    def centery(self):
        return self.top + self.height / 2

    @centery.setter
    def centery(self, value):
        self.top = value - self.height / 2

    def colliderect(self, other):
        return (
            self.left < other.right and
            self.right > other.left and
            self.top < other.bottom and
            self.bottom > other.top
        )

class GameObject:
    def __init__(self, x, y, width, height):
        self.rect = Rect(x, y, width, height)

class Paddle(GameObject):
    def __init__(self, x, y, width, height, speed, config=None):
        super().__init__(x, y, width, height)
        self.speed = speed
        self.direction = 0  # 0 = stationary/up, 1 = down
        self.config = config or GameConfig()
    
    def move_up(self):
        self.direction = 0
        self.rect.top = max(self.rect.top - self.speed, 0)
    
    def move_down(self):
        self.direction = 1
        self.rect.top = min(
            self.rect.top + self.speed, 
            self.config.SCREEN_HEIGHT - self.rect.height
        )
    
    def reset(self):
        self.rect.centery = self.config.SCREEN_HEIGHT // 2
        self.direction = 0

class Ball(GameObject):
    def __init__(self, x, y, radius, speed, config=None):
        super().__init__(x - radius, y - radius, radius * 2, radius * 2)
        self.radius = radius
        self.speed_x = speed
        self.speed_y = speed
        self.config = config or GameConfig()
    
    def move(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y
        
        # Bounce off top and bottom
        if self.rect.top <= 0:
            self.rect.top = 0
            self.speed_y = abs(self.speed_y)
        elif self.rect.bottom >= self.config.SCREEN_HEIGHT:
            self.rect.bottom = self.config.SCREEN_HEIGHT
            self.speed_y = -abs(self.speed_y)

    def reset(self):
        center_x = self.config.SCREEN_WIDTH // 2
        center_y = self.config.SCREEN_HEIGHT // 2
        self.rect.centerx = center_x
        self.rect.centery = center_y
        self.speed_x = abs(self.speed_x)
        self.speed_y = abs(self.speed_y)

# AI Class
class AI:
    def __init__(self, rules=None, input_size=6, num_states=32):
        self.input_size = input_size
        self.num_states = num_states
        self.head = 0
        # Corrected bits_per_rule calculation
        self.bits_per_rule = (self.num_states - 1).bit_length() + 1  # Bits for next_state + write_symbol
        self.rules = rules if rules is not None else self.init_binary_rules()

    def reset(self):
        self.head = 0

    def init_binary_rules(self):
        num_rules = self.num_states * 2
        rules = np.zeros((num_rules, self.bits_per_rule), dtype=int)
        for state in range(self.num_states):
            for symbol in range(2):
                rule_index = state * 2 + symbol
                
                # Generate a valid next_state within [0, num_states - 1]
                next_state = np.random.randint(0, self.num_states)
                write_symbol = np.random.choice([0, 1], replace=False)
                
                # Convert next_state to binary bits
                next_state_bits = list(map(int, bin(next_state)[2:].zfill(self.bits_per_rule - 1)))
                
                # Write to the rule
                rules[rule_index, :-1] = next_state_bits  # Store state bits
                rules[rule_index, -1] = write_symbol     # Store write symbol
        return rules

    def lookup_rule(self, state, symbol):
        # Ensure valid state and symbol
        if not (0 <= state < self.num_states):
            raise ValueError(f"Invalid state {state}, must be in range [0, {self.num_states - 1}]")
        if not (0 <= symbol < 2):
            raise ValueError(f"Invalid symbol {symbol}, must be 0 or 1")
        
        # Locate the rule chunk
        rule_index = state * 2 + symbol  # Sequential mapping
        rule_bits = self.rules[rule_index]
        
        # Decode `next_state` from the first `self.bits_per_rule - 1` bits
        next_state_bits = rule_bits[:-1]
        next_state = next_state_bits.dot(1 << np.arange(next_state_bits.size)[::-1])
        
        # Decode `write_symbol` (last bit)
        write_symbol = rule_bits[-1]
        return int(next_state), write_symbol

    def run(self, input_data):
        output = np.zeros(input_data.shape, dtype=int)
        for i in range(self.input_size):
            symbol = input_data[i]
            # Fetch next state and write symbol using the helper
            next_state, write_symbol = self.lookup_rule(self.head, symbol)
            self.head = next_state
            output[i] = write_symbol
        
        left = output[:output.size // 2].sum()
        right = output[output.size // 2:].sum()
        if left == right:
            return 0
        elif left > right:
            return 1
        else:
            return -1
    
    def mutate_rules(self, mutation_rate=0.05):
        mutation_mask = np.random.rand(*self.rules.shape) < mutation_rate
        mutated_rules = self.rules.copy()
        mutated_rules[mutation_mask] = np.random.randint(0, 2, size=mutation_mask.sum())
        self.rules = mutated_rules

class Game:
    def __init__(self, ai=None, config=None):
        self.config = config or GameConfig()
        self.ai = ai
        self.paddle = None
        self.ball = None
        self.score = 0
        self.initialize_objects()

    def initialize_objects(self):
        paddle_center_y = self.config.SCREEN_HEIGHT // 2 - self.config.paddle_height // 2

        self.paddle = Paddle(
            x=self.config.paddle_offset,
            y=paddle_center_y,
            width=self.config.paddle_width,
            height=self.config.paddle_height,
            speed=self.config.paddle_speed,
            config=self.config
        )
        self.ball = Ball(
            x=self.config.SCREEN_WIDTH // 2,
            y=self.config.SCREEN_HEIGHT // 2,
            radius=self.config.ball_radius,
            speed=self.config.ball_speed,
            config=self.config
        )

    def reset(self):
        self.paddle.reset()
        self.ball.reset()
        if self.ai:
            self.ai.reset()
    
    def get_normalized_inputs(self):
        paddle_direction = self.paddle.direction
        paddle_screen_half = 1 if self.paddle.rect.centery > self.config.SCREEN_HEIGHT // 2 else 0
        ball_screen_half = 1 if self.ball.rect.centery > self.config.SCREEN_HEIGHT // 2 else 0
        ball_y_direction = 1 if self.ball.speed_y > 0 else 0
        ball_x_direction = 1 if self.ball.speed_x > 0 else 0
        relation_to_ball = 1 if self.paddle.rect.centery > self.ball.rect.centery else 0
        return np.array([
            paddle_direction,
            paddle_screen_half,
            ball_screen_half,
            ball_y_direction,
            ball_x_direction,
            relation_to_ball
        ], dtype=int)
    
    def update(self):
        # AI control for paddle
        if self.ai:
            inputs = self.get_normalized_inputs()
            output = self.ai.run(inputs)
            if output == 1:
                self.paddle.move_up()
            elif output == -1:
                self.paddle.move_down()
        
        # Move ball and handle collisions
        self.ball.move()
        
        # Collision detection
        hit = False
        if self.ball.rect.colliderect(self.paddle.rect):
            self.ball.rect.left = self.paddle.rect.right + 1
            self.ball.speed_x = -self.ball.speed_x
            hit = True
        return hit

def simulate_match(game, max_attempts=50):
    total_hits = 0
    total_misses = 0
    hit_accuracies = []  # Track how centered the hits are
    paddle_height = game.paddle.rect.height

    game.reset()
    while (total_hits + total_misses) < max_attempts:
        # Store paddle and ball positions before update
        paddle_center_y = game.paddle.rect.centery
        ball_y = game.ball.rect.centery
        
        # Update game state
        if game.update():
            total_hits += 1

            # Calculate hit accuracy (0 = perfect center, 1 = edge hit)
            hit_offset = abs(paddle_center_y - ball_y)
            hit_accuracy = 1 - (hit_offset / (paddle_height / 2))
            hit_accuracies.append(max(0, hit_accuracy))  # Clamp to [0,1]

        # Paddle missed the ball
        if game.ball.rect.left <= 0:
            total_misses += 1
            game.ball.reset()

        elif game.ball.rect.right >= game.config.SCREEN_WIDTH:
            game.ball.speed_x = -game.ball.speed_x

    hit_miss_ratio = (total_hits / max_attempts)
    avg_hit_accuracy = sum(hit_accuracies) / len(hit_accuracies) if hit_accuracies else 0
    fitness = (0.9 * hit_miss_ratio) + (0.1 * avg_hit_accuracy)
    
    return fitness

def play_game(ai):
    import pygame  # Import pygame locally
    pygame.init()
    game = Game(ai=ai)
    screen = pygame.display.set_mode((game.config.SCREEN_WIDTH, game.config.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    game.reset()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        game.update()
        
        # Handle scoring
        if game.ball.rect.left <= 0:
            game.ball.reset()
        elif game.ball.rect.right >= game.config.SCREEN_WIDTH:
            game.score += 1
            game.ball.speed_x = -game.ball.speed_x

        # Draw objects on the screen
        screen.fill(game.config.BLACK)
        pygame.draw.rect(
            screen, 
            game.config.WHITE, 
            pygame.Rect(game.paddle.rect.left, game.paddle.rect.top, 
                        game.paddle.rect.width, game.paddle.rect.height)
        )
        pygame.draw.ellipse(
            screen, 
            game.config.WHITE, 
            pygame.Rect(game.ball.rect.left, game.ball.rect.top, 
                        game.ball.rect.width, game.ball.rect.height)
        )
        
        pygame.display.flip()
        clock.tick(60)

def compute_fitness(ai):
    game = Game(ai=ai)
    fitness = simulate_match(game)
    return fitness, ai
    
def train_population(population_size=100, generations=100, config=None):
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

if __name__ == '__main__':
    population_size = 100
    generations = 50

    config = GameConfig()
    ais = train_population(population_size, generations=generations, config=config)

    # Get best performer for the game
    play_game(ais[0])

