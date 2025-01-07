import numpy as np
from game_objects import Paddle, Ball
from config import GameConfig

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