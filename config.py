from dataclasses import dataclass

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