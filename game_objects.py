from geometry import Rect
from config import GameConfig

class GameObject:
    def __init__(self, x, y, width, height):
        self.rect = Rect(x, y, width, height)

class Paddle(GameObject):
    def __init__(self, x, y, width, height, speed, config=None):
        super().__init__(x, y, width, height)
        self.speed = speed
        self.direction = 0
        self.config = config or GameConfig()
    
    def move_up(self):
        self.direction = 0
        self.rect.top = max(self.rect.top - self.speed, 0)
    
    def move_down(self):
        self.direction = 1
        self.rect.top = min(self.rect.top + self.speed, self.config.SCREEN_HEIGHT - self.rect.height)
    
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
        
        if self.rect.top <= 0:
            self.rect.top = 0
            self.speed_y = abs(self.speed_y)
        elif self.rect.bottom >= self.config.SCREEN_HEIGHT:
            self.rect.bottom = self.config.SCREEN_HEIGHT
            self.speed_y = -abs(self.speed_y)

    def reset(self):
        self.rect.centerx = self.config.SCREEN_WIDTH // 2
        self.rect.centery = self.config.SCREEN_HEIGHT // 2
        self.speed_x = abs(self.speed_x)
        self.speed_y = abs(self.speed_y)
