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
        self.top = value - self.height

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