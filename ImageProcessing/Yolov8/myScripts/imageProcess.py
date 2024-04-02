import random

def generate_random_color():
    """Generate a random color."""
    r = random.randint(0, 255)  # Random value for red component
    g = random.randint(0, 255)  # Random value for green component
    b = random.randint(0, 255)  # Random value for blue component
    return (r, g, b)