import pygame
import time
import threading
import random

pygame.init()

# Load images
face_open = pygame.image.load("D:/Projects/Project 2/Project_Dating/animations/face_open.png")  
face_closed = pygame.image.load("D:/Projects/Project 2/Project_Dating/animations/face_closed.png")  
mouth_open = pygame.image.load("D:/Projects/Project 2/Project_Dating/animations/mouth_open.png")
mouth_closed = pygame.image.load("animations/mouth_closed.png")

# Set screen
screen = pygame.display.set_mode(face_open.get_size())
pygame.display.set_caption("AI Girl Animation")

# Global variables
running = True
is_speaking = False
last_blink_time = time.time()
blink_duration = 0.1  
blink_interval = random.uniform(2, 5)  # Random blink interval

# Function to update speaking status
def set_speaking(status):
    global is_speaking
    is_speaking = status

# Main animation loop
def run_animation():
    global running, is_speaking, last_blink_time, blink_interval  

    while running:
        current_time = time.time()

        # Blinking logic (happens even during speech)
        if current_time - last_blink_time > blink_interval:
            screen.blit(face_closed, (0, 0))
            pygame.display.flip()
            time.sleep(blink_duration)
            last_blink_time = current_time
            blink_interval = random.uniform(2, 5)  # Set new random blink time

        # Speaking logic
        if is_speaking:
            screen.blit(mouth_open, (0, 0))
            pygame.display.flip()
            time.sleep(0.15)
            screen.blit(mouth_closed, (0, 0))
            pygame.display.flip()
            time.sleep(0.15)
        else:
            screen.blit(face_open, (0, 0))
            pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False  

# Start animation in a separate thread
animation_thread = threading.Thread(target=run_animation)
animation_thread.start()

def draw_mouth_open():
    screen.blit(mouth_open, (0, 0))
    pygame.display.flip()

def draw_mouth_closed():
    screen.blit(mouth_closed, (0, 0))
    pygame.display.flip()

def stop_lip_movement():
    """Ensure lips stay closed after speech ends"""
    draw_mouth_closed()