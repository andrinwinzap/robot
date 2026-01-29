from robot_api import Robot, CartesianSpace
import pygame
import numpy as np
import time
import os

# --- ROBOT SETUP ---
robot = Robot()
robot.set_fake_hardware_mode(True)
robot.set_debug_mode(False)

# --- PYGAME SETUP ---
if "DISPLAY" not in os.environ:
    print("WARNING: No display detected. Use 'ssh -X' for remote windows.")

pygame.init()
screen = pygame.display.set_mode((450, 350))
pygame.display.set_caption("6-DOF Robot Controller")
font = pygame.font.SysFont("Arial", 16)

def main():
    running = True
    clock = pygame.time.Clock()

    print("Pygame Window Started. FOCUS THE WINDOW TO CONTROL.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Quit with Escape or X
        if keys[pygame.K_ESCAPE] or keys[pygame.K_x]:
            running = False

        lin_vel = np.zeros(3)
        ang_vel = np.zeros(3)
        
        # --- GAINS ---
        SPEED = 0.03
        ROT_SPEED = 0.5

        # 1. LINEAR MOVEMENT (X, Y, Z)
        # Forward/Back (X)
        if keys[pygame.K_w]:      lin_vel[0] += SPEED
        if keys[pygame.K_s]:      lin_vel[0] -= SPEED
        # Strafe Left/Right (Y)
        if keys[pygame.K_a]:      lin_vel[1] += SPEED
        if keys[pygame.K_d]:      lin_vel[1] -= SPEED
        # Vertical Up/Down (Z)
        if keys[pygame.K_SPACE]:  lin_vel[2] += SPEED
        if keys[pygame.K_LSHIFT]: lin_vel[2] -= SPEED

        # 2. ANGULAR MOVEMENT (Roll, Pitch, Yaw)
        # ROLL (X-axis rotation) - Keys Q and E
        if keys[pygame.K_LEFT]:      ang_vel[0] += ROT_SPEED
        if keys[pygame.K_RIGHT]:      ang_vel[0] -= ROT_SPEED
        # PITCH (Y-axis rotation) - Arrow Up and Down
        if keys[pygame.K_UP]:    ang_vel[1] += ROT_SPEED
        if keys[pygame.K_DOWN]:  ang_vel[1] -= ROT_SPEED
        # YAW (Z-axis rotation) - Arrow Left and Right
        if keys[pygame.K_q]:  ang_vel[2] += ROT_SPEED
        if keys[pygame.K_e]: ang_vel[2] -= ROT_SPEED

        # Send command
        robot.cartesian_space.twist(tuple(lin_vel), tuple(ang_vel))

        # --- UI UPDATE ---
        screen.fill((30, 30, 30))
        
        # Display feedback
        lines = [
            f"LINEAR  X: {lin_vel[0]:.2f} Y: {lin_vel[1]:.2f} Z: {lin_vel[2]:.2f}",
            f"ANGULAR R: {ang_vel[0]:.2f} P: {ang_vel[1]:.2f} Y: {ang_vel[2]:.2f}",
            "",
            "CONTROLS:",
            "WASD: Translate XY",
            "SPACE/L-SHIFT: Translate Z",
            "Q/E: Roll (X-Rot)",
            "UP/DOWN: Pitch (Y-Rot)",
            "LEFT/RIGHT: Yaw (Z-Rot)",
            "ESC: Quit"
        ]

        for i, text in enumerate(lines):
            img = font.render(text, True, (200, 255, 200) if i < 2 else (200, 200, 200))
            screen.blit(img, (20, 20 + (i * 25)))
        
        pygame.display.flip()
        clock.tick(50) 

    robot.cartesian_space.twist((0, 0, 0), (0, 0, 0))
    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        robot.shutdown()