import pygame
import numpy as np
import argparse
from robot_api import Robot

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Robot Command Center")
parser.add_argument("--fake-hardware", action="store_true", help="Run in simulation mode")
parser.add_argument("--debug", action="store_true", help="Enable robot debug logs")
args = parser.parse_args()

# --- ROBOT SETUP ---
robot = Robot()
robot.set_fake_hardware_mode(args.fake_hardware)
robot.set_debug_mode(args.debug)

# --- UI CONSTANTS ---
WIDTH, HEIGHT = 1150, 580
BG_COLOR = (15, 15, 18)
PANEL_COLOR = (28, 28, 33)
BTN_COLOR = (45, 45, 52)
BTN_ACTIVE = (0, 160, 255)
TEXT_COLOR = (240, 240, 240)
HEADER_COLOR = (0, 255, 150)

class AxisButton:
    def __init__(self, x, y, w, h, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.is_pressed = False
        self.mouse_down = False

    def draw(self, screen, font):
        color = BTN_ACTIVE if self.is_pressed else BTN_COLOR
        pygame.draw.rect(screen, color, self.rect, border_radius=4)
        pygame.draw.rect(screen, (70, 70, 80), self.rect, 1, border_radius=4)
        txt_img = font.render(self.label, True, TEXT_COLOR)
        screen.blit(txt_img, (self.rect.centerx - txt_img.get_width()//2, 
                             self.rect.centery - txt_img.get_height()//2))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.mouse_down = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.mouse_down = False

class Slider:
    def __init__(self, x, y, w, min_val, max_val, start_val, label):
        self.rect = pygame.Rect(x, y, w, 6)
        self.handle_rect = pygame.Rect(x, y - 12, 12, 30)
        self.min_val, self.max_val = min_val, max_val
        self.value = start_val
        self.label = label
        self.dragging = False

    def draw(self, screen, font):
        pygame.draw.rect(screen, (60, 60, 70), self.rect, border_radius=3)
        pygame.draw.rect(screen, (200, 200, 200), self.handle_rect, border_radius=2)
        val_txt = font.render(f"{self.label}: {self.value:.3f}", True, (180, 180, 180))
        screen.blit(val_txt, (self.rect.x, self.rect.y - 25))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.handle_rect.collidepoint(event.pos):
            self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP: self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.handle_rect.centerx = max(self.rect.left, min(event.pos[0], self.rect.right))
            pos_ratio = (self.handle_rect.centerx - self.rect.left) / self.rect.width
            self.value = self.min_val + pos_ratio * (self.max_val - self.min_val)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Robot Command Center")
    
    label_font = pygame.font.SysFont("Arial", 20, bold=True)
    font = pygame.font.SysFont("Consolas", 14) 
    big_font = pygame.font.SysFont("Arial", 18, bold=True)
    clock = pygame.time.Clock()

    l_slider = Slider(460, 500, 300, 0.005, 0.1, 0.03, "Linear Speed (m/s)")
    a_slider = Slider(800, 500, 300, 0.1, 1.5, 0.5, "Angular Speed (rad/s)")

    btns = [
        # Linear
        AxisButton(120, 60, 60, 60, "+X"),  AxisButton(120, 190, 60, 60, "-X"),
        AxisButton(55, 125, 60, 60, "+Y"),  AxisButton(185, 125, 60, 60, "-Y"),
        AxisButton(300, 60, 65, 60, "+Z"),  AxisButton(300, 190, 65, 60, "-Z"),
        # Rotation
        AxisButton(120, 330, 60, 60, "+P"), AxisButton(120, 460, 60, 60, "-P"),
        AxisButton(55, 395, 60, 60, "+Yw"), AxisButton(185, 395, 60, 60, "-Yw"),
        AxisButton(300, 330, 65, 60, "+R"), AxisButton(300, 460, 65, 60, "-R")
    ]

    while True:
        screen.fill(BG_COLOR)
        pose = robot.cartesian_space.read()
        joint_pt = robot.joint_space.read()
        
        lin_vel, ang_vel = np.zeros(3), np.zeros(3)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return
            l_slider.handle_event(event)
            a_slider.handle_event(event)
            for b in btns: b.handle_event(event)

        keys = pygame.key.get_pressed()
        l_spd, a_spd = l_slider.value, a_slider.value
        key_states = [
            keys[pygame.K_w], keys[pygame.K_s], keys[pygame.K_a], keys[pygame.K_d], 
            keys[pygame.K_SPACE], keys[pygame.K_LSHIFT], keys[pygame.K_UP], 
            keys[pygame.K_DOWN], keys[pygame.K_q], keys[pygame.K_e], 
            keys[pygame.K_LEFT], keys[pygame.K_RIGHT]
        ]
        
        for i, down in enumerate(key_states):
            btns[i].is_pressed = down or btns[i].mouse_down

        if btns[0].is_pressed: lin_vel[0] += l_spd
        if btns[1].is_pressed: lin_vel[0] -= l_spd
        if btns[2].is_pressed: lin_vel[1] += l_spd
        if btns[3].is_pressed: lin_vel[1] -= l_spd
        if btns[4].is_pressed: lin_vel[2] += l_spd
        if btns[5].is_pressed: lin_vel[2] -= l_spd
        if btns[6].is_pressed: ang_vel[1] += a_spd
        if btns[7].is_pressed: ang_vel[1] -= a_spd
        if btns[8].is_pressed: ang_vel[2] += a_spd
        if btns[9].is_pressed: ang_vel[2] -= a_spd
        if btns[10].is_pressed: ang_vel[0] += a_spd
        if btns[11].is_pressed: ang_vel[0] -= a_spd

        robot.cartesian_space.twist(tuple(lin_vel), tuple(ang_vel))

        # --- DRAW ---
        pygame.draw.rect(screen, PANEL_COLOR, (30, 50, 380, 220), border_radius=10)
        pygame.draw.rect(screen, PANEL_COLOR, (30, 320, 380, 220), border_radius=10)
        pygame.draw.rect(screen, PANEL_COLOR, (440, 50, 680, 380), border_radius=10)

        # UI Info bar
        mode_col = (50, 200, 255) if args.fake_hardware else (255, 100, 100)
        mode_txt = "FAKE HARDWARE: ON" if args.fake_hardware else "REAL HARDWARE: ON"
        screen.blit(big_font.render(mode_txt, True, mode_col), (880, 20))
        
        if args.debug:
            screen.blit(font.render("DEBUG MODE: ENABLED", True, (255, 255, 0)), (460, 440))

        screen.blit(big_font.render("LINEAR (WASD)", True, (150, 150, 160)), (50, 20))
        screen.blit(big_font.render("ROTATION (Arrows)", True, (150, 150, 160)), (50, 290))
        screen.blit(big_font.render("SYSTEM TELEMETRY", True, HEADER_COLOR), (460, 20))

        for b in btns: b.draw(screen, label_font)
        l_slider.draw(screen, font)
        a_slider.draw(screen, font)

        # Data columns
        col1_x, col2_x, y_start = 460, 780, 80
        c_lines = [
            ("CARTESIAN POSE", big_font, HEADER_COLOR),
            (f"X:     {pose.position[0]: .5f}", font, TEXT_COLOR),
            (f"Y:     {pose.position[1]: .5f}", font, TEXT_COLOR),
            (f"Z:     {pose.position[2]: .5f}", font, TEXT_COLOR),
            (f"Roll:  {pose.orientation[0]: .5f}", font, TEXT_COLOR),
            (f"Pitch: {pose.orientation[1]: .5f}", font, TEXT_COLOR),
            (f"Yaw:   {pose.orientation[2]: .5f}", font, TEXT_COLOR),
        ]
        j_lines = [("JOINT SPACE", big_font, HEADER_COLOR)]
        for i, val in enumerate(joint_pt.joint_configuration):
            j_lines.append((f"q{i+1}: {val: .5f} rad", font, TEXT_COLOR))

        for i, (txt, f, c) in enumerate(c_lines): screen.blit(f.render(txt, True, c), (col1_x, y_start + i*28))
        for i, (txt, f, c) in enumerate(j_lines): screen.blit(f.render(txt, True, c), (col2_x, y_start + i*28))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    try: main()
    finally: robot.shutdown(); pygame.quit()