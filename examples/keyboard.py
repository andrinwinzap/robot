import pygame
import numpy as np
import os
from robot_api import Robot, CartesianSpace

# --- ROBOT SETUP ---
robot = Robot()
robot.set_fake_hardware_mode(True)

# --- UI CONSTANTS ---
WIDTH, HEIGHT = 850, 520
BG_COLOR = (15, 15, 18)
PANEL_COLOR = (28, 28, 33)
BTN_COLOR = (45, 45, 52)
BTN_ACTIVE = (0, 160, 255)
TEXT_COLOR = (240, 240, 240)

class AxisButton:
    def __init__(self, x, y, w, h, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.is_pressed = False
        self.mouse_down = False # Track mouse state separately

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
        pygame.draw.rect(screen, (60, 60, 70), self.rect)
        pygame.draw.rect(screen, (200, 200, 200), self.handle_rect)
        val_txt = font.render(f"{self.label}: {self.value:.3f}", True, (140, 140, 140))
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
    pygame.display.set_caption("Robot Axis Control")
    
    label_font = pygame.font.SysFont("Arial", 22, bold=True)
    font = pygame.font.SysFont("Arial", 14)
    big_font = pygame.font.SysFont("Arial", 18, bold=True)
    clock = pygame.time.Clock()

    l_slider = Slider(550, 360, 240, 0.005, 0.1, 0.03, "Linear Speed")
    a_slider = Slider(550, 430, 240, 0.1, 1.5, 0.5, "Angular Speed")

    btns = [
        AxisButton(120, 60, 60, 60, "+X"),  # 0: W
        AxisButton(120, 190, 60, 60, "-X"), # 1: S
        AxisButton(55, 125, 60, 60, "+Y"),  # 2: A
        AxisButton(185, 125, 60, 60, "-Y"), # 3: D
        AxisButton(300, 60, 65, 60, "+Z"),  # 4: Space
        AxisButton(300, 190, 65, 60, "-Z"), # 5: Shift

        AxisButton(120, 310, 60, 60, "+P"), # 6: Up
        AxisButton(120, 440, 60, 60, "-P"), # 7: Down
        AxisButton(55, 375, 60, 60, "+Yw"), # 8: Q
        AxisButton(185, 375, 60, 60, "-Yw"),# 9: E
        AxisButton(300, 310, 65, 60, "+R"), # 10: Left
        AxisButton(300, 440, 65, 60, "-R")  # 11: Right
    ]

    while True:
        screen.fill(BG_COLOR)
        lin_vel, ang_vel = np.zeros(3), np.zeros(3)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return
            l_slider.handle_event(event)
            a_slider.handle_event(event)
            for b in btns: b.handle_event(event)

        keys = pygame.key.get_pressed()
        l_spd, a_spd = l_slider.value, a_slider.value

        # FIX: Map current key states to button highlight state (No latching)
        key_map = [
            keys[pygame.K_w], keys[pygame.K_s], keys[pygame.K_a], keys[pygame.K_d], 
            keys[pygame.K_SPACE], keys[pygame.K_LSHIFT], keys[pygame.K_UP], 
            keys[pygame.K_DOWN], keys[pygame.K_q], keys[pygame.K_e], 
            keys[pygame.K_LEFT], keys[pygame.K_RIGHT]
        ]
        
        for i, is_key_down in enumerate(key_map):
            # Button is active ONLY if key is down OR mouse is clicking it
            btns[i].is_pressed = is_key_down or btns[i].mouse_down

        # Velocity Map based on active buttons
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

        # --- DRAWING ---
        pygame.draw.rect(screen, PANEL_COLOR, (30, 50, 380, 220), border_radius=10)
        pygame.draw.rect(screen, PANEL_COLOR, (30, 300, 380, 215), border_radius=10)
        pygame.draw.rect(screen, PANEL_COLOR, (460, 50, 360, 270), border_radius=10)

        screen.blit(big_font.render("LINEAR AXES", True, (150, 150, 160)), (50, 20))
        screen.blit(big_font.render("ROTATIONAL AXES", True, (150, 150, 160)), (50, 270))

        for b in btns: b.draw(screen, label_font)
        l_slider.draw(screen, font)
        a_slider.draw(screen, font)

        stats = [("TELEMETRY", big_font, (0, 255, 150)),
                 (f"Vel X: {lin_vel[0]:+.3f}", font, TEXT_COLOR),
                 (f"Vel Y: {lin_vel[1]:+.3f}", font, TEXT_COLOR),
                 (f"Vel Z: {lin_vel[2]:+.3f}", font, TEXT_COLOR),
                 (f"Roll:  {ang_vel[0]:+.2f}", font, TEXT_COLOR),
                 (f"Pitch: {ang_vel[1]:+.2f}", font, TEXT_COLOR),
                 (f"Yaw:   {ang_vel[2]:+.2f}", font, TEXT_COLOR)]
        
        for i, (text, f, col) in enumerate(stats):
            screen.blit(f.render(text, True, col), (490, 80 + (i * 25)))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    try: main()
    finally: robot.shutdown(); pygame.quit()