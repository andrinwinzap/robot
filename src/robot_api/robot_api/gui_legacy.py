import argparse
import json
import os
import threading

import math
import time

import numpy as np
import pygame

from robot_api import Robot
from robot_api.cartesian_space import CartesianSpace
from robot_api.joint_space import JointSpace

# Allow controller input while window is not focused.
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

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
WIDTH, HEIGHT = 1680, 1050

# Panel geometry (base resolution)
LP_X, LP_Y, LP_W, LP_H = 18, 18, 500, 1014
RP_X, RP_Y, RP_W, RP_H = 534, 18, 1128, 1014
PAD = 22  # inner content padding

# Color palette
BG_COLOR      = (10, 10, 14)
PANEL_BG      = (19, 19, 26)
CARD_BG       = (14, 14, 20)
BORDER_COLOR  = (36, 36, 52)
TEXT_COLOR    = (215, 218, 228)
DIM_COLOR     = (100, 105, 122)
HEADER_COLOR  = (0, 210, 120)
ACCENT_COLOR  = (55, 155, 255)
DANGER_BG     = (130, 35, 35)
DANGER_HOVER  = (180, 55, 55)
DANGER_BORDER = (210, 65, 65)
ATTACH_BG     = (25, 90, 40)
ATTACH_HOVER  = (35, 125, 58)
ATTACH_BORDER = (0, 185, 90)
HOVER_COLOR   = (48, 50, 68)
BTN_COLOR     = (27, 29, 42)
BTN_ACTIVE    = (0, 125, 215)

# Robot parameters
GRIPPER_MIN, GRIPPER_MAX = 0.0, 0.045
GRIPPER_SPD_MIN, GRIPPER_SPD_MAX, GRIPPER_SPD_STEP, GRIPPER_SPD_DEFAULT = 0.005, 0.5, 0.005, 0.05
LIN_SPD_MIN,   LIN_SPD_MAX,   LIN_SPD_STEP,   LIN_SPD_DEFAULT   = 0.01, 1.0,  0.01, 0.02
ANG_SPD_MIN,   ANG_SPD_MAX,   ANG_SPD_STEP,   ANG_SPD_DEFAULT   = 0.1,  2.0,  0.05, 0.2
LIN_ACCEL_MIN, LIN_ACCEL_MAX, LIN_ACCEL_STEP, LIN_ACCEL_DEFAULT = 0.01, 1.0,  0.01, 0.02
ANG_ACCEL_MIN, ANG_ACCEL_MAX, ANG_ACCEL_STEP, ANG_ACCEL_DEFAULT = 0.05, 1.0,  0.05, 0.2
JNT_VEL_MIN,   JNT_VEL_MAX,  JNT_VEL_STEP,  JNT_VEL_DEFAULT   = 0.1,  3.0,  0.1,  1.0

MAPPINGS = {
    "drone": {
        "label": "Drone Style (Legacy)",
        "apply": lambda js, lspd, aspd: (
            np.array([
                -apply_deadzone(js.get_axis(4)) * lspd,
                -apply_deadzone(js.get_axis(3)) * lspd,
                -apply_deadzone(js.get_axis(1)) * lspd,
            ]),
            np.array([0.0, 0.0, apply_deadzone(js.get_axis(0)) * aspd]),
        ),
        "legend": [
            "L-Stick Vert:  Up / Down",
            "L-Stick Horiz: Yaw",
            "R-Stick Vert:  Fwd / Back",
            "R-Stick Horiz: Side",
            "D-Pad Vert:    Pitch",
            "D-Pad Horiz:   Roll",
            "LB / RB:       Gripper",
        ],
    },
    "standard": {
        "label": "Standard Dual Stick",
        "apply": lambda js, lspd, aspd: (
            np.array([
                -apply_deadzone(js.get_axis(1)) * lspd,
                -apply_deadzone(js.get_axis(0)) * lspd,
                0.0,
            ]),
            np.array([
                apply_deadzone(js.get_axis(3)) * aspd,
                -apply_deadzone(js.get_axis(4)) * aspd,
                0.0,
            ]),
        ),
        "legend": [
            "L-Stick Vert:  Fwd / Back",
            "L-Stick Horiz: Side",
            "R-Stick Horiz: Roll",
            "R-Stick Vert:  Pitch",
            "L2 / R2:       Up / Down",
            "L1 / R1:       Yaw",
            "Circle / X:    Gripper Open/Close",
        ],
    },
}

INPUT_METHODS = {
    "controller": "Controller",
    "keyboard":   "Keyboard",
}

KEYBOARD_LEGEND = [
    "W / S:         Fwd / Back",
    "A / D:         Side",
    "Space / LShift: Up / Down",
    "Left / Right:  Roll (inverted)",
    "Up / Down:     Pitch",
    "Q / E:         Yaw",
    "O / P:         Gripper",
]


# ---------------------------------------------------------------------------
# DEMO FUNCTIONS
# ---------------------------------------------------------------------------

_DEMO_JOINT_SPEED = 0.1
_DEMO_LIN_SPEED   = 0.03
_DEMO_LIN_ACCEL   = 0.03


def _demo_pick_and_place(robot):
    START_X = 0.25;  START_Y = 0.5
    END_X   = 0.25;  END_Y   = 0.2
    PICK_PLACE_HEIGHT = 0.03;  TRAVEL_HEIGHT = 0.1;  OBJECT_SIZE = 0.018

    robot.joint_space.speed                 = _DEMO_JOINT_SPEED
    robot.cartesian_space.linear_speed      = _DEMO_LIN_SPEED
    robot.cartesian_space.acceleration      = _DEMO_LIN_ACCEL

    robot.tool_changer.attach_tool(robot.tools.gripper)
    robot.cartesian_space.move(CartesianSpace.Pose((START_X, START_Y, TRAVEL_HEIGHT), (0, 0, 0)), False)
    robot.tools.gripper.set_distance(0.05)
    time.sleep(1)
    robot.cartesian_space.move(CartesianSpace.Pose((START_X, START_Y, PICK_PLACE_HEIGHT), (0, 0, 0)))
    robot.tools.gripper.set_distance(OBJECT_SIZE)
    time.sleep(1)
    robot.cartesian_space.move(CartesianSpace.Pose((START_X, START_Y, TRAVEL_HEIGHT), (0, 0, 0)))
    robot.cartesian_space.move(CartesianSpace.Pose((END_X, END_Y, TRAVEL_HEIGHT), (0, 0, 0)))
    robot.cartesian_space.move(CartesianSpace.Pose((END_X, END_Y, PICK_PLACE_HEIGHT), (0, 0, 0)))
    robot.tools.gripper.set_distance(0.05)
    time.sleep(1)
    robot.cartesian_space.move(CartesianSpace.Pose((END_X, END_Y, TRAVEL_HEIGHT), (0, 0, 0)))


def _demo_sine(robot):
    X0 = 0.175;  Y0 = 0.2;  Z0 = 0.1
    WAVE_LENGTH = 0.3;  NUM_PERIODS = 2;  WAVE_AMPLITUDE = 0.03;  NUM_POINTS = 50

    robot.joint_space.speed            = _DEMO_JOINT_SPEED
    robot.cartesian_space.linear_speed = _DEMO_LIN_SPEED
    robot.cartesian_space.acceleration = _DEMO_LIN_ACCEL

    robot.cartesian_space.move(CartesianSpace.Pose((X0, Y0, Z0), (0, 0, 0)), False)

    total_distance = 0.0
    for i in range(1000):
        a1, a2 = i / 1000, (i + 1) / 1000
        x1 = X0 + WAVE_AMPLITUDE * np.sin(2 * np.pi * a1 * NUM_PERIODS)
        x2 = X0 + WAVE_AMPLITUDE * np.sin(2 * np.pi * a2 * NUM_PERIODS)
        total_distance += np.sqrt((x2 - x1) ** 2 + (WAVE_LENGTH / 1000) ** 2)

    t_accel = _DEMO_LIN_SPEED / _DEMO_LIN_ACCEL
    d_accel = 0.5 * _DEMO_LIN_ACCEL * t_accel ** 2
    if 2 * d_accel > total_distance:
        t_accel = np.sqrt(total_distance / _DEMO_LIN_ACCEL)
        t_cruise = 0;  d_accel = 0.5 * total_distance;  d_cruise = 0
        actual_max_speed = _DEMO_LIN_ACCEL * t_accel
    else:
        d_cruise = total_distance - 2 * d_accel
        t_cruise = d_cruise / _DEMO_LIN_SPEED
        actual_max_speed = _DEMO_LIN_SPEED
    t_decel = t_accel
    total_time = t_accel + t_cruise + t_decel

    def trap(t):
        if t <= t_accel:
            s = 0.5 * _DEMO_LIN_ACCEL * t ** 2
        elif t <= t_accel + t_cruise:
            s = d_accel + actual_max_speed * (t - t_accel)
        elif t <= total_time:
            td = t - t_accel - t_cruise
            s = d_accel + d_cruise + actual_max_speed * td - 0.5 * _DEMO_LIN_ACCEL * td ** 2
        else:
            s = total_distance
        return s / total_distance

    path = CartesianSpace.Path()
    for i in range(NUM_POINTS):
        t = (i / (NUM_POINTS - 1)) * total_time
        alpha = trap(t)
        y = Y0 + alpha * WAVE_LENGTH
        x = X0 + WAVE_AMPLITUDE * np.sin(2 * np.pi * alpha * NUM_PERIODS)
        path.add(CartesianSpace.Pose(position=(x, y, Z0), orientation=(0, 0, 0), time_from_start=t))
    robot.cartesian_space.follow_path(path)


def _demo_circle(robot):
    X0 = 0.2;  Y0 = 0.35;  Z0 = 0.1;  RADIUS = 0.07;  NUM_POINTS = 60

    robot.joint_space.speed            = _DEMO_JOINT_SPEED
    robot.cartesian_space.linear_speed = _DEMO_LIN_SPEED
    robot.cartesian_space.acceleration = _DEMO_LIN_ACCEL

    robot.cartesian_space.move(CartesianSpace.Pose((X0 + RADIUS, Y0, Z0), (0, 0, 0)), True)

    total_distance = 2 * math.pi * RADIUS
    t_accel = _DEMO_LIN_SPEED / _DEMO_LIN_ACCEL
    d_accel = 0.5 * _DEMO_LIN_ACCEL * t_accel ** 2
    if 2 * d_accel > total_distance:
        t_accel = math.sqrt(total_distance / _DEMO_LIN_ACCEL)
        t_cruise = 0;  d_accel = 0.5 * total_distance;  d_cruise = 0
        actual_max_speed = _DEMO_LIN_ACCEL * t_accel
    else:
        d_cruise = total_distance - 2 * d_accel
        t_cruise = d_cruise / _DEMO_LIN_SPEED
        actual_max_speed = _DEMO_LIN_SPEED
    t_decel = t_accel
    total_time = t_accel + t_cruise + t_decel

    def trap(t):
        if t <= t_accel:
            s = 0.5 * _DEMO_LIN_ACCEL * t ** 2
        elif t <= t_accel + t_cruise:
            s = d_accel + actual_max_speed * (t - t_accel)
        elif t <= total_time:
            td = t - t_accel - t_cruise
            s = d_accel + d_cruise + actual_max_speed * td - 0.5 * _DEMO_LIN_ACCEL * td ** 2
        else:
            s = total_distance
        return s / total_distance

    path = CartesianSpace.Path()
    for i in range(NUM_POINTS):
        t = (i / (NUM_POINTS - 1)) * total_time
        alpha = trap(t)
        angle = 2 * math.pi * alpha
        x = X0 + RADIUS * math.cos(angle)
        y = Y0 + RADIUS * math.sin(angle)
        path.add(CartesianSpace.Pose(position=(x, y, Z0), orientation=(0, 0, 0), time_from_start=t))
    robot.cartesian_space.follow_path(path)


def _demo_cone_motion(robot):
    POSITION   = (0.25, 0.35, 0.1)
    CONE_ANGLE = math.radians(15)
    PERIOD     = 10.0
    ROTATIONS  = 2

    robot.joint_space.speed = _DEMO_JOINT_SPEED
    robot.cartesian_space.move(CartesianSpace.Pose(POSITION, (0.0, CONE_ANGLE, 0.0)), enforce_linearity=False)

    omega      = 2 * math.pi / PERIOD
    total_time = PERIOD * ROTATIONS
    t_start    = time.time()
    while True:
        t = time.time() - t_start
        if t >= total_time:
            break
        theta     = omega * t
        pitch     =  CONE_ANGLE * math.cos(theta)
        roll_dot  =  CONE_ANGLE * omega * math.cos(theta)
        pitch_dot = -CONE_ANGLE * omega * math.sin(theta)
        cp, sp = math.cos(pitch), math.sin(pitch)
        robot.cartesian_space.twist([0.0, 0.0, 0.0], [-cp * roll_dot, -pitch_dot, -sp * roll_dot])
        time.sleep(0.01)
    robot.cartesian_space.twist([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


_DEMO_MAP = {
    "pick_and_place": _demo_pick_and_place,
    "sine":           _demo_sine,
    "circle":         _demo_circle,
    "cone_motion":    _demo_cone_motion,
}


def apply_deadzone(value, deadzone=0.1):
    return value if abs(value) > deadzone else 0.0


def clamp(value, low, high):
    return max(low, min(value, high))


def find_current_tool_name(tool_map, current_tool):
    for name, tool in tool_map.items():
        if tool is current_tool:
            return name
    return "none"


def limit_acceleration(current, target, max_accel, dt):
    max_delta = max_accel * dt
    if max_delta <= 0.0:
        return current
    return current + np.clip(target - current, -max_delta, max_delta)


def read_button(joystick, button_idx):
    return joystick.get_numbuttons() > button_idx and joystick.get_button(button_idx)


def read_trigger(joystick, axis_idx, button_idx):
    trigger = 0.0
    if joystick.get_numaxes() > axis_idx:
        raw = joystick.get_axis(axis_idx)
        trigger = (raw + 1.0) / 2.0 if raw < -0.05 else raw
        trigger = max(0.0, trigger)
    if trigger <= 0.0 and read_button(joystick, button_idx):
        trigger = 1.0
    return trigger


def apply_mapping(joystick, mapping_key, l_spd, a_spd):
    lin_vel, ang_vel = MAPPINGS[mapping_key]["apply"](joystick, l_spd, a_spd)
    if mapping_key == "drone":
        dpad = joystick.get_hat(0)
        if dpad[1] != 0:
            ang_vel[1] = dpad[1] * a_spd
        if dpad[0] != 0:
            ang_vel[0] = dpad[0] * a_spd
    elif mapping_key == "standard":
        l2 = read_trigger(joystick, 2, 6)
        r2 = read_trigger(joystick, 5, 7)
        lin_vel[2] = (l2 - r2) * l_spd
        ang_vel[2] = (float(read_button(joystick, 5)) - float(read_button(joystick, 4))) * a_spd
    return lin_vel, ang_vel


class TextInput:
    def __init__(self, text="0.0000", numeric_only=True):
        self.text = text
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.focused = False
        self.cursor = len(text)
        self.numeric_only = numeric_only

    def handle_key(self, event):
        if event.key == pygame.K_BACKSPACE:
            if self.cursor > 0:
                self.text = self.text[:self.cursor - 1] + self.text[self.cursor:]
                self.cursor -= 1
        elif event.key == pygame.K_DELETE:
            if self.cursor < len(self.text):
                self.text = self.text[:self.cursor] + self.text[self.cursor + 1:]
        elif event.key == pygame.K_LEFT:
            self.cursor = max(0, self.cursor - 1)
        elif event.key == pygame.K_RIGHT:
            self.cursor = min(len(self.text), self.cursor + 1)
        elif event.key == pygame.K_HOME:
            self.cursor = 0
        elif event.key == pygame.K_END:
            self.cursor = len(self.text)
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_ESCAPE):
            self.focused = False
        elif self.numeric_only and event.unicode in "0123456789.-":
            self.text = self.text[:self.cursor] + event.unicode + self.text[self.cursor:]
            self.cursor += 1
        elif not self.numeric_only and event.unicode and event.unicode.isprintable():
            self.text = self.text[:self.cursor] + event.unicode + self.text[self.cursor:]
            self.cursor += 1

    def get_float(self, default=0.0):
        try:
            return float(self.text)
        except ValueError:
            return default

    def draw(self, surf, font, br=4):
        bg = (30, 35, 55) if self.focused else (20, 22, 34)
        border = ACCENT_COLOR if self.focused else BORDER_COLOR
        pygame.draw.rect(surf, bg, self.rect, border_radius=br)
        pygame.draw.rect(surf, border, self.rect, 1, border_radius=br)
        img = font.render(self.text, True, TEXT_COLOR)
        surf.blit(img, (self.rect.x + 6, self.rect.centery - img.get_height() // 2))
        if self.focused:
            pre_w = font.size(self.text[:self.cursor])[0]
            cur_x = self.rect.x + 6 + pre_w
            pygame.draw.line(surf, ACCENT_COLOR,
                             (cur_x, self.rect.y + 4), (cur_x, self.rect.bottom - 4))


# --- DRAW HELPERS ---

def draw_panel(surf, rect, br=12):
    pygame.draw.rect(surf, PANEL_BG, rect, border_radius=br)
    pygame.draw.rect(surf, BORDER_COLOR, rect, 1, border_radius=br)


def draw_card(surf, rect, br=8):
    pygame.draw.rect(surf, CARD_BG, rect, border_radius=br)
    pygame.draw.rect(surf, BORDER_COLOR, rect, 1, border_radius=br)


def draw_sep(surf, x, y, w):
    pygame.draw.line(surf, BORDER_COLOR, (x, y), (x + w, y))


def draw_btn(surf, font, rect, text, hovered, active=False, danger=False, attach=False, br=6):
    if danger:
        bg = DANGER_HOVER if hovered else DANGER_BG
        border = DANGER_BORDER
    elif attach:
        bg = ATTACH_HOVER if hovered else ATTACH_BG
        border = ATTACH_BORDER
    elif active:
        bg = (0, 105, 185) if not hovered else BTN_ACTIVE
        border = BTN_ACTIVE
    else:
        bg = HOVER_COLOR if hovered else BTN_COLOR
        border = BORDER_COLOR
    pygame.draw.rect(surf, bg, rect, border_radius=br)
    pygame.draw.rect(surf, border, rect, 1, border_radius=br)
    img = font.render(text, True, TEXT_COLOR)
    surf.blit(img, (rect.centerx - img.get_width() // 2, rect.centery - img.get_height() // 2))


def draw_icon_btn(surf, rect, hovered, icon, danger=False, active=False, br=6):
    if danger:
        bg = DANGER_HOVER if hovered else DANGER_BG
        border = DANGER_BORDER
        color = (255, 190, 180)
    elif active:
        bg = (0, 105, 185) if not hovered else BTN_ACTIVE
        border = BTN_ACTIVE
        color = TEXT_COLOR
    else:
        bg = HOVER_COLOR if hovered else BTN_COLOR
        border = BORDER_COLOR
        color = DIM_COLOR
    pygame.draw.rect(surf, bg, rect, border_radius=br)
    pygame.draw.rect(surf, border, rect, 1, border_radius=br)
    cx, cy = rect.centerx, rect.centery
    pad = max(5, rect.w // 4)
    lw = max(1, rect.w // 14)
    if icon == "delete":
        # X mark
        x0, y0 = rect.x + pad, rect.y + pad
        x1, y1 = rect.right - pad, rect.bottom - pad
        pygame.draw.line(surf, color, (x0, y0), (x1, y1), lw + 1)
        pygame.draw.line(surf, color, (x1, y0), (x0, y1), lw + 1)
    elif icon == "edit":
        sz = max(4, min(rect.w, rect.h) // 2 - pad)
        lw2 = max(2, lw + 1)
        pygame.draw.line(surf, color, (cx - sz, cy + sz), (cx + sz, cy - sz), lw2)
        nib = max(2, sz // 3)
        pygame.draw.polygon(surf, color, [
            (cx + sz,       cy - sz),
            (cx + sz + nib, cy - sz - nib),
            (cx + sz + nib, cy - sz),
        ])


def draw_dropdown(surf, font, rect, text, is_open, hovered, br=6):
    bg = HOVER_COLOR if hovered else BTN_COLOR
    border = ACCENT_COLOR if is_open else BORDER_COLOR
    pygame.draw.rect(surf, bg, rect, border_radius=br)
    pygame.draw.rect(surf, border, rect, 1, border_radius=br)
    img = font.render(text, True, TEXT_COLOR)
    surf.blit(img, (rect.x + 10, rect.centery - img.get_height() // 2))
    arrow = font.render("▲" if is_open else "▼", True, DIM_COLOR)
    surf.blit(arrow, (rect.right - arrow.get_width() - 10, rect.centery - arrow.get_height() // 2))


def draw_dropdown_menu(surf, font, anchor_rect, options, selected, hovered_pos, br=6):
    item_h = anchor_rect.height
    for idx, text in enumerate(options):
        opt = pygame.Rect(anchor_rect.x, anchor_rect.bottom + idx * item_h, anchor_rect.width, item_h)
        is_sel = idx == selected
        bg = (45, 48, 68) if is_sel else (HOVER_COLOR if opt.collidepoint(hovered_pos) else (22, 22, 32))
        pygame.draw.rect(surf, bg, opt, border_radius=br)
        pygame.draw.rect(surf, BORDER_COLOR, opt, 1, border_radius=br)
        img = font.render(text, True, TEXT_COLOR if not is_sel else ACCENT_COLOR)
        surf.blit(img, (opt.x + 10, opt.centery - img.get_height() // 2))


POSITIONS_FILE = os.path.expanduser("~/.robot_saved_positions.json")


def load_positions():
    try:
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def save_positions_to_file(positions):
    try:
        with open(POSITIONS_FILE, "w") as f:
            json.dump(positions, f, indent=2)
    except Exception:
        pass


def draw_section_label(surf, font, text, x, y):
    bar_h = max(font.get_height() - 4, 6)
    pygame.draw.rect(surf, HEADER_COLOR, pygame.Rect(x, y + 2, 3, bar_h), border_radius=2)
    img = font.render(text, True, HEADER_COLOR)
    surf.blit(img, (x + 9, y))


def main():
    pygame.init()
    pygame.joystick.init()

    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Robot Command Center")
    _icon_path = os.path.join(os.path.dirname(__file__), "robot.png")
    if os.path.exists(_icon_path):
        pygame.display.set_icon(pygame.image.load(_icon_path))

    last_s = 0.0
    font       = pygame.font.SysFont("Consolas", 16)
    ui_font    = pygame.font.SysFont("Arial", 16)
    label_font = pygame.font.SysFont("Arial", 20, bold=True)
    sec_font   = pygame.font.SysFont("Arial", 12, bold=True)
    clock = pygame.time.Clock()

    l_spd          = LIN_SPD_DEFAULT
    a_spd          = ANG_SPD_DEFAULT
    max_lin_accel  = LIN_ACCEL_DEFAULT
    max_ang_accel  = ANG_ACCEL_DEFAULT
    joint_vel      = JNT_VEL_DEFAULT
    gripper_speed  = GRIPPER_SPD_DEFAULT
    fake_mode      = args.fake_hardware
    orientation_lock = False
    idle_mode = False
    gripper_pos    = 0.0
    last_sent_gripper_pos = None
    current_lin_vel = np.zeros(3)
    current_ang_vel = np.zeros(3)

    mapping_keys      = list(MAPPINGS.keys())
    input_method_keys = list(INPUT_METHODS.keys())
    tool_map = {
        name: value
        for name in dir(robot.tools)
        for value in [getattr(robot.tools, name)]
        if not name.startswith("_") and isinstance(value, robot.tools.Tool)
    }
    tool_names = list(tool_map.keys())

    selected_mapping      = "standard"
    selected_input_method = "controller" if joystick else "keyboard"
    input_dropdown_open   = False
    mapping_dropdown_open = False
    tool_dropdown_open    = False
    selected_tool_idx     = tool_names.index("gripper") if "gripper" in tool_names else 0
    tool_status           = "Ready"
    grip_input            = TextInput(f"{GRIPPER_MIN:.4f}")

    try:
        robot.tool_changer.attach_tool(robot.tools.gripper)
    except Exception as e:
        robot.node.get_logger().warn(f"Failed to attach gripper: {e}")

    motion_mode       = "cartesian"
    enforce_linearity = True
    move_inputs   = [TextInput() for _ in range(6)]
    focused_input = None
    move_thread   = None
    move_status   = ""
    demo_thread   = None
    demo_status   = ""

    saved_positions = load_positions()
    save_name_input = TextInput("pos_1", numeric_only=False)
    saved_scroll    = 0
    editing_idx     = None  # index of saved position being edited, or None for new

    def do_move():
        nonlocal move_status
        move_status = "Moving..."
        try:
            vals = [inp.get_float() for inp in move_inputs]
            if motion_mode == "cartesian":
                robot.cartesian_space.max_linear_velocity    = l_spd
                robot.cartesian_space.max_angular_velocity   = a_spd
                robot.cartesian_space.max_linear_acceleration = max_lin_accel
                ok = robot.cartesian_space.move(CartesianSpace.Pose(vals[:3], vals[3:]), enforce_linearity=enforce_linearity)
            else:
                robot.joint_space.max_joint_velocity = joint_vel
                ok = robot.joint_space.move(JointSpace.Point(vals))
            move_status = "Done" if ok else "Failed"
        except Exception as e:
            move_status = f"Error: {e}"

    def do_demo(fn, label):
        nonlocal demo_status, idle_mode
        demo_status = f"Running: {label}"
        prev_idle = idle_mode
        idle_mode = True
        try:
            robot.set_idle_mode(True)
        except Exception:
            pass
        try:
            fn(robot)
            demo_status = "Done"
        except Exception as e:
            demo_status = f"Error: {str(e)[:40]}"
        finally:
            idle_mode = prev_idle
            try:
                robot.set_idle_mode(prev_idle)
            except Exception:
                pass

    while True:
        screen.fill(BG_COLOR)
        dt_sec = max(clock.get_time() / 1000.0, 1.0 / 240.0)
        target_lin_vel = np.zeros(3)
        target_ang_vel = np.zeros(3)

        win_w, win_h = screen.get_size()
        s = min(win_w / WIDTH, win_h / HEIGHT)
        def sc(v): return int(v * s)

        if abs(s - last_s) > 0.005:
            font       = pygame.font.SysFont("Consolas", max(8, sc(16)))
            ui_font    = pygame.font.SysFont("Arial",    max(8, sc(16)))
            label_font = pygame.font.SysFont("Arial",    max(9, sc(20)), bold=True)
            sec_font   = pygame.font.SysFont("Arial",    max(7, sc(12)), bold=True)
            last_s = s

        # --- LAYOUT ---
        panel_h = win_h - sc(LP_Y) * 2
        lp = pygame.Rect(sc(LP_X), sc(LP_Y), sc(LP_W), panel_h)
        rp_x = lp.right + sc(LP_X)
        rp = pygame.Rect(rp_x, sc(LP_Y), win_w - rp_x - sc(LP_X), panel_h)
        cx = lp.x + sc(PAD)
        cw = lp.w - sc(PAD) * 2
        row_h  = sc(34)
        row_gap = sc(7)
        br = sc(6)

        # Control section
        ctrl_y      = lp.y + sc(16)
        hw_rect          = pygame.Rect(cx, ctrl_y + sc(20), cw, row_h)
        orient_lock_rect = pygame.Rect(cx, hw_rect.bottom + row_gap, cw, row_h)
        idle_mode_rect   = pygame.Rect(cx, orient_lock_rect.bottom + row_gap, cw, row_h)
        input_rect       = pygame.Rect(cx, idle_mode_rect.bottom + row_gap, cw, row_h)
        if selected_input_method == "controller":
            mapping_rect = pygame.Rect(cx, input_rect.bottom + row_gap, cw, row_h)
            legend_y = mapping_rect.bottom + sc(12)
        else:
            mapping_rect = None
            legend_y = input_rect.bottom + sc(12)

        legend_lines = (
            MAPPINGS[selected_mapping]["legend"] if selected_input_method == "controller"
            else KEYBOARD_LEGEND
        )
        legend_h = sc(23)

        if selected_input_method == "controller" and joystick is None:
            warn_y = legend_y
            legend_y += sc(26)
        else:
            warn_y = None

        legend_end_y = legend_y + len(legend_lines) * legend_h

        # Kinematics section
        sep1_y  = legend_end_y + sc(14)
        kin_y   = sep1_y + sc(14)
        kin_row_h  = sc(38)
        kin_row_gap = sc(8)
        kin_btn_w  = sc(34)
        kin_data = [
            ("Lin Vel",  f"{l_spd:.3f} m/s"),
            ("Ang Vel",  f"{a_spd:.3f} rad/s"),
            ("Lin Acc",  f"{max_lin_accel:.3f} m/s²"),
            ("Ang Acc",  f"{max_ang_accel:.3f} rad/s²"),
            ("Jnt Vel",  f"{joint_vel:.2f} rad/s"),
            ("Grp Spd",  f"{gripper_speed:.3f} m/s"),
        ]
        kin_rows_y    = [kin_y + sc(22) + i * (kin_row_h + kin_row_gap) for i in range(6)]
        kin_dec_rects = [pygame.Rect(cx,                  y, kin_btn_w, kin_row_h) for y in kin_rows_y]
        kin_inc_rects = [pygame.Rect(cx + cw - kin_btn_w, y, kin_btn_w, kin_row_h) for y in kin_rows_y]
        kin_end_y = kin_rows_y[-1] + kin_row_h

        # Tool changer section
        sep2_y       = kin_end_y + sc(14)
        tool_sec_y   = sep2_y + sc(14)
        tool_row_h   = sc(36)
        tool_dd_w    = int(cw * 0.54)
        tool_btn_w   = cw - tool_dd_w - sc(7)
        tool_row1_y  = tool_sec_y + sc(22)
        tool_dd_rect     = pygame.Rect(cx,                              tool_row1_y, tool_dd_w, tool_row_h)
        tool_attach_rect = pygame.Rect(cx + tool_dd_w + sc(7),          tool_row1_y, tool_btn_w, tool_row_h)
        tool_detach_rect = pygame.Rect(cx, tool_dd_rect.bottom + sc(7), cw,          tool_row_h)
        tool_status_y    = tool_detach_rect.bottom + sc(10)
        grip_inp_y       = tool_status_y + sc(28)
        grip_inp_lbl_w   = font.size("Gripper")[0] + sc(10)
        grip_inp_w       = int(cw * 0.44)
        grip_set_w       = cw - grip_inp_lbl_w - grip_inp_w - sc(7)
        grip_inp_h       = sc(32)
        grip_input.rect  = pygame.Rect(cx + grip_inp_lbl_w, grip_inp_y, grip_inp_w, grip_inp_h)
        grip_set_rect    = pygame.Rect(cx + grip_inp_lbl_w + grip_inp_w + sc(7), grip_inp_y, grip_set_w, grip_inp_h)

        # Telemetry card (top of right panel)
        tel_card = pygame.Rect(rp.x + sc(14), rp.y + sc(14), rp.w - sc(28), sc(310))
        btn_card = pygame.Rect(rp.x + sc(14), tel_card.bottom + sc(14), rp.w - sc(28),
                               rp.bottom - tel_card.bottom - sc(28))

        # Motion target section (inside btn_card) — vertically centred
        mcx = btn_card.x + sc(14)
        mcw = btn_card.w - sc(28)
        inp_h     = sc(30)
        inp_gap   = sc(5)
        inp_lbl_w = sc(46)
        col_w     = (mcw - sc(10)) // 2
        fld_w     = col_w - inp_lbl_w - sc(4)

        inputs_h = 3 * inp_h + 2 * inp_gap
        y_start  = btn_card.y + sc(14)

        motion_mode_y     = y_start + sc(22) + sc(10)
        motion_mode_btn_w = (mcw - sc(7)) // 2
        cart_mode_rect    = pygame.Rect(mcx, motion_mode_y, motion_mode_btn_w, row_h)
        joint_mode_rect   = pygame.Rect(
            mcx + motion_mode_btn_w + sc(7), motion_mode_y,
            mcw - motion_mode_btn_w - sc(7), row_h,
        )
        inp_y0 = cart_mode_rect.bottom + sc(8)
        for i, inp in enumerate(move_inputs):
            col = i // 3
            row = i % 3
            inp.rect = pygame.Rect(
                mcx + col * (col_w + sc(10)) + inp_lbl_w + sc(4),
                inp_y0 + row * (inp_h + inp_gap),
                fld_w, inp_h,
            )
        if motion_mode == "cartesian":
            lin_toggle_y    = inp_y0 + inputs_h + sc(8)
            lin_toggle_rect = pygame.Rect(mcx, lin_toggle_y, mcw, row_h)
            move_btn_y      = lin_toggle_y + row_h + sc(7)
        else:
            lin_toggle_rect = None
            move_btn_y      = inp_y0 + inputs_h + sc(8)
        move_btn_w_px = int(mcw * 0.62)
        use_cur_w     = mcw - move_btn_w_px - sc(7)
        move_btn_rect = pygame.Rect(mcx, move_btn_y, move_btn_w_px, row_h)
        use_cur_rect  = pygame.Rect(mcx + move_btn_w_px + sc(7), move_btn_y, use_cur_w, row_h)
        motion_status_y = move_btn_y + row_h + sc(8)

        # Demo section (inside btn_card, below move status)
        demo_sep_y    = motion_status_y + sc(22)
        demo_sec_y    = demo_sep_y + sc(14)
        demo_btn_h    = sc(30)
        demo_half_w   = (mcw - sc(7)) // 2
        demo_btn_y0   = demo_sec_y + sc(20)
        demo_btn_y1   = demo_btn_y0 + demo_btn_h + sc(6)
        demo_rects    = [
            pygame.Rect(mcx,                       demo_btn_y0, demo_half_w,                demo_btn_h),
            pygame.Rect(mcx + demo_half_w + sc(7), demo_btn_y0, mcw - demo_half_w - sc(7), demo_btn_h),
            pygame.Rect(mcx,                       demo_btn_y1, demo_half_w,                demo_btn_h),
            pygame.Rect(mcx + demo_half_w + sc(7), demo_btn_y1, mcw - demo_half_w - sc(7), demo_btn_h),
        ]
        demo_status_y = demo_btn_y1 + demo_btn_h + sc(8)

        # Saved positions section (inside btn_card, below demo section)
        saved_sec_y      = demo_status_y + sc(18)
        saved_row_h      = sc(30)
        save_name_w      = int(mcw * 0.55)
        save_btns_w      = mcw - save_name_w - sc(7)
        save_name_input.rect = pygame.Rect(mcx, saved_sec_y + sc(22), save_name_w, saved_row_h)
        save_btn_w      = (save_btns_w - sc(5)) // 2
        cancel_btn_w    = save_btns_w - save_btn_w - sc(5)
        save_btn_rect   = pygame.Rect(mcx + save_name_w + sc(7), saved_sec_y + sc(22), save_btn_w, saved_row_h)
        cancel_btn_rect = pygame.Rect(save_btn_rect.right + sc(5), saved_sec_y + sc(22), cancel_btn_w, saved_row_h)
        save_btn_full_rect = pygame.Rect(mcx + save_name_w + sc(7), saved_sec_y + sc(22), save_btns_w, saved_row_h)
        saved_list_y     = save_name_input.rect.bottom + sc(6)
        saved_item_h     = sc(32)
        saved_go_w       = sc(44)
        saved_edt_w      = sc(50)
        saved_del_w      = sc(36)
        saved_name_col_w = mcw - saved_go_w - saved_edt_w - saved_del_w - sc(21) - sc(12)
        saved_area       = pygame.Rect(mcx, saved_list_y, mcw, btn_card.bottom - sc(14) - saved_list_y)
        saved_visible    = max(1, saved_area.h // saved_item_h)

        mouse_pos = pygame.mouse.get_pos()

        # --- EVENT HANDLING ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            ep = event.pos if hasattr(event, "pos") else None
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                hc = False

                # Close open dropdowns — check for option selection first
                if input_dropdown_open:
                    input_dropdown_open = False
                    for idx, key in enumerate(input_method_keys):
                        opt = pygame.Rect(input_rect.x, input_rect.bottom + idx * row_h, input_rect.w, row_h)
                        if opt.collidepoint(ep):
                            selected_input_method = key
                            hc = True
                            break

                elif mapping_dropdown_open:
                    mapping_dropdown_open = False
                    if mapping_rect:
                        for idx, key in enumerate(mapping_keys):
                            opt = pygame.Rect(mapping_rect.x, mapping_rect.bottom + idx * row_h, mapping_rect.w, row_h)
                            if opt.collidepoint(ep):
                                selected_mapping = key
                                hc = True
                                break

                elif tool_dropdown_open:
                    tool_dropdown_open = False
                    for idx, name in enumerate(tool_names):
                        opt = pygame.Rect(tool_dd_rect.x, tool_dd_rect.bottom + idx * tool_row_h, tool_dd_rect.w, tool_row_h)
                        if opt.collidepoint(ep):
                            selected_tool_idx = idx
                            hc = True
                            break

                if not hc:
                    if hw_rect.collidepoint(ep):
                        fake_mode = not fake_mode
                        robot.set_fake_hardware_mode(fake_mode)
                        hc = True
                    elif orient_lock_rect.collidepoint(ep):
                        orientation_lock = not orientation_lock
                        hc = True
                    elif idle_mode_rect.collidepoint(ep):
                        idle_mode = not idle_mode
                        robot.set_idle_mode(idle_mode)
                        hc = True
                    elif input_rect.collidepoint(ep):
                        input_dropdown_open = not input_dropdown_open
                        mapping_dropdown_open = False
                        tool_dropdown_open = False
                        hc = True
                    elif mapping_rect and mapping_rect.collidepoint(ep):
                        mapping_dropdown_open = not mapping_dropdown_open
                        input_dropdown_open = False
                        tool_dropdown_open = False
                        hc = True
                    elif kin_dec_rects[0].collidepoint(ep):
                        l_spd = clamp(l_spd - LIN_SPD_STEP, LIN_SPD_MIN, LIN_SPD_MAX); hc = True
                    elif kin_inc_rects[0].collidepoint(ep):
                        l_spd = clamp(l_spd + LIN_SPD_STEP, LIN_SPD_MIN, LIN_SPD_MAX); hc = True
                    elif kin_dec_rects[1].collidepoint(ep):
                        a_spd = clamp(a_spd - ANG_SPD_STEP, ANG_SPD_MIN, ANG_SPD_MAX); hc = True
                    elif kin_inc_rects[1].collidepoint(ep):
                        a_spd = clamp(a_spd + ANG_SPD_STEP, ANG_SPD_MIN, ANG_SPD_MAX); hc = True
                    elif kin_dec_rects[2].collidepoint(ep):
                        max_lin_accel = clamp(max_lin_accel - LIN_ACCEL_STEP, LIN_ACCEL_MIN, LIN_ACCEL_MAX); hc = True
                    elif kin_inc_rects[2].collidepoint(ep):
                        max_lin_accel = clamp(max_lin_accel + LIN_ACCEL_STEP, LIN_ACCEL_MIN, LIN_ACCEL_MAX); hc = True
                    elif kin_dec_rects[3].collidepoint(ep):
                        max_ang_accel = clamp(max_ang_accel - ANG_ACCEL_STEP, ANG_ACCEL_MIN, ANG_ACCEL_MAX); hc = True
                    elif kin_inc_rects[3].collidepoint(ep):
                        max_ang_accel = clamp(max_ang_accel + ANG_ACCEL_STEP, ANG_ACCEL_MIN, ANG_ACCEL_MAX); hc = True
                    elif kin_dec_rects[4].collidepoint(ep):
                        joint_vel = clamp(joint_vel - JNT_VEL_STEP, JNT_VEL_MIN, JNT_VEL_MAX); hc = True
                    elif kin_inc_rects[4].collidepoint(ep):
                        joint_vel = clamp(joint_vel + JNT_VEL_STEP, JNT_VEL_MIN, JNT_VEL_MAX); hc = True
                    elif kin_dec_rects[5].collidepoint(ep):
                        gripper_speed = clamp(gripper_speed - GRIPPER_SPD_STEP, GRIPPER_SPD_MIN, GRIPPER_SPD_MAX); hc = True
                    elif kin_inc_rects[5].collidepoint(ep):
                        gripper_speed = clamp(gripper_speed + GRIPPER_SPD_STEP, GRIPPER_SPD_MIN, GRIPPER_SPD_MAX); hc = True
                    elif tool_dd_rect.collidepoint(ep) and tool_names:
                        tool_dropdown_open = not tool_dropdown_open
                        input_dropdown_open = False
                        mapping_dropdown_open = False
                        hc = True
                    elif tool_attach_rect.collidepoint(ep):
                        if tool_names:
                            try:
                                robot.tool_changer.attach_tool(tool_map[tool_names[selected_tool_idx]])
                                tool_status = f"Attached: {tool_names[selected_tool_idx]}"
                            except Exception as e:
                                tool_status = f"Error: {e}"
                        tool_dropdown_open = False
                        hc = True
                    elif tool_detach_rect.collidepoint(ep):
                        try:
                            robot.tool_changer.detach_tool()
                            tool_status = "Detached"
                        except Exception as e:
                            tool_status = f"Error: {e}"
                        hc = True
                    elif cart_mode_rect.collidepoint(ep):
                        if motion_mode != "cartesian":
                            motion_mode = "cartesian"
                            p = robot.cartesian_space.read()
                            vals = list(p.position) + list(p.orientation)
                            for inp, v in zip(move_inputs, vals):
                                inp.text = f"{v:.4f}"
                        hc = True
                    elif joint_mode_rect.collidepoint(ep):
                        if motion_mode != "joint":
                            motion_mode = "joint"
                            p = robot.joint_space.read()
                            vals = list(p.joint_configuration)
                            for inp, v in zip(move_inputs, vals):
                                inp.text = f"{v:.4f}"
                        hc = True
                    elif lin_toggle_rect is not None and lin_toggle_rect.collidepoint(ep) and motion_mode == "cartesian":
                        enforce_linearity = not enforce_linearity
                        hc = True
                    elif move_btn_rect.collidepoint(ep):
                        if move_thread is None or not move_thread.is_alive():
                            move_thread = threading.Thread(target=do_move, daemon=True)
                            move_thread.start()
                        hc = True
                    elif use_cur_rect.collidepoint(ep):
                        if motion_mode == "cartesian":
                            p = robot.cartesian_space.read()
                            vals = list(p.position) + list(p.orientation)
                        else:
                            p = robot.joint_space.read()
                            vals = list(p.joint_configuration)
                        for inp, v in zip(move_inputs, vals):
                            inp.text = f"{v:.4f}"
                        hc = True
                    elif not (demo_thread and demo_thread.is_alive()):
                        _demo_defs = [
                            (demo_rects[0], _DEMO_MAP["pick_and_place"], "Pick & Place"),
                            (demo_rects[1], _DEMO_MAP["sine"],           "Sine Wave"),
                            (demo_rects[2], _DEMO_MAP["circle"],         "Circle"),
                            (demo_rects[3], _DEMO_MAP["cone_motion"],    "Cone Motion"),
                        ]
                        for r, fn, lbl in _demo_defs:
                            if r.collidepoint(ep):
                                demo_thread = threading.Thread(
                                    target=do_demo, args=(fn, lbl), daemon=True
                                )
                                demo_thread.start()
                                hc = True
                                break

            # Text input focus management
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and ep is not None:
                clicked_on_input = False
                for inp in list(move_inputs) + [grip_input, save_name_input]:
                    if inp.rect.collidepoint(ep):
                        clicked_on_input = True
                        if focused_input is not inp:
                            if focused_input:
                                focused_input.focused = False
                            focused_input = inp
                            inp.focused = True
                        break
                if not clicked_on_input and focused_input is not None:
                    focused_input.focused = False
                    focused_input = None
                if ep is not None and grip_set_rect.collidepoint(ep) and find_current_tool_name(tool_map, robot.tool_changer.current_tool) == "gripper":
                    val = float(np.clip(grip_input.get_float(gripper_pos), GRIPPER_MIN, GRIPPER_MAX))
                    grip_input.text = f"{val:.4f}"
                    grip_input.cursor = len(grip_input.text)
                    try:
                        robot.tools.gripper.set_distance(val)
                        gripper_pos = val
                    except RuntimeError:
                        pass

            # Saved positions buttons
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and ep is not None:
                clicked_save = (
                    (editing_idx is not None and save_btn_rect.collidepoint(ep)) or
                    (editing_idx is None and save_btn_full_rect.collidepoint(ep))
                )
                if clicked_save:
                    name = save_name_input.text.strip() or f"pos_{len(saved_positions) + 1}"
                    vals = [inp.get_float() for inp in move_inputs]
                    if editing_idx is not None and 0 <= editing_idx < len(saved_positions):
                        saved_positions[editing_idx] = {"name": name, "mode": motion_mode, "values": vals}
                        editing_idx = None
                    else:
                        saved_positions.append({"name": name, "mode": motion_mode, "values": vals})
                    save_positions_to_file(saved_positions)
                    save_name_input.text = f"pos_{len(saved_positions) + 1}"
                    save_name_input.cursor = len(save_name_input.text)
                elif cancel_btn_rect.collidepoint(ep) and editing_idx is not None:
                    editing_idx = None
                    save_name_input.text = f"pos_{len(saved_positions) + 1}"
                    save_name_input.cursor = len(save_name_input.text)
                elif saved_area.collidepoint(ep):
                    rel_y = ep[1] - saved_list_y
                    idx = saved_scroll + rel_y // saved_item_h
                    if 0 <= idx < len(saved_positions):
                        item_y  = saved_list_y + (idx - saved_scroll) * saved_item_h
                        go_rect  = pygame.Rect(mcx + saved_name_col_w + sc(7), item_y + sc(1), saved_go_w, saved_item_h - sc(2))
                        edt_rect = pygame.Rect(go_rect.right + sc(7), item_y + sc(1), saved_edt_w, saved_item_h - sc(2))
                        del_rect = pygame.Rect(edt_rect.right + sc(7), item_y + sc(1), saved_del_w, saved_item_h - sc(2))
                        if go_rect.collidepoint(ep):
                            for inp, v in zip(move_inputs, saved_positions[idx]["values"]):
                                inp.text = f"{v:.4f}"
                                inp.cursor = len(inp.text)
                            motion_mode = saved_positions[idx]["mode"]
                        elif edt_rect.collidepoint(ep):
                            editing_idx = idx
                            save_name_input.text = saved_positions[idx]["name"]
                            save_name_input.cursor = len(save_name_input.text)
                            for inp, v in zip(move_inputs, saved_positions[idx]["values"]):
                                inp.text = f"{v:.4f}"
                                inp.cursor = len(inp.text)
                            motion_mode = saved_positions[idx]["mode"]
                        elif del_rect.collidepoint(ep):
                            if editing_idx == idx:
                                editing_idx = None
                                save_name_input.text = f"pos_{len(saved_positions)}"
                                save_name_input.cursor = len(save_name_input.text)
                            saved_positions.pop(idx)
                            save_positions_to_file(saved_positions)
                            saved_scroll = max(0, min(saved_scroll, len(saved_positions) - saved_visible))

            # Scroll saved list with mouse wheel
            if event.type == pygame.MOUSEWHEEL and saved_area.collidepoint(mouse_pos):
                saved_scroll = max(0, min(saved_scroll - event.y, max(0, len(saved_positions) - saved_visible)))

            if event.type == pygame.KEYDOWN and focused_input is not None:
                focused_input.handle_key(event)
                if not focused_input.focused:
                    focused_input = None

        # --- INPUT ---
        keys = pygame.key.get_pressed()
        if selected_input_method == "controller":
            if joystick:
                target_lin_vel, target_ang_vel = apply_mapping(joystick, selected_mapping, l_spd, a_spd)
                if selected_mapping == "drone":
                    if read_button(joystick, 4): gripper_pos += gripper_speed * dt_sec
                    if read_button(joystick, 5): gripper_pos -= gripper_speed * dt_sec
                elif selected_mapping == "standard":
                    if read_button(joystick, 1): gripper_pos += gripper_speed * dt_sec  # Circle
                    if read_button(joystick, 0): gripper_pos -= gripper_speed * dt_sec  # X
        elif selected_input_method == "keyboard":
            if focused_input is None:
                if keys[pygame.K_w]:      target_lin_vel[0] += l_spd
                if keys[pygame.K_s]:      target_lin_vel[0] -= l_spd
                if keys[pygame.K_a]:      target_lin_vel[1] += l_spd
                if keys[pygame.K_d]:      target_lin_vel[1] -= l_spd
                if keys[pygame.K_SPACE]:  target_lin_vel[2] += l_spd
                if keys[pygame.K_LSHIFT]: target_lin_vel[2] -= l_spd
                if keys[pygame.K_LEFT]:   target_ang_vel[0] -= a_spd
                if keys[pygame.K_RIGHT]:  target_ang_vel[0] += a_spd
                if keys[pygame.K_UP]:     target_ang_vel[1] += a_spd
                if keys[pygame.K_DOWN]:   target_ang_vel[1] -= a_spd
                if keys[pygame.K_q]:      target_ang_vel[2] += a_spd
                if keys[pygame.K_e]:      target_ang_vel[2] -= a_spd
        if keys[pygame.K_o]: gripper_pos += gripper_speed * dt_sec
        if keys[pygame.K_p]: gripper_pos -= gripper_speed * dt_sec
        gripper_pos = float(np.clip(gripper_pos, GRIPPER_MIN, GRIPPER_MAX))

        if orientation_lock:
            target_ang_vel = np.zeros(3)
        if not idle_mode:
            current_lin_vel = limit_acceleration(current_lin_vel, target_lin_vel, max_lin_accel, dt_sec)
            current_ang_vel = limit_acceleration(current_ang_vel, target_ang_vel, max_ang_accel, dt_sec)
            robot.cartesian_space.twist(tuple(current_lin_vel), tuple(current_ang_vel))
        else:
            current_lin_vel = np.zeros(3)
            current_ang_vel = np.zeros(3)

        if last_sent_gripper_pos is None or abs(gripper_pos - last_sent_gripper_pos) > 1e-4:
            try:
                robot.tools.gripper.set_distance(gripper_pos)
                last_sent_gripper_pos = gripper_pos
            except RuntimeError:
                pass

        # --- DRAW ---
        pose = robot.cartesian_space.read()
        joint_pt = robot.joint_space.read()

        draw_panel(screen, lp, br=sc(10))
        draw_panel(screen, rp, br=sc(10))

        # ── LEFT PANEL ─────────────────────────────────────────────────────
        # Title
        title_img = label_font.render("COMMAND CENTER", True, TEXT_COLOR)
        screen.blit(title_img, (lp.centerx - title_img.get_width() // 2, ctrl_y))

        # CONTROL section
        draw_section_label(screen, sec_font, "CONTROL", cx, hw_rect.y - sc(18))

        # Hardware toggle
        hw_hov = hw_rect.collidepoint(mouse_pos)
        hw_color_bg     = (22, 70, 38) if not fake_mode else (70, 25, 25)
        hw_color_border = (0, 160, 70) if not fake_mode else (180, 50, 50)
        hw_color_text   = (0, 200, 90) if not fake_mode else (220, 70, 70)
        pygame.draw.rect(screen, HOVER_COLOR if hw_hov else hw_color_bg, hw_rect, border_radius=br)
        pygame.draw.rect(screen, hw_color_border, hw_rect, 1, border_radius=br)
        hw_text = "● REAL HARDWARE" if not fake_mode else "● FAKE HARDWARE"
        hw_img = ui_font.render(hw_text, True, hw_color_text)
        screen.blit(hw_img, (hw_rect.centerx - hw_img.get_width() // 2, hw_rect.centery - hw_img.get_height() // 2))

        # Orientation lock toggle
        ol_hov = orient_lock_rect.collidepoint(mouse_pos)
        ol_bg     = (22, 50, 80) if orientation_lock else (40, 25, 65)
        ol_border = (55, 155, 255) if orientation_lock else (120, 60, 180)
        ol_text_c = (80, 190, 255) if orientation_lock else (160, 100, 230)
        pygame.draw.rect(screen, HOVER_COLOR if ol_hov else ol_bg, orient_lock_rect, border_radius=br)
        pygame.draw.rect(screen, ol_border, orient_lock_rect, 1, border_radius=br)
        ol_text = "ORIENTATION LOCKED" if orientation_lock else "ORIENTATION FREE"
        ol_img = ui_font.render(ol_text, True, ol_text_c)
        screen.blit(ol_img, (orient_lock_rect.centerx - ol_img.get_width() // 2,
                              orient_lock_rect.centery - ol_img.get_height() // 2))

        # Idle mode toggle
        im_hov = idle_mode_rect.collidepoint(mouse_pos)
        im_bg     = (70, 45, 10) if idle_mode else (25, 55, 30)
        im_border = (220, 140, 30) if idle_mode else (0, 170, 80)
        im_text_c = (255, 175, 50) if idle_mode else (0, 215, 100)
        pygame.draw.rect(screen, HOVER_COLOR if im_hov else im_bg, idle_mode_rect, border_radius=br)
        pygame.draw.rect(screen, im_border, idle_mode_rect, 1, border_radius=br)
        im_text = "● IDLE" if idle_mode else "● ACTIVE"
        im_img = ui_font.render(im_text, True, im_text_c)
        screen.blit(im_img, (idle_mode_rect.centerx - im_img.get_width() // 2,
                              idle_mode_rect.centery - im_img.get_height() // 2))

        # Input method dropdown
        draw_dropdown(screen, ui_font, input_rect,
                      f"Input:  {INPUT_METHODS[selected_input_method]}",
                      input_dropdown_open, input_rect.collidepoint(mouse_pos), br=br)

        # Mapping dropdown (controller only)
        if selected_input_method == "controller" and mapping_rect:
            draw_dropdown(screen, ui_font, mapping_rect,
                          f"Mapping:  {MAPPINGS[selected_mapping]['label']}",
                          mapping_dropdown_open, mapping_rect.collidepoint(mouse_pos), br=br)

        # Controller warning
        if warn_y is not None:
            warn_img = ui_font.render("⚠  No controller detected", True, (220, 160, 60))
            screen.blit(warn_img, (cx, warn_y))

        # Legend
        for i, line in enumerate(legend_lines):
            color = ACCENT_COLOR if i == len(legend_lines) - 1 else DIM_COLOR
            screen.blit(font.render(line, True, color), (cx, legend_y + i * legend_h))

        # KINEMATICS section
        draw_sep(screen, cx, sep1_y, cw)
        draw_section_label(screen, sec_font, "KINEMATICS", cx, kin_y)

        for i, (label, value) in enumerate(kin_data):
            row_rect = pygame.Rect(cx, kin_rows_y[i], cw, kin_row_h)
            dec_r    = kin_dec_rects[i]
            inc_r    = kin_inc_rects[i]
            pygame.draw.rect(screen, CARD_BG, row_rect, border_radius=br)
            pygame.draw.rect(screen, BORDER_COLOR, row_rect, 1, border_radius=br)
            draw_btn(screen, label_font, dec_r, "−", dec_r.collidepoint(mouse_pos), br=br)
            draw_btn(screen, label_font, inc_r, "+", inc_r.collidepoint(mouse_pos), br=br)
            lbl_img = sec_font.render(label.upper(), True, DIM_COLOR)
            val_img = font.render(value, True, TEXT_COLOR)
            mid_x = dec_r.right + sc(8)
            mid_w = inc_r.left - dec_r.right - sc(16)
            text_y = row_rect.centery - lbl_img.get_height() // 2
            screen.blit(lbl_img, (mid_x, text_y))
            screen.blit(val_img, (mid_x + mid_w - val_img.get_width(),
                                  row_rect.centery - val_img.get_height() // 2))

        # TOOL CHANGER section
        draw_sep(screen, cx, sep2_y, cw)
        draw_section_label(screen, sec_font, "TOOL CHANGER", cx, tool_sec_y)

        selected_tool_name = tool_names[selected_tool_idx] if tool_names else "none"
        current_tool_name  = find_current_tool_name(tool_map, robot.tool_changer.current_tool)

        draw_dropdown(screen, ui_font, tool_dd_rect, selected_tool_name,
                      tool_dropdown_open, tool_dd_rect.collidepoint(mouse_pos), br=br)
        draw_btn(screen, ui_font, tool_attach_rect, "ATTACH",
                 tool_attach_rect.collidepoint(mouse_pos), attach=True, br=br)
        draw_btn(screen, ui_font, tool_detach_rect, "DETACH",
                 tool_detach_rect.collidepoint(mouse_pos), danger=True, br=br)

        # Status row
        cur_img = font.render(f"Current:  {current_tool_name}", True, DIM_COLOR)
        screen.blit(cur_img, (cx, tool_status_y))
        if "Error" in tool_status or "failed" in tool_status.lower():
            st_color = (220, 100, 80)
        elif "Attached" in tool_status or "Detached" in tool_status:
            st_color = (0, 190, 110)
        else:
            st_color = DIM_COLOR
        st_img = font.render(tool_status, True, st_color)
        screen.blit(st_img, (cx + cw - st_img.get_width(), tool_status_y))

        # Gripper position input (only when gripper is attached)
        if current_tool_name == "gripper":
            grip_lbl_img = font.render("gripper", True, DIM_COLOR)
            screen.blit(grip_lbl_img, (cx, grip_inp_y + grip_inp_h // 2 - grip_lbl_img.get_height() // 2))
            grip_input.draw(screen, font, br=sc(4))
            draw_btn(screen, ui_font, grip_set_rect, "SET",
                     grip_set_rect.collidepoint(mouse_pos), br=br)

        # ── RIGHT PANEL ────────────────────────────────────────────────────
        # Telemetry card
        draw_card(screen, tel_card, br=sc(8))
        tel_title = label_font.render("SYSTEM TELEMETRY", True, HEADER_COLOR)
        screen.blit(tel_title, (tel_card.centerx - tel_title.get_width() // 2, tel_card.y + sc(12)))
        draw_sep(screen, tel_card.x + sc(16), tel_card.y + sc(12) + tel_title.get_height() + sc(6),
                 tel_card.w - sc(32))

        c_lines = [
            ("CARTESIAN POSE", label_font, HEADER_COLOR),
            (f"X      {pose.position[0]: .4f} m",    font, TEXT_COLOR),
            (f"Y      {pose.position[1]: .4f} m",    font, TEXT_COLOR),
            (f"Z      {pose.position[2]: .4f} m",    font, TEXT_COLOR),
            (f"Roll   {pose.orientation[0]: .4f} rad", font, TEXT_COLOR),
            (f"Pitch  {pose.orientation[1]: .4f} rad", font, TEXT_COLOR),
            (f"Yaw    {pose.orientation[2]: .4f} rad", font, TEXT_COLOR),
        ]
        j_lines = [("JOINT ANGLES", label_font, HEADER_COLOR)]
        for i, val in enumerate(joint_pt.joint_configuration):
            j_lines.append((f"q{i + 1}     {val: .4f} rad", font, TEXT_COLOR))

        col_y    = tel_card.y + sc(48)
        row_step = sc(32)
        c_col_w  = max(fnt.size(txt)[0] for txt, fnt, _ in c_lines)
        j_col_w  = max(fnt.size(txt)[0] for txt, fnt, _ in j_lines)
        col_gap  = sc(60)
        total_w  = c_col_w + col_gap + j_col_w
        col_x1   = tel_card.centerx - total_w // 2
        col_x2   = col_x1 + c_col_w + col_gap
        for i, (txt, fnt, color) in enumerate(c_lines):
            screen.blit(fnt.render(txt, True, color), (col_x1, col_y + i * row_step))
        for i, (txt, fnt, color) in enumerate(j_lines):
            screen.blit(fnt.render(txt, True, color), (col_x2, col_y + i * row_step))



        # Move to target card
        draw_card(screen, btn_card, br=sc(8))
        mot_title = label_font.render("MOVE TO TARGET", True, HEADER_COLOR)
        screen.blit(mot_title, (btn_card.centerx - mot_title.get_width() // 2, y_start))

        draw_btn(screen, ui_font, cart_mode_rect, "Cartesian",
                 cart_mode_rect.collidepoint(mouse_pos), active=(motion_mode == "cartesian"), br=br)
        draw_btn(screen, ui_font, joint_mode_rect, "Joint",
                 joint_mode_rect.collidepoint(mouse_pos), active=(motion_mode == "joint"), br=br)

        inp_labels = (
            ["X (m)", "Y (m)", "Z (m)", "Roll", "Pitch", "Yaw"]
            if motion_mode == "cartesian" else
            ["q1", "q2", "q3", "q4", "q5", "q6"]
        )
        for i, (inp, lbl) in enumerate(zip(move_inputs, inp_labels)):
            col = i // 3
            row = i % 3
            lbl_x = mcx + col * (col_w + sc(10))
            lbl_y = inp_y0 + row * (inp_h + inp_gap)
            lbl_img = font.render(lbl, True, DIM_COLOR)
            screen.blit(lbl_img, (lbl_x, lbl_y + inp_h // 2 - lbl_img.get_height() // 2))
            inp.draw(screen, font, br=sc(4))

        if motion_mode == "cartesian" and lin_toggle_rect is not None:
            lt_hov = lin_toggle_rect.collidepoint(mouse_pos)
            lt_bg     = (22, 60, 38) if enforce_linearity else (50, 30, 22)
            lt_border = (0, 180, 90)  if enforce_linearity else (200, 110, 40)
            lt_text_c = (0, 210, 110) if enforce_linearity else (230, 140, 60)
            pygame.draw.rect(screen, HOVER_COLOR if lt_hov else lt_bg, lin_toggle_rect, border_radius=br)
            pygame.draw.rect(screen, lt_border, lin_toggle_rect, 1, border_radius=br)
            lt_text = "Enforce Linearity: ON" if enforce_linearity else "Enforce Linearity: OFF"
            lt_img = ui_font.render(lt_text, True, lt_text_c)
            screen.blit(lt_img, (lin_toggle_rect.centerx - lt_img.get_width() // 2,
                                 lin_toggle_rect.centery - lt_img.get_height() // 2))

        moving = move_thread is not None and move_thread.is_alive()
        draw_btn(screen, ui_font, move_btn_rect,
                 "MOVING..." if moving else "MOVE",
                 move_btn_rect.collidepoint(mouse_pos) and not moving,
                 active=not moving, br=br)
        draw_btn(screen, ui_font, use_cur_rect, "USE CURRENT",
                 use_cur_rect.collidepoint(mouse_pos), br=br)

        if move_status:
            if "Error" in move_status or "Failed" in move_status:
                ms_color = (220, 100, 80)
            elif "Done" in move_status:
                ms_color = (0, 190, 110)
            elif "Moving" in move_status:
                ms_color = ACCENT_COLOR
            else:
                ms_color = DIM_COLOR
            ms_img = font.render(move_status, True, ms_color)
            screen.blit(ms_img, (mcx, motion_status_y))

        # ── DEMO SECTION ───────────────────────────────────────────────────
        draw_sep(screen, mcx, demo_sep_y, mcw)
        draw_section_label(screen, sec_font, "DEMO", mcx, demo_sec_y)

        _demo_running = demo_thread is not None and demo_thread.is_alive()
        _demo_labels  = ["Pick & Place", "Sine Wave", "Circle", "Cone Motion"]
        for r, lbl in zip(demo_rects, _demo_labels):
            hov = r.collidepoint(mouse_pos) and not _demo_running
            draw_btn(screen, ui_font, r, lbl, hov, br=br)

        if demo_status:
            if "Error" in demo_status:
                _ds_color = (220, 100, 80)
            elif "Done" in demo_status:
                _ds_color = (0, 190, 110)
            else:
                _ds_color = ACCENT_COLOR
            screen.blit(font.render(demo_status, True, _ds_color), (mcx, demo_status_y))

        # ── SAVED POSITIONS ────────────────────────────────────────────────
        draw_sep(screen, mcx, saved_sec_y - sc(8), mcw)
        draw_section_label(screen, sec_font, "SAVED POSITIONS", mcx, saved_sec_y)
        save_name_input.draw(screen, font, br=sc(4))
        if editing_idx is not None:
            draw_btn(screen, ui_font, save_btn_rect, "UPDATE",
                     save_btn_rect.collidepoint(mouse_pos), active=True, br=br)
            draw_btn(screen, ui_font, cancel_btn_rect, "CANCEL",
                     cancel_btn_rect.collidepoint(mouse_pos), br=br)
        else:
            draw_btn(screen, ui_font, save_btn_full_rect, "SAVE",
                     save_btn_full_rect.collidepoint(mouse_pos), br=br)

        # Clip saved list to its area
        clip_rect = screen.get_clip()
        screen.set_clip(saved_area)
        for i in range(saved_visible):
            idx = saved_scroll + i
            if idx >= len(saved_positions):
                break
            entry   = saved_positions[idx]
            item_y  = saved_list_y + i * saved_item_h
            row_bg   = pygame.Rect(mcx, item_y + sc(1), mcw, saved_item_h - sc(2))
            go_rect  = pygame.Rect(mcx + saved_name_col_w + sc(7), item_y + sc(1), saved_go_w, saved_item_h - sc(2))
            edt_rect = pygame.Rect(go_rect.right + sc(7), item_y + sc(1), saved_edt_w, saved_item_h - sc(2))
            del_rect = pygame.Rect(edt_rect.right + sc(7), item_y + sc(1), saved_del_w, saved_item_h - sc(2))
            is_editing = (idx == editing_idx)
            row_bg_color     = (22, 38, 60) if is_editing else CARD_BG
            row_border_color = ACCENT_COLOR if is_editing else BORDER_COLOR
            pygame.draw.rect(screen, row_bg_color, row_bg, border_radius=sc(4))
            pygame.draw.rect(screen, row_border_color, row_bg, 1, border_radius=sc(4))
            mode_tag = "C" if entry["mode"] == "cartesian" else "J"
            tag_color = ACCENT_COLOR if entry["mode"] == "cartesian" else (180, 120, 255)
            tag_img = sec_font.render(mode_tag, True, tag_color)
            screen.blit(tag_img, (mcx + sc(6), item_y + saved_item_h // 2 - tag_img.get_height() // 2))
            name_img = font.render(entry["name"], True, TEXT_COLOR)
            screen.blit(name_img, (mcx + sc(6) + tag_img.get_width() + sc(8),
                                   item_y + saved_item_h // 2 - name_img.get_height() // 2))
            draw_btn(screen, sec_font, go_rect, "GO", go_rect.collidepoint(mouse_pos), attach=True, br=sc(4))
            draw_btn(screen, sec_font, edt_rect, "EDIT", edt_rect.collidepoint(mouse_pos), active=is_editing, br=sc(4))
            draw_icon_btn(screen, del_rect, del_rect.collidepoint(mouse_pos), "delete", danger=True, br=sc(4))
        screen.set_clip(clip_rect)

        # Scroll indicator
        if len(saved_positions) > saved_visible:
            total = len(saved_positions)
            track_h = saved_area.h
            thumb_h = max(sc(20), int(track_h * saved_visible / total))
            thumb_y = saved_area.y + int((track_h - thumb_h) * saved_scroll / max(1, total - saved_visible))
            pygame.draw.rect(screen, BORDER_COLOR,
                             pygame.Rect(saved_area.right - sc(4), saved_area.y, sc(4), track_h), border_radius=sc(2))
            pygame.draw.rect(screen, DIM_COLOR,
                             pygame.Rect(saved_area.right - sc(4), thumb_y, sc(4), thumb_h), border_radius=sc(2))

        # ── DROPDOWN OVERLAYS (drawn last) ─────────────────────────────────
        if input_dropdown_open:
            draw_dropdown_menu(screen, ui_font, input_rect,
                               [INPUT_METHODS[k] for k in input_method_keys],
                               input_method_keys.index(selected_input_method),
                               mouse_pos, br=br)

        if mapping_dropdown_open and mapping_rect:
            draw_dropdown_menu(screen, ui_font, mapping_rect,
                               [MAPPINGS[k]["label"] for k in mapping_keys],
                               mapping_keys.index(selected_mapping),
                               mouse_pos, br=br)

        if tool_dropdown_open and tool_names:
            draw_dropdown_menu(screen, ui_font, tool_dd_rect,
                               tool_names, selected_tool_idx, mouse_pos, br=br)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    try:
        main()
    finally:
        robot.shutdown()
        pygame.quit()
