import argparse
import os

import numpy as np
import pygame

from robot_api import Robot

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
WIDTH, HEIGHT = 1680, 1080
BG_COLOR = (15, 15, 18)
PANEL_COLOR = (28, 28, 33)
TEXT_COLOR = (240, 240, 240)
HEADER_COLOR = (0, 255, 150)
ACCENT_COLOR = (54, 162, 255)
HOVER_COLOR = (62, 62, 78)
BTN_COLOR = (45, 45, 52)
BTN_ACTIVE = (0, 160, 255)
GRIPPER_MIN = 0.0
GRIPPER_MAX = 0.05
GRIPPER_SPEED = 0.1  # units/s in [GRIPPER_MIN, GRIPPER_MAX]
LIN_SPD_MIN = 0.01
LIN_SPD_MAX = 1.0
LIN_SPD_STEP = 0.01
LIN_SPD_DEFAULT = 0.02
ANG_SPD_MIN = 0.1
ANG_SPD_MAX = 2.0
ANG_SPD_STEP = 0.05
ANG_SPD_DEFAULT = 0.2
LIN_ACCEL_MIN = 0.01
LIN_ACCEL_MAX = 1.0
LIN_ACCEL_STEP = 0.01
LIN_ACCEL_DEFAULT = 0.02
ANG_ACCEL_MIN = 0.05
ANG_ACCEL_MAX = 1.0
ANG_ACCEL_STEP = 0.05
ANG_ACCEL_DEFAULT = 0.2


MAPPINGS = {
    "drone": {
        "label": "Drone Style (Legacy)",
        "apply": lambda js, lspd, aspd: (
            np.array(
                [
                    -apply_deadzone(js.get_axis(4)) * lspd,
                    -apply_deadzone(js.get_axis(3)) * lspd,
                    -apply_deadzone(js.get_axis(1)) * lspd,
                ]
            ),
            np.array([0.0, 0.0, apply_deadzone(js.get_axis(0)) * aspd]),
        ),
        "legend": [
            "L-Stick Vert:  Up / Down",
            "L-Stick Horiz: Yaw (Rotate)",
            "R-Stick Vert:  Forward / Back",
            "R-Stick Horiz: Side-to-Side",
            "D-Pad Vert:    Pitch",
            "D-Pad Horiz:   Roll",
            "LB/RB:         Gripper Open/Close",
        ],
    },
    "standard": {
        "label": "Standard Dual Stick",
        "apply": lambda js, lspd, aspd: (
            np.array(
                [
                    -apply_deadzone(js.get_axis(1)) * lspd,
                    -apply_deadzone(js.get_axis(0)) * lspd,
                    0.0,
                ]
            ),
            np.array(
                [
                    apply_deadzone(js.get_axis(3)) * aspd,
                    -apply_deadzone(js.get_axis(4)) * aspd,
                    0.0,
                ]
            ),
        ),
        "legend": [
            "L-Stick Vert:  Forward / Back",
            "L-Stick Horiz: Side-to-Side",
            "R-Stick Horiz: Tilt (Roll)",
            "R-Stick Vert:  Pitch",
            "L2/R2:         Yaw (Rotate)",
            "L1/R1:         Up / Down",
            "D-Pad Up/Down: Gripper Open/Close",
        ],
    },
}

INPUT_METHODS = {
    "controller": "Controller",
    "keyboard": "Keyboard",
    "buttons": "Buttons",
}

KEYBOARD_LEGEND = [
    "W/S:           Forward / Back",
    "A/D:           Side-to-Side",
    "Space/LShift:  Up / Down",
    "Left/Right:    Tilt (Roll)",
    "Up/Down:       Pitch",
    "Q/E:           Yaw (Rotate)",
    "O/P:           Gripper Open/Close",
]


def apply_deadzone(value, deadzone=0.1):
    return value if abs(value) > deadzone else 0.0


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
        screen.blit(
            txt_img,
            (self.rect.centerx - txt_img.get_width() // 2, self.rect.centery - txt_img.get_height() // 2),
        )

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.mouse_down = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.mouse_down = False


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
    delta = target - current
    return current + np.clip(delta, -max_delta, max_delta)


def read_button(joystick, button_idx):
    return joystick.get_numbuttons() > button_idx and joystick.get_button(button_idx)


def read_trigger(joystick, axis_idx, button_idx):
    trigger = 0.0
    if joystick.get_numaxes() > axis_idx:
        raw = joystick.get_axis(axis_idx)
        if raw < -0.05:
            trigger = (raw + 1.0) / 2.0
        else:
            trigger = raw
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
        # L1/R1 controls Z (up/down) in standard mode.
        lin_vel[2] = (float(read_button(joystick, 5)) - float(read_button(joystick, 4))) * l_spd
        # L2/R2 controls yaw. Common mappings are L2 axis=2 and R2 axis=5.
        l2 = read_trigger(joystick, 2, 6)
        r2 = read_trigger(joystick, 5, 7)
        ang_vel[2] = (l2 - r2) * a_spd

    return lin_vel, ang_vel


def main():
    pygame.init()
    pygame.joystick.init()

    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Robot Command Center")

    font = pygame.font.SysFont("Consolas", 18)
    big_font = pygame.font.SysFont("Arial", 24, bold=True)
    label_font = pygame.font.SysFont("Arial", 26, bold=True)
    clock = pygame.time.Clock()

    # Speed Multipliers
    l_spd = LIN_SPD_DEFAULT  # Linear m/s
    a_spd = ANG_SPD_DEFAULT  # Angular rad/s
    max_lin_accel = LIN_ACCEL_DEFAULT
    max_ang_accel = ANG_ACCEL_DEFAULT
    fake_mode = args.fake_hardware
    gripper_pos = 0.0
    last_sent_gripper_pos = None
    current_lin_vel = np.zeros(3)
    current_ang_vel = np.zeros(3)

    mapping_keys = list(MAPPINGS.keys())
    input_method_keys = list(INPUT_METHODS.keys())
    tool_map = {
        name: value
        for name in dir(robot.tools)
        for value in [getattr(robot.tools, name)]
        if not name.startswith("_") and isinstance(value, robot.tools.Tool)
    }
    tool_names = list(tool_map.keys())
    selected_mapping = "standard"
    selected_input_method = "controller" if joystick else "keyboard"
    input_dropdown_open = False
    mapping_dropdown_open = False
    left_panel_x = 30
    left_panel_w = 520
    dropdown_w = 360
    dropdown_x = left_panel_x + (left_panel_w - dropdown_w) // 2
    hardware_toggle_rect = pygame.Rect(dropdown_x, 90, dropdown_w, 36)
    input_dropdown_rect = pygame.Rect(dropdown_x, 134, dropdown_w, 36)
    mapping_dropdown_rect = pygame.Rect(dropdown_x, 178, dropdown_w, 36)
    dropdown_item_height = 36

    controls_start_y = 252
    controls_line_step = 26
    velocity_title_y = 500
    velocity_row_start_y = 548
    velocity_row_step = 48
    velocity_row_h = 40
    velocity_btn_left_x = left_panel_x + 24
    velocity_btn_right_x = left_panel_x + left_panel_w - 24 - 40
    tool_title_y = 758
    tool_row_start_y = 780
    tool_row_step = 50
    tool_apply_y = 890

    lin_dec_rect = pygame.Rect(velocity_btn_left_x, velocity_row_start_y, 40, velocity_row_h)
    lin_inc_rect = pygame.Rect(velocity_btn_right_x, velocity_row_start_y, 40, velocity_row_h)
    ang_dec_rect = pygame.Rect(velocity_btn_left_x, velocity_row_start_y + velocity_row_step, 40, velocity_row_h)
    ang_inc_rect = pygame.Rect(velocity_btn_right_x, velocity_row_start_y + velocity_row_step, 40, velocity_row_h)
    lin_accel_dec_rect = pygame.Rect(velocity_btn_left_x, velocity_row_start_y + velocity_row_step * 2, 40, velocity_row_h)
    lin_accel_inc_rect = pygame.Rect(
        velocity_btn_right_x,
        velocity_row_start_y + velocity_row_step * 2,
        40,
        velocity_row_h,
    )
    ang_accel_dec_rect = pygame.Rect(velocity_btn_left_x, velocity_row_start_y + velocity_row_step * 3, 40, velocity_row_h)
    ang_accel_inc_rect = pygame.Rect(
        velocity_btn_right_x,
        velocity_row_start_y + velocity_row_step * 3,
        40,
        velocity_row_h,
    )
    tool_mode = "attach"
    selected_tool_idx = tool_names.index("gripper") if "gripper" in tool_names else 0
    tool_status = "Ready"
    tool_mode_prev_rect = pygame.Rect(dropdown_x, tool_row_start_y, 40, 40)
    tool_mode_next_rect = pygame.Rect(dropdown_x + dropdown_w - 40, tool_row_start_y, 40, 40)
    tool_prev_rect = pygame.Rect(dropdown_x, tool_row_start_y + tool_row_step, 40, 40)
    tool_next_rect = pygame.Rect(dropdown_x + dropdown_w - 40, tool_row_start_y + tool_row_step, 40, 40)
    tool_apply_rect = pygame.Rect(dropdown_x + (dropdown_w - 170) // 2, tool_apply_y, 170, 44)
    btns = [
        # Linear
        AxisButton(828, 550, 62, 58, "+X"),
        AxisButton(828, 680, 62, 58, "-X"),
        AxisButton(758, 615, 62, 58, "+Y"),
        AxisButton(898, 615, 62, 58, "-Y"),
        AxisButton(1003, 550, 68, 58, "+Z"),
        AxisButton(1003, 680, 68, 58, "-Z"),
        # Rotation
        AxisButton(1228, 550, 62, 58, "+P"),
        AxisButton(1228, 680, 62, 58, "-P"),
        AxisButton(1158, 615, 62, 58, "+Yw"),
        AxisButton(1298, 615, 62, 58, "-Yw"),
        AxisButton(1403, 550, 68, 58, "+R"),
        AxisButton(1403, 680, 68, 58, "-R"),
    ]

    # Ensure gripper command path is available.
    try:
        robot.tool_changer.attach_tool(robot.tools.gripper)
    except Exception as e:
        robot.node.get_logger().warn(f"Failed to attach gripper tool: {e}")

    while True:
        screen.fill(BG_COLOR)
        dt_sec = max(clock.get_time() / 1000.0, 1.0 / 240.0)
        target_lin_vel = np.zeros(3)
        target_ang_vel = np.zeros(3)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if selected_input_method == "buttons":
                for b in btns:
                    b.handle_event(event)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                handled_click = False
                if input_dropdown_open:
                    input_dropdown_open = False
                    for idx, key in enumerate(input_method_keys):
                        option_rect = pygame.Rect(
                            input_dropdown_rect.x,
                            input_dropdown_rect.y + dropdown_item_height * (idx + 1),
                            input_dropdown_rect.width,
                            dropdown_item_height,
                        )
                        if option_rect.collidepoint(event.pos):
                            selected_input_method = key
                            handled_click = True
                            break

                if mapping_dropdown_open and not handled_click:
                    mapping_dropdown_open = False
                    for idx, key in enumerate(mapping_keys):
                        option_rect = pygame.Rect(
                            mapping_dropdown_rect.x,
                            mapping_dropdown_rect.y + dropdown_item_height * (idx + 1),
                            mapping_dropdown_rect.width,
                            dropdown_item_height,
                        )
                        if option_rect.collidepoint(event.pos):
                            selected_mapping = key
                            handled_click = True
                            break

                if not handled_click and input_dropdown_rect.collidepoint(event.pos):
                    input_dropdown_open = not input_dropdown_open
                    mapping_dropdown_open = False
                    handled_click = True
                if not handled_click and hardware_toggle_rect.collidepoint(event.pos):
                    fake_mode = not fake_mode
                    robot.set_fake_hardware_mode(fake_mode)
                    handled_click = True
                if (
                    not handled_click
                    and selected_input_method == "controller"
                    and mapping_dropdown_rect.collidepoint(event.pos)
                ):
                    mapping_dropdown_open = not mapping_dropdown_open
                    input_dropdown_open = False
                    handled_click = True

                if not handled_click and lin_dec_rect.collidepoint(event.pos):
                    l_spd = clamp(l_spd - LIN_SPD_STEP, LIN_SPD_MIN, LIN_SPD_MAX)
                    handled_click = True
                if not handled_click and lin_inc_rect.collidepoint(event.pos):
                    l_spd = clamp(l_spd + LIN_SPD_STEP, LIN_SPD_MIN, LIN_SPD_MAX)
                    handled_click = True
                if not handled_click and ang_dec_rect.collidepoint(event.pos):
                    a_spd = clamp(a_spd - ANG_SPD_STEP, ANG_SPD_MIN, ANG_SPD_MAX)
                    handled_click = True
                if not handled_click and ang_inc_rect.collidepoint(event.pos):
                    a_spd = clamp(a_spd + ANG_SPD_STEP, ANG_SPD_MIN, ANG_SPD_MAX)
                    handled_click = True
                if not handled_click and lin_accel_dec_rect.collidepoint(event.pos):
                    max_lin_accel = clamp(max_lin_accel - LIN_ACCEL_STEP, LIN_ACCEL_MIN, LIN_ACCEL_MAX)
                    handled_click = True
                if not handled_click and lin_accel_inc_rect.collidepoint(event.pos):
                    max_lin_accel = clamp(max_lin_accel + LIN_ACCEL_STEP, LIN_ACCEL_MIN, LIN_ACCEL_MAX)
                    handled_click = True
                if not handled_click and ang_accel_dec_rect.collidepoint(event.pos):
                    max_ang_accel = clamp(max_ang_accel - ANG_ACCEL_STEP, ANG_ACCEL_MIN, ANG_ACCEL_MAX)
                    handled_click = True
                if not handled_click and ang_accel_inc_rect.collidepoint(event.pos):
                    max_ang_accel = clamp(max_ang_accel + ANG_ACCEL_STEP, ANG_ACCEL_MIN, ANG_ACCEL_MAX)
                    handled_click = True
                if not handled_click and tool_mode_prev_rect.collidepoint(event.pos):
                    tool_mode = "detach" if tool_mode == "attach" else "attach"
                    handled_click = True
                if not handled_click and tool_mode_next_rect.collidepoint(event.pos):
                    tool_mode = "detach" if tool_mode == "attach" else "attach"
                    handled_click = True
                if (
                    not handled_click
                    and tool_prev_rect.collidepoint(event.pos)
                    and len(tool_names) > 0
                ):
                    selected_tool_idx = (selected_tool_idx - 1) % len(tool_names)
                    handled_click = True
                if (
                    not handled_click
                    and tool_next_rect.collidepoint(event.pos)
                    and len(tool_names) > 0
                ):
                    selected_tool_idx = (selected_tool_idx + 1) % len(tool_names)
                    handled_click = True
                if not handled_click and tool_apply_rect.collidepoint(event.pos):
                    try:
                        if tool_mode == "detach":
                            robot.tool_changer.detach_tool()
                            tool_status = "Tool detached"
                        elif len(tool_names) > 0:
                            chosen_tool = tool_map[tool_names[selected_tool_idx]]
                            robot.tool_changer.attach_tool(chosen_tool)
                            tool_status = f"Attached: {tool_names[selected_tool_idx]}"
                        else:
                            tool_status = "No tools available"
                    except Exception as e:
                        tool_status = f"Tool command failed: {e}"
                    handled_click = True

        keys = pygame.key.get_pressed()
        if selected_input_method == "controller":
            if joystick:
                target_lin_vel, target_ang_vel = apply_mapping(joystick, selected_mapping, l_spd, a_spd)

                if selected_mapping == "drone":
                    # Legacy scheme: shoulder buttons open/close gripper.
                    if read_button(joystick, 4):
                        gripper_pos += GRIPPER_SPEED * dt_sec
                    if read_button(joystick, 5):
                        gripper_pos -= GRIPPER_SPEED * dt_sec
                elif selected_mapping == "standard":
                    # Standard scheme: D-pad up/down open/close gripper.
                    dpad = joystick.get_hat(0)
                    if dpad[1] > 0:
                        gripper_pos += GRIPPER_SPEED * dt_sec
                    if dpad[1] < 0:
                        gripper_pos -= GRIPPER_SPEED * dt_sec
        elif selected_input_method == "keyboard":
            if keys[pygame.K_w]:
                target_lin_vel[0] += l_spd
            if keys[pygame.K_s]:
                target_lin_vel[0] -= l_spd
            if keys[pygame.K_a]:
                target_lin_vel[1] += l_spd
            if keys[pygame.K_d]:
                target_lin_vel[1] -= l_spd
            if keys[pygame.K_SPACE]:
                target_lin_vel[2] += l_spd
            if keys[pygame.K_LSHIFT]:
                target_lin_vel[2] -= l_spd
            if keys[pygame.K_LEFT]:
                target_ang_vel[0] += a_spd
            if keys[pygame.K_RIGHT]:
                target_ang_vel[0] -= a_spd
            if keys[pygame.K_UP]:
                target_ang_vel[1] += a_spd
            if keys[pygame.K_DOWN]:
                target_ang_vel[1] -= a_spd
            if keys[pygame.K_q]:
                target_ang_vel[2] += a_spd
            if keys[pygame.K_e]:
                target_ang_vel[2] -= a_spd
        elif selected_input_method == "buttons":
            for b in btns:
                b.is_pressed = b.mouse_down
            if btns[0].is_pressed:
                target_lin_vel[0] += l_spd
            if btns[1].is_pressed:
                target_lin_vel[0] -= l_spd
            if btns[2].is_pressed:
                target_lin_vel[1] += l_spd
            if btns[3].is_pressed:
                target_lin_vel[1] -= l_spd
            if btns[4].is_pressed:
                target_lin_vel[2] += l_spd
            if btns[5].is_pressed:
                target_lin_vel[2] -= l_spd
            if btns[6].is_pressed:
                target_ang_vel[1] += a_spd
            if btns[7].is_pressed:
                target_ang_vel[1] -= a_spd
            if btns[8].is_pressed:
                target_ang_vel[2] += a_spd
            if btns[9].is_pressed:
                target_ang_vel[2] -= a_spd
            if btns[10].is_pressed:
                target_ang_vel[0] += a_spd
            if btns[11].is_pressed:
                target_ang_vel[0] -= a_spd

        # Keyboard fallback for gripper control.
        if keys[pygame.K_o]:
            gripper_pos += GRIPPER_SPEED * dt_sec
        if keys[pygame.K_p]:
            gripper_pos -= GRIPPER_SPEED * dt_sec
        gripper_pos = float(np.clip(gripper_pos, GRIPPER_MIN, GRIPPER_MAX))

        # Acceleration limiting smooths command changes and enforces max acceleration.
        current_lin_vel = limit_acceleration(current_lin_vel, target_lin_vel, max_lin_accel, dt_sec)
        current_ang_vel = limit_acceleration(current_ang_vel, target_ang_vel, max_ang_accel, dt_sec)

        # Execute movement
        robot.cartesian_space.twist(tuple(current_lin_vel), tuple(current_ang_vel))
        if last_sent_gripper_pos is None or abs(gripper_pos - last_sent_gripper_pos) > 1e-4:
            try:
                robot.tools.gripper.set_distance(gripper_pos)
                last_sent_gripper_pos = gripper_pos
            except RuntimeError:
                # Gripper not attached as current tool.
                pass

        # --- DRAW INTERFACE ---
        pose = robot.cartesian_space.read()
        joint_pt = robot.joint_space.read()

        pygame.draw.rect(screen, PANEL_COLOR, (30, 30, 520, 1020), border_radius=12)
        pygame.draw.rect(screen, PANEL_COLOR, (580, 30, 1070, 1020), border_radius=12)
        pygame.draw.rect(screen, (24, 24, 30), (620, 70, 990, 320), border_radius=10)
        pygame.draw.rect(screen, (24, 24, 30), (620, 420, 990, 600), border_radius=10)

        # Controls panel
        header_img = big_font.render("CONTROL", True, HEADER_COLOR)
        screen.blit(header_img, (left_panel_x + (left_panel_w - header_img.get_width()) // 2, 60))
        mouse_pos = pygame.mouse.get_pos()
        input_bg = HOVER_COLOR if input_dropdown_rect.collidepoint(mouse_pos) else (35, 35, 45)
        pygame.draw.rect(screen, input_bg, input_dropdown_rect, border_radius=6)
        pygame.draw.rect(screen, ACCENT_COLOR, input_dropdown_rect, width=1, border_radius=6)
        screen.blit(
            font.render(f"Input: {INPUT_METHODS[selected_input_method]}", True, TEXT_COLOR),
            (input_dropdown_rect.x + 8, input_dropdown_rect.y + 7),
        )
        screen.blit(
            font.render("v" if not input_dropdown_open else "^", True, TEXT_COLOR),
            (input_dropdown_rect.x + input_dropdown_rect.width - 20, input_dropdown_rect.y + 7),
        )
        hw_bg = HOVER_COLOR if hardware_toggle_rect.collidepoint(mouse_pos) else (35, 35, 45)
        pygame.draw.rect(screen, hw_bg, hardware_toggle_rect, border_radius=6)
        pygame.draw.rect(screen, ACCENT_COLOR, hardware_toggle_rect, width=1, border_radius=6)
        hw_label = "Hardware: FAKE (click to switch)" if fake_mode else "Hardware: REAL (click to switch)"
        screen.blit(font.render(hw_label, True, TEXT_COLOR), (hardware_toggle_rect.x + 8, hardware_toggle_rect.y + 7))

        if selected_input_method == "controller":
            mapping_bg = HOVER_COLOR if mapping_dropdown_rect.collidepoint(mouse_pos) else (35, 35, 45)
            pygame.draw.rect(screen, mapping_bg, mapping_dropdown_rect, border_radius=6)
            pygame.draw.rect(screen, ACCENT_COLOR, mapping_dropdown_rect, width=1, border_radius=6)
            selected_label = MAPPINGS[selected_mapping]["label"]
            screen.blit(
                font.render(f"Mapping: {selected_label}", True, TEXT_COLOR),
                (mapping_dropdown_rect.x + 8, mapping_dropdown_rect.y + 7),
            )
            screen.blit(
                font.render("v" if not mapping_dropdown_open else "^", True, TEXT_COLOR),
                (mapping_dropdown_rect.x + mapping_dropdown_rect.width - 20, mapping_dropdown_rect.y + 7),
            )

        if selected_input_method == "controller" and joystick is None:
            screen.blit(
                font.render("No joystick detected. Switch Input to Keyboard.", True, (255, 140, 140)),
                (left_panel_x + 20, hardware_toggle_rect.y + 34),
            )

        vel_title = big_font.render("KINEMATICS", True, HEADER_COLOR)
        screen.blit(vel_title, (left_panel_x + (left_panel_w - vel_title.get_width()) // 2, velocity_title_y))
        for rect, label in [
            (lin_dec_rect, "-"),
            (lin_inc_rect, "+"),
            (ang_dec_rect, "-"),
            (ang_inc_rect, "+"),
            (lin_accel_dec_rect, "-"),
            (lin_accel_inc_rect, "+"),
            (ang_accel_dec_rect, "-"),
            (ang_accel_inc_rect, "+"),
        ]:
            bg = HOVER_COLOR if rect.collidepoint(mouse_pos) else (35, 35, 45)
            pygame.draw.rect(screen, bg, rect, border_radius=6)
            pygame.draw.rect(screen, ACCENT_COLOR, rect, width=1, border_radius=6)
            label_img = big_font.render(label, True, TEXT_COLOR)
            screen.blit(
                label_img,
                (rect.x + (rect.width - label_img.get_width()) // 2, rect.y + (rect.height - label_img.get_height()) // 2),
            )

        velocity_rows = [
            (f"Lin Vel: {l_spd: .3f} m/s", lin_dec_rect),
            (f"Ang Vel: {a_spd: .3f} rad/s", ang_dec_rect),
            (f"Lin Acc: {max_lin_accel: .3f} m/s^2", lin_accel_dec_rect),
            (f"Ang Acc: {max_ang_accel: .3f} rad/s^2", ang_accel_dec_rect),
        ]
        velocity_text_left = lin_dec_rect.right + 14
        velocity_text_right = lin_inc_rect.left - 14
        velocity_text_w = max(0, velocity_text_right - velocity_text_left)
        for text, row_rect in velocity_rows:
            text_img = font.render(text, True, TEXT_COLOR)
            text_x = velocity_text_left + (velocity_text_w - text_img.get_width()) // 2
            screen.blit(
                text_img,
                (
                    text_x,
                    row_rect.y + (row_rect.height - text_img.get_height()) // 2,
                ),
            )
        tool_title = big_font.render("TOOL CHANGER", True, HEADER_COLOR)
        screen.blit(tool_title, (left_panel_x + (left_panel_w - tool_title.get_width()) // 2, tool_title_y))
        for rect, label in [
            (tool_mode_prev_rect, "<"),
            (tool_mode_next_rect, ">"),
            (tool_prev_rect, "<"),
            (tool_next_rect, ">"),
        ]:
            bg = HOVER_COLOR if rect.collidepoint(mouse_pos) else (35, 35, 45)
            pygame.draw.rect(screen, bg, rect, border_radius=6)
            pygame.draw.rect(screen, ACCENT_COLOR, rect, width=1, border_radius=6)
            label_img = big_font.render(label, True, TEXT_COLOR)
            screen.blit(
                label_img,
                (rect.x + (rect.width - label_img.get_width()) // 2, rect.y + (rect.height - label_img.get_height()) // 2),
            )

        apply_bg = HOVER_COLOR if tool_apply_rect.collidepoint(mouse_pos) else (35, 35, 45)
        pygame.draw.rect(screen, apply_bg, tool_apply_rect, border_radius=6)
        pygame.draw.rect(screen, ACCENT_COLOR, tool_apply_rect, width=1, border_radius=6)
        apply_txt = font.render("Apply", True, TEXT_COLOR)
        screen.blit(
            apply_txt,
            (
                tool_apply_rect.x + (tool_apply_rect.width - apply_txt.get_width()) // 2,
                tool_apply_rect.y + (tool_apply_rect.height - apply_txt.get_height()) // 2,
            ),
        )

        selected_tool_name = tool_names[selected_tool_idx] if len(tool_names) > 0 else "none"
        current_tool_name = find_current_tool_name(tool_map, robot.tool_changer.current_tool)
        tool_mode_txt = font.render(f"Mode:          {tool_mode}", True, TEXT_COLOR)
        tool_sel_txt = font.render(f"Selected Tool: {selected_tool_name}", True, TEXT_COLOR)
        screen.blit(
            tool_mode_txt,
            (dropdown_x + 52, tool_mode_prev_rect.y + (tool_mode_prev_rect.height - tool_mode_txt.get_height()) // 2),
        )
        screen.blit(
            tool_sel_txt,
            (dropdown_x + 52, tool_prev_rect.y + (tool_prev_rect.height - tool_sel_txt.get_height()) // 2),
        )
        screen.blit(font.render(f"Current Tool:  {current_tool_name}", True, TEXT_COLOR), (dropdown_x, 960))
        screen.blit(font.render(f"Status: {tool_status}", True, (190, 210, 255)), (dropdown_x, 992))

        controls = (
            MAPPINGS[selected_mapping]["legend"]
            if selected_input_method == "controller"
            else KEYBOARD_LEGEND if selected_input_method == "keyboard" else [
                "Use on-screen buttons in right panel",
                "Button layout mirrors keyboard mode",
                "O/P:           Gripper Open/Close",
            ]
        ) + [f"Gripper Cmd:   {gripper_pos: .3f}"]
        for i, text in enumerate(controls):
            screen.blit(font.render(text, True, TEXT_COLOR), (dropdown_x, controls_start_y + i * controls_line_step))

        if selected_input_method == "buttons":
            btn_title = big_font.render("BUTTON CONTROLS", True, HEADER_COLOR)
            screen.blit(btn_title, (620 + (990 - btn_title.get_width()) // 2, 450))
            for b in btns:
                b.draw(screen, label_font)

        # Telemetry columns
        c_lines = [
            ("CARTESIAN POSE", big_font, HEADER_COLOR),
            (f"X:     {pose.position[0]: .4f}", font, TEXT_COLOR),
            (f"Y:     {pose.position[1]: .4f}", font, TEXT_COLOR),
            (f"Z:     {pose.position[2]: .4f}", font, TEXT_COLOR),
            (f"Roll:  {pose.orientation[0]: .4f}", font, TEXT_COLOR),
            (f"Pitch: {pose.orientation[1]: .4f}", font, TEXT_COLOR),
            (f"Yaw:   {pose.orientation[2]: .4f}", font, TEXT_COLOR),
        ]
        j_lines = [("JOINT ANGLES", big_font, HEADER_COLOR)]
        for i, val in enumerate(joint_pt.joint_configuration):
            j_lines.append((f"q{i+1}: {val: .4f} rad", font, TEXT_COLOR))
        telemetry_title = big_font.render("SYSTEM TELEMETRY", True, HEADER_COLOR)
        screen.blit(telemetry_title, (620 + (990 - telemetry_title.get_width()) // 2, 90))
        c_col_w = max(fnt.size(txt)[0] for txt, fnt, _ in c_lines)
        j_col_w = max(fnt.size(txt)[0] for txt, fnt, _ in j_lines)
        col_gap = 80
        total_cols_w = c_col_w + col_gap + j_col_w
        cols_start_x = 620 + (990 - total_cols_w) // 2
        c_col_x = cols_start_x
        j_col_x = c_col_x + c_col_w + col_gap
        for i, (txt, fnt, color) in enumerate(c_lines):
            screen.blit(fnt.render(txt, True, color), (c_col_x, 140 + i * 34))
        for i, (txt, fnt, color) in enumerate(j_lines):
            screen.blit(fnt.render(txt, True, color), (j_col_x, 140 + i * 34))

        # Draw dropdown menus at the very end so nothing can overwrite them.
        if selected_input_method == "controller" and mapping_dropdown_open:
            menu_height = dropdown_item_height * len(mapping_keys)
            menu_rect = pygame.Rect(
                mapping_dropdown_rect.x,
                mapping_dropdown_rect.y + dropdown_item_height,
                mapping_dropdown_rect.width,
                menu_height,
            )
            pygame.draw.rect(screen, (12, 12, 16), menu_rect)
            pygame.draw.rect(screen, ACCENT_COLOR, menu_rect, width=1)

            for idx, key in enumerate(mapping_keys):
                option_rect = pygame.Rect(
                    mapping_dropdown_rect.x,
                    mapping_dropdown_rect.y + dropdown_item_height * (idx + 1),
                    mapping_dropdown_rect.width,
                    dropdown_item_height,
                )
                option_bg = (48, 48, 62) if key == selected_mapping else (30, 30, 38)
                if option_rect.collidepoint(mouse_pos):
                    option_bg = HOVER_COLOR
                pygame.draw.rect(screen, option_bg, option_rect)
                pygame.draw.rect(screen, (20, 20, 26), option_rect, width=1)
                screen.blit(
                    font.render(MAPPINGS[key]["label"], True, TEXT_COLOR),
                    (option_rect.x + 8, option_rect.y + 7),
                )

        if input_dropdown_open:
            menu_height = dropdown_item_height * len(input_method_keys)
            menu_rect = pygame.Rect(
                input_dropdown_rect.x,
                input_dropdown_rect.y + dropdown_item_height,
                input_dropdown_rect.width,
                menu_height,
            )
            pygame.draw.rect(screen, (12, 12, 16), menu_rect)
            pygame.draw.rect(screen, ACCENT_COLOR, menu_rect, width=1)

            for idx, key in enumerate(input_method_keys):
                option_rect = pygame.Rect(
                    input_dropdown_rect.x,
                    input_dropdown_rect.y + dropdown_item_height * (idx + 1),
                    input_dropdown_rect.width,
                    dropdown_item_height,
                )
                option_bg = (48, 48, 62) if key == selected_input_method else (30, 30, 38)
                if option_rect.collidepoint(mouse_pos):
                    option_bg = HOVER_COLOR
                pygame.draw.rect(screen, option_bg, option_rect)
                pygame.draw.rect(screen, (20, 20, 26), option_rect, width=1)
                screen.blit(
                    font.render(INPUT_METHODS[key], True, TEXT_COLOR),
                    (option_rect.x + 8, option_rect.y + 7),
                )

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    try:
        main()
    finally:
        robot.shutdown()
        pygame.quit()
