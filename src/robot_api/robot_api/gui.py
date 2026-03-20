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
GRIPPER_MIN, GRIPPER_MAX, GRIPPER_SPEED = 0.0, 0.05, 0.1
LIN_SPD_MIN,   LIN_SPD_MAX,   LIN_SPD_STEP,   LIN_SPD_DEFAULT   = 0.01, 1.0,  0.01, 0.02
ANG_SPD_MIN,   ANG_SPD_MAX,   ANG_SPD_STEP,   ANG_SPD_DEFAULT   = 0.1,  2.0,  0.05, 0.2
LIN_ACCEL_MIN, LIN_ACCEL_MAX, LIN_ACCEL_STEP, LIN_ACCEL_DEFAULT = 0.01, 1.0,  0.01, 0.02
ANG_ACCEL_MIN, ANG_ACCEL_MAX, ANG_ACCEL_STEP, ANG_ACCEL_DEFAULT = 0.05, 1.0,  0.05, 0.2

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
    "buttons":    "Buttons",
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

BTN_DEFS = [
    # Linear  (x, y, w, h, label, axis_color)
    (838, 530, 62, 56, "X+",  (220,  70,  70)),
    (838, 650, 62, 56, "X−",  (220,  70,  70)),
    (768, 590, 62, 56, "Y+",  ( 70, 200,  90)),
    (908, 590, 62, 56, "Y−",  ( 70, 200,  90)),
    (1008,530, 68, 56, "Z+",  ( 55, 155, 255)),
    (1008,650, 68, 56, "Z−",  ( 55, 155, 255)),
    # Rotation
    (1228,530, 62, 56, "P+",  (210, 120, 255)),
    (1228,650, 62, 56, "P−",  (210, 120, 255)),
    (1158,590, 62, 56, "Yw+", (255, 190,  50)),
    (1298,590, 62, 56, "Yw−", (255, 190,  50)),
    (1398,530, 68, 56, "R+",  (  0, 210, 200)),
    (1398,650, 68, 56, "R−",  (  0, 210, 200)),
]

# Bounding boxes of the two button groups (base resolution)
BTN_GROUP_LINEAR   = (754, 516, 336, 204)
BTN_GROUP_ROTATION = (1144, 516, 336, 204)


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


class AxisButton:
    def __init__(self, x, y, w, h, label, color=(55, 155, 255)):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.color = color
        self.is_pressed = False
        self.mouse_down = False

    def draw(self, surf, font, br=6):
        r = self.rect
        pressed = self.is_pressed

        # Background
        bg = self.color if pressed else (28, 30, 46)
        pygame.draw.rect(surf, bg, r, border_radius=br)

        # Colored border — dim when idle, bright when pressed
        border_alpha = self.color
        border_w = 2 if pressed else 1
        pygame.draw.rect(surf, border_alpha, r, border_w, border_radius=br)

        # Top highlight strip (gives depth when not pressed)
        if not pressed:
            hl = pygame.Rect(r.x + br, r.y + 2, r.w - br * 2, 2)
            pygame.draw.rect(surf, tuple(min(255, c + 40) for c in self.color), hl)

        # Label
        axis = self.label[:-1]   # e.g. "X", "Yw", "R"
        sign = self.label[-1]    # "+" or "−"
        text_color = (15, 15, 20) if pressed else TEXT_COLOR
        sign_color = (15, 15, 20) if pressed else self.color

        axis_img = font.render(axis, True, text_color)
        sign_img = font.render(sign, True, sign_color)

        total_w = axis_img.get_width() + sign_img.get_width() + 3
        tx = r.centerx - total_w // 2
        ty = r.centery - axis_img.get_height() // 2
        surf.blit(axis_img, (tx, ty))
        surf.blit(sign_img, (tx + axis_img.get_width() + 3, ty))

    def handle_event(self, event, pos=None):
        if event.type == pygame.MOUSEBUTTONDOWN and pos is not None and self.rect.collidepoint(pos):
            self.mouse_down = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.mouse_down = False


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
    fake_mode      = args.fake_hardware
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
    btns = [AxisButton(0, 0, 0, 0, label, color) for _, _, _, _, label, color in BTN_DEFS]

    try:
        robot.tool_changer.attach_tool(robot.tools.gripper)
    except Exception as e:
        robot.node.get_logger().warn(f"Failed to attach gripper: {e}")

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
        hw_rect     = pygame.Rect(cx, ctrl_y + sc(20), cw, row_h)
        input_rect  = pygame.Rect(cx, hw_rect.bottom + row_gap, cw, row_h)
        if selected_input_method == "controller":
            mapping_rect = pygame.Rect(cx, input_rect.bottom + row_gap, cw, row_h)
            legend_y = mapping_rect.bottom + sc(12)
        else:
            mapping_rect = None
            legend_y = input_rect.bottom + sc(12)

        legend_lines = (
            MAPPINGS[selected_mapping]["legend"] if selected_input_method == "controller"
            else KEYBOARD_LEGEND if selected_input_method == "keyboard"
            else ["On-screen buttons  →  right panel", "O / P:  Gripper"]
        ) + [f"Gripper pos:  {gripper_pos:.3f} m"]
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
        ]
        kin_rows_y    = [kin_y + sc(22) + i * (kin_row_h + kin_row_gap) for i in range(4)]
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

        # Telemetry card (top of right panel)
        tel_card = pygame.Rect(rp.x + sc(14), rp.y + sc(14), rp.w - sc(28), sc(400))
        btn_card = pygame.Rect(rp.x + sc(14), tel_card.bottom + sc(14), rp.w - sc(28),
                               rp.bottom - tel_card.bottom - sc(28))

        # Update axis button rects
        for b, (bx, by, bw, bh, _, _col) in zip(btns, BTN_DEFS):
            b.rect = pygame.Rect(sc(bx), sc(by), sc(bw), sc(bh))

        mouse_pos = pygame.mouse.get_pos()

        # --- EVENT HANDLING ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            ep = event.pos if hasattr(event, "pos") else None
            if selected_input_method == "buttons":
                for b in btns:
                    b.handle_event(event, ep)

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

        # --- INPUT ---
        keys = pygame.key.get_pressed()
        if selected_input_method == "controller":
            if joystick:
                target_lin_vel, target_ang_vel = apply_mapping(joystick, selected_mapping, l_spd, a_spd)
                if selected_mapping == "drone":
                    if read_button(joystick, 4): gripper_pos += GRIPPER_SPEED * dt_sec
                    if read_button(joystick, 5): gripper_pos -= GRIPPER_SPEED * dt_sec
                elif selected_mapping == "standard":
                    if read_button(joystick, 1): gripper_pos += GRIPPER_SPEED * dt_sec  # Circle
                    if read_button(joystick, 0): gripper_pos -= GRIPPER_SPEED * dt_sec  # X
        elif selected_input_method == "keyboard":
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
        elif selected_input_method == "buttons":
            for b in btns:
                b.is_pressed = b.mouse_down
            if btns[0].is_pressed:  target_lin_vel[0] += l_spd
            if btns[1].is_pressed:  target_lin_vel[0] -= l_spd
            if btns[2].is_pressed:  target_lin_vel[1] += l_spd
            if btns[3].is_pressed:  target_lin_vel[1] -= l_spd
            if btns[4].is_pressed:  target_lin_vel[2] += l_spd
            if btns[5].is_pressed:  target_lin_vel[2] -= l_spd
            if btns[6].is_pressed:  target_ang_vel[1] += a_spd
            if btns[7].is_pressed:  target_ang_vel[1] -= a_spd
            if btns[8].is_pressed:  target_ang_vel[2] += a_spd
            if btns[9].is_pressed:  target_ang_vel[2] -= a_spd
            if btns[10].is_pressed: target_ang_vel[0] += a_spd
            if btns[11].is_pressed: target_ang_vel[0] -= a_spd

        if keys[pygame.K_o]: gripper_pos += GRIPPER_SPEED * dt_sec
        if keys[pygame.K_p]: gripper_pos -= GRIPPER_SPEED * dt_sec
        gripper_pos = float(np.clip(gripper_pos, GRIPPER_MIN, GRIPPER_MAX))

        current_lin_vel = limit_acceleration(current_lin_vel, target_lin_vel, max_lin_accel, dt_sec)
        current_ang_vel = limit_acceleration(current_ang_vel, target_ang_vel, max_ang_accel, dt_sec)
        robot.cartesian_space.twist(tuple(current_lin_vel), tuple(current_ang_vel))

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

        # Button controls card
        if selected_input_method == "buttons":
            draw_card(screen, btn_card, br=sc(8))
            btn_title = label_font.render("BUTTON CONTROLS", True, HEADER_COLOR)
            screen.blit(btn_title, (btn_card.centerx - btn_title.get_width() // 2, btn_card.y + sc(14)))

            # Group background cards
            for gx, gy, gw, gh in (BTN_GROUP_LINEAR, BTN_GROUP_ROTATION):
                grect = pygame.Rect(sc(gx), sc(gy), sc(gw), sc(gh))
                pygame.draw.rect(screen, (22, 24, 36), grect, border_radius=sc(10))
                pygame.draw.rect(screen, BORDER_COLOR, grect, 1, border_radius=sc(10))

            # Group labels
            lin_lbl  = sec_font.render("TRANSLATION", True, DIM_COLOR)
            rot_lbl  = sec_font.render("ROTATION", True, DIM_COLOR)
            lin_cx = sc(BTN_GROUP_LINEAR[0])   + sc(BTN_GROUP_LINEAR[2]) // 2
            rot_cx = sc(BTN_GROUP_ROTATION[0]) + sc(BTN_GROUP_ROTATION[2]) // 2
            lbl_y  = sc(BTN_GROUP_LINEAR[1]) - lin_lbl.get_height() - sc(4)
            screen.blit(lin_lbl, (lin_cx - lin_lbl.get_width() // 2, lbl_y))
            screen.blit(rot_lbl, (rot_cx - rot_lbl.get_width() // 2, lbl_y))

            for b in btns:
                b.draw(screen, label_font, br=sc(6))
        else:
            # Velocity bars (visual feedback) in the lower card
            draw_card(screen, btn_card, br=sc(8))
            vel_title = label_font.render("VELOCITY FEEDBACK", True, HEADER_COLOR)
            screen.blit(vel_title, (btn_card.centerx - vel_title.get_width() // 2, btn_card.y + sc(14)))

            bar_labels = ["Vx", "Vy", "Vz", "ωx", "ωy", "ωz"]
            bar_vals   = list(current_lin_vel) + list(current_ang_vel)
            bar_maxes  = [l_spd] * 3 + [a_spd] * 3
            bar_h    = sc(18)
            bar_step = sc(34)
            lbl_w  = max(font.size(lbl)[0] for lbl in bar_labels) + sc(12)
            val_w  = font.size("+0.000")[0] + sc(10)
            bar_x  = btn_card.x + sc(14) + lbl_w
            bar_w  = btn_card.w - sc(28) - lbl_w - val_w
            bar_y0 = btn_card.y + sc(56)
            for i, (lbl, val, mx) in enumerate(zip(bar_labels, bar_vals, bar_maxes)):
                by = bar_y0 + i * bar_step
                lbl_img = font.render(lbl, True, DIM_COLOR)
                screen.blit(lbl_img, (btn_card.x + sc(14), by + bar_h // 2 - lbl_img.get_height() // 2))
                pygame.draw.rect(screen, CARD_BG, pygame.Rect(bar_x, by, bar_w, bar_h), border_radius=sc(3))
                pygame.draw.rect(screen, BORDER_COLOR, pygame.Rect(bar_x, by, bar_w, bar_h), 1, border_radius=sc(3))
                if mx > 0:
                    frac = abs(val) / mx
                    fill_w = int(bar_w * min(frac, 1.0))
                    bar_color = ACCENT_COLOR if val >= 0 else (220, 80, 80)
                    if fill_w > 0:
                        pygame.draw.rect(screen, bar_color,
                                         pygame.Rect(bar_x, by, fill_w, bar_h), border_radius=sc(3))
                val_img = font.render(f"{val:+.3f}", True, TEXT_COLOR)
                screen.blit(val_img, (bar_x + bar_w + sc(6), by + bar_h // 2 - val_img.get_height() // 2))

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
