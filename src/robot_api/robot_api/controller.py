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
TEXT_COLOR = (240, 240, 240)
HEADER_COLOR = (0, 255, 150)
GRIPPER_MIN = 0.0
GRIPPER_MAX = 1.0
GRIPPER_SPEED = 1.2  # units/s in [GRIPPER_MIN, GRIPPER_MAX]

def apply_deadzone(value, deadzone=0.1):
    return value if abs(value) > deadzone else 0.0

def main():
    pygame.init()
    pygame.joystick.init()
    
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Robot Command Center - Drone Style Mapping")
    
    font = pygame.font.SysFont("Consolas", 14) 
    big_font = pygame.font.SysFont("Arial", 18, bold=True)
    clock = pygame.time.Clock()

    # Speed Multipliers
    L_SPD = 0.05  # Linear m/s
    A_SPD = 0.6   # Angular rad/s
    gripper_pos = 0.0
    last_sent_gripper_pos = None

    # Ensure gripper command path is available.
    try:
        robot.tool_changer.attach_tool(robot.tools.gripper)
    except Exception as e:
        robot.node.get_logger().warn(f"Failed to attach gripper tool: {e}")

    while True:
        screen.fill(BG_COLOR)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return

        lin_vel = np.zeros(3)
        ang_vel = np.zeros(3)

        if joystick:
            # --- LEFT STICK: Up/Down & Yaw ---
            # Axis 1 (Vertical), Axis 0 (Horizontal)
            lin_vel[2] = -apply_deadzone(joystick.get_axis(1)) * L_SPD # Z (Up/Down)
            ang_vel[2] = apply_deadzone(joystick.get_axis(0)) * A_SPD  # Yaw

            # --- RIGHT STICK: Forward/Side ---
            # Axis 4 (Vertical), Axis 3 (Horizontal) 
            # Note: PS4 Axis indices can vary by OS; 4/3 is common for Right Stick
            lin_vel[0] = -apply_deadzone(joystick.get_axis(4)) * L_SPD # X (Forward/Back)
            lin_vel[1] = -apply_deadzone(joystick.get_axis(3)) * L_SPD  # Y (Side/Side)

            # --- D-PAD (The "Cross thing"): Pitch & Roll ---
            # get_hat(0) returns (x, y) where x is -1 to 1 and y is -1 to 1
            dpad = joystick.get_hat(0)
            if dpad[1] != 0: # D-pad Up/Down
                ang_vel[1] = dpad[1] * A_SPD # Pitch
            if dpad[0] != 0: # D-pad Left/Right
                ang_vel[0] = dpad[0] * A_SPD # Roll

            # Shoulder buttons: open/close gripper.
            # Typical PS-style mapping: L1=4, R1=5.
            if joystick.get_numbuttons() > 5:
                if joystick.get_button(4):
                    gripper_pos += GRIPPER_SPEED * (clock.get_time() / 1000.0)
                if joystick.get_button(5):
                    gripper_pos -= GRIPPER_SPEED * (clock.get_time() / 1000.0)

        keys = pygame.key.get_pressed()
        # Keyboard fallback for gripper control.
        if keys[pygame.K_o]:
            gripper_pos += GRIPPER_SPEED * (clock.get_time() / 1000.0)
        if keys[pygame.K_p]:
            gripper_pos -= GRIPPER_SPEED * (clock.get_time() / 1000.0)
        gripper_pos = float(np.clip(gripper_pos, GRIPPER_MIN, GRIPPER_MAX))

        # Execute movement
        robot.cartesian_space.twist(tuple(lin_vel), tuple(ang_vel))
        if last_sent_gripper_pos is None or abs(gripper_pos - last_sent_gripper_pos) > 1e-3:
            try:
                robot.tools.gripper.set_distance(gripper_pos)
                last_sent_gripper_pos = gripper_pos
            except RuntimeError:
                # Gripper not attached as current tool.
                pass

        # --- DRAW INTERFACE ---
        pose = robot.cartesian_space.read()
        joint_pt = robot.joint_space.read()

        pygame.draw.rect(screen, PANEL_COLOR, (30, 50, 400, 480), border_radius=10)
        pygame.draw.rect(screen, PANEL_COLOR, (460, 50, 660, 400), border_radius=10)

        # Controls Legend
        screen.blit(big_font.render("REMOTE MAPPING", True, HEADER_COLOR), (50, 70))
        controls = [
            "L-Stick Vert:  Up / Down",
            "L-Stick Horiz: Yaw (Rotate)",
            "R-Stick Vert:  Forward / Back",
            "R-Stick Horiz: Side-to-Side",
            "D-Pad Vert:    Pitch",
            "D-Pad Horiz:   Roll",
            "L1/R1:         Gripper Open/Close",
            "O/P keys:      Gripper Open/Close",
            f"Gripper Cmd:   {gripper_pos: .3f}"
        ]
        for i, text in enumerate(controls):
            screen.blit(font.render(text, True, TEXT_COLOR), (60, 110 + i*30))

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
        for i, (txt, f, c) in enumerate(c_lines): screen.blit(f.render(txt, True, c), (490, 80 + i*28))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    try: main()
    finally: robot.shutdown(); pygame.quit()
