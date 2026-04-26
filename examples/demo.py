from robot_api import Robot, JointSpace, CartesianSpace

import math
import numpy as np
import time
import argparse

JOINT_SPACE_SPEED = 0.1
CARTESIAN_SPACE_SPEED = 0.03
CARTESIAN_SPACE_ACCELERATION = 0.03


def pick_and_place(robot):

    ########################################
    START_X = 0.25
    START_Y = 0.5

    END_X = 0.25
    END_Y = 0.2

    PICK_PLACE_HEIGHT = 0.03
    TRAVEL_HEIGHT = 0.1

    OBJECT_SIZE = 0.018
    ########################################

    robot.joint_space.speed = JOINT_SPACE_SPEED
    robot.cartesian_space.linear_speed = CARTESIAN_SPACE_SPEED
    robot.cartesian_space.acceleration = CARTESIAN_SPACE_ACCELERATION

    robot.tool_changer.attach_tool(robot.tools.gripper)

    robot.cartesian_space.move(
        CartesianSpace.Pose((START_X, START_Y, TRAVEL_HEIGHT), (0, 0, 0)), False
    )

    robot.tools.gripper.set_distance(0.05)

    time.sleep(1)

    robot.cartesian_space.move(
        CartesianSpace.Pose((START_X, START_Y, PICK_PLACE_HEIGHT), (0, 0, 0))
    )

    robot.tools.gripper.set_distance(OBJECT_SIZE)

    time.sleep(1)

    robot.cartesian_space.move(
        CartesianSpace.Pose((START_X, START_Y, TRAVEL_HEIGHT), (0, 0, 0))
    )

    robot.cartesian_space.move(
        CartesianSpace.Pose((END_X, END_Y, TRAVEL_HEIGHT), (0, 0, 0))
    )

    robot.cartesian_space.move(
        CartesianSpace.Pose((END_X, END_Y, PICK_PLACE_HEIGHT), (0, 0, 0))
    )

    robot.tools.gripper.set_distance(0.05)

    time.sleep(1)

    robot.cartesian_space.move(
        CartesianSpace.Pose((END_X, END_Y, TRAVEL_HEIGHT), (0, 0, 0))
    )


def sine(robot):

    ########################################
    X0 = 0.175
    Y0 = 0.2
    Z0 = 0.1

    WAVE_LENGTH = 0.3
    NUM_PERIODS = 2
    WAVE_AMPLITUDE = 0.03
    NUM_POINTS = 50
    ########################################

    robot.joint_space.speed = JOINT_SPACE_SPEED
    robot.cartesian_space.linear_speed = CARTESIAN_SPACE_SPEED
    robot.cartesian_space.acceleration = CARTESIAN_SPACE_ACCELERATION

    robot.cartesian_space.move(
        CartesianSpace.Pose((X0, Y0, Z0), (0, 0, 0)), False
    )

    # Calculate total arc length of the sine wave path
    # Use numerical integration to get accurate path length
    num_samples = 1000
    total_distance = 0.0
    for i in range(num_samples):
        alpha1 = i / num_samples
        alpha2 = (i + 1) / num_samples

        y1 = Y0 + alpha1 * WAVE_LENGTH
        x1 = X0 + WAVE_AMPLITUDE * np.sin(2 * np.pi * alpha1 * NUM_PERIODS)

        y2 = Y0 + alpha2 * WAVE_LENGTH
        x2 = X0 + WAVE_AMPLITUDE * np.sin(2 * np.pi * alpha2 * NUM_PERIODS)

        segment_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_distance += segment_length

    # Calculate trapezoidal profile timing
    t_accel = (
        robot.cartesian_space.linear_speed / robot.cartesian_space.acceleration
    )  # Time to reach max speed
    d_accel = (
        0.5 * robot.cartesian_space.acceleration * t_accel**2
    )  # Distance during acceleration
    d_decel = d_accel  # Distance during deceleration (symmetric)

    # Check if we can reach max speed
    if 2 * d_accel > total_distance:
        # Triangular profile - never reach max speed
        t_accel = np.sqrt(total_distance / robot.cartesian_space.acceleration)
        t_cruise = 0
        t_decel = t_accel
        d_accel = 0.5 * total_distance
        d_cruise = 0
        d_decel = 0.5 * total_distance
        actual_max_speed = robot.cartesian_space.acceleration * t_accel
    else:
        # Trapezoidal profile - reach max speed
        d_cruise = total_distance - 2 * d_accel
        t_cruise = d_cruise / robot.cartesian_space.linear_speed
        t_decel = t_accel
        actual_max_speed = robot.cartesian_space.linear_speed

    total_time = t_accel + t_cruise + t_decel

    def trapezoidal_profile(t):
        """
        Returns normalized position (0 to 1) for given time.
        """
        if t <= t_accel:
            # Acceleration phase
            s = 0.5 * robot.cartesian_space.acceleration * t**2
            return s / total_distance
        elif t <= t_accel + t_cruise:
            # Constant velocity phase
            s = d_accel + actual_max_speed * (t - t_accel)
            return s / total_distance
        elif t <= total_time:
            # Deceleration phase
            t_dec = t - t_accel - t_cruise
            s = (
                d_accel
                + d_cruise
                + actual_max_speed * t_dec
                - 0.5 * robot.cartesian_space.acceleration * t_dec**2
            )
            return s / total_distance
        else:
            return 1.0

    path = CartesianSpace.Path()
    for i in range(NUM_POINTS):
        t = (i / (NUM_POINTS - 1)) * total_time  # Actual time
        alpha = trapezoidal_profile(t)  # Position fraction with trapezoid profile

        y = Y0 + alpha * WAVE_LENGTH
        x = X0 + WAVE_AMPLITUDE * np.sin(2 * np.pi * alpha * NUM_PERIODS)
        z = Z0

        pose = CartesianSpace.Pose(
            position=(x, y, z),
            orientation=(0, 0, 0),
            time_from_start=t,
        )
        path.add(pose)

    robot.cartesian_space.follow_path(path)


def circle(robot):

    ########################################
    X0 = 0.2
    Y0 = 0.35
    Z0 = 0.1

    RADIUS = 0.07
    NUM_POINTS = 60
    ########################################

    robot.joint_space.speed = JOINT_SPACE_SPEED
    robot.cartesian_space.linear_speed = CARTESIAN_SPACE_SPEED
    robot.cartesian_space.acceleration = CARTESIAN_SPACE_ACCELERATION

    robot.cartesian_space.move(
        CartesianSpace.Pose((X0 + RADIUS, Y0, Z0), (0, 0, 0)), True
    )

    total_distance = 2 * math.pi * RADIUS

    t_accel = robot.cartesian_space.linear_speed / robot.cartesian_space.acceleration
    d_accel = 0.5 * robot.cartesian_space.acceleration * t_accel**2

    if 2 * d_accel > total_distance:
        t_accel = math.sqrt(total_distance / robot.cartesian_space.acceleration)
        t_cruise = 0
        t_decel = t_accel
        d_accel = 0.5 * total_distance
        d_cruise = 0
        d_decel = 0.5 * total_distance
        actual_max_speed = robot.cartesian_space.acceleration * t_accel
    else:
        d_cruise = total_distance - 2 * d_accel
        t_cruise = d_cruise / robot.cartesian_space.linear_speed
        t_decel = t_accel
        d_decel = d_accel
        actual_max_speed = robot.cartesian_space.linear_speed

    total_time = t_accel + t_cruise + t_decel

    def trapezoidal_profile(t):
        if t <= t_accel:
            s = 0.5 * robot.cartesian_space.acceleration * t**2
        elif t <= t_accel + t_cruise:
            s = d_accel + actual_max_speed * (t - t_accel)
        elif t <= total_time:
            t_dec = t - t_accel - t_cruise
            s = d_accel + d_cruise + actual_max_speed * t_dec - 0.5 * robot.cartesian_space.acceleration * t_dec**2
        else:
            s = total_distance
        return s / total_distance

    path = CartesianSpace.Path()
    for i in range(NUM_POINTS):
        t = (i / (NUM_POINTS - 1)) * total_time
        alpha = trapezoidal_profile(t)

        angle = 2 * math.pi * alpha
        x = X0 + RADIUS * math.cos(angle)
        y = Y0 + RADIUS * math.sin(angle)

        pose = CartesianSpace.Pose(
            position=(x, y, Z0),
            orientation=(0, 0, 0),
            time_from_start=t,
        )
        path.add(pose)

    robot.cartesian_space.follow_path(path)


def cone_motion(robot):

    ########################################
    POSITION = (0.25, 0.35, 0.1)
    CONE_ANGLE = math.radians(15)
    PERIOD = 10.0
    ROTATIONS = 2
    ########################################

    robot.joint_space.speed = JOINT_SPACE_SPEED

    print("Moving to start pose...")
    robot.cartesian_space.move(
        CartesianSpace.Pose(POSITION, (0.0, CONE_ANGLE, 0.0)),
        enforce_linearity=False,
    )

    print("Executing cone motion...")
    omega = 2 * math.pi / PERIOD
    total_time = PERIOD * ROTATIONS

    t_start = time.time()
    while True:
        t = time.time() - t_start
        if t >= total_time:
            break
        theta = omega * t

        pitch = CONE_ANGLE * math.cos(theta)
        roll_dot = CONE_ANGLE * omega * math.cos(theta)
        pitch_dot = -CONE_ANGLE * omega * math.sin(theta)

        cp = math.cos(pitch)
        sp = math.sin(pitch)
        robot.cartesian_space.twist(
            [0.0, 0.0, 0.0],
            [-cp * roll_dot, -pitch_dot, -sp * roll_dot],
        )
        time.sleep(0.01)

    robot.cartesian_space.twist([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def estop(robot):

    ########################################
    X0 = 0.2
    Y0 = 0.35
    Z0 = 0.05
    ########################################

    robot.joint_space.speed = JOINT_SPACE_SPEED
    robot.cartesian_space.linear_speed = CARTESIAN_SPACE_SPEED
    robot.cartesian_space.acceleration = CARTESIAN_SPACE_ACCELERATION

    robot.cartesian_space.move(
        CartesianSpace.Pose((X0, Y0, Z0), (0, 0, 0)), False
    )

    robot.cartesian_space.move(
        CartesianSpace.Pose((X0, Y0, Z0 - 0.01), (0, 0, 0)), False
    )


def main():

    parser = argparse.ArgumentParser(description="Robot Demo CLI")

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--fake-hardware", action="store_true", help="Enable fake hardware mode"
    )
    args = parser.parse_args()

    print("\nWelcome to the robot demo wizard!\n")

    print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
    print(f"Fake hardware mode: {'ON' if args.fake_hardware else 'OFF'}")

    print("\nStarting Setup...")

    robot = Robot()
    robot.set_fake_hardware_mode(args.fake_hardware)
    robot.set_debug_mode(args.debug)

    print("Setup Complete!")

    while True:
        print("\nAvailable commands:")
        print("1. Pick and Place")
        print("2. Sine Wave")
        print("3. Circle")
        print("4. Cone Motion")
        print("5. E-STOP")
        print("6. Attach Tool")
        print("7. Detach Tool")

        choice = input("\nEnter your choice: ")
        print()

        if choice == "":
            break
        elif choice == "1":
            pick_and_place(robot)
        elif choice == "2":
            sine(robot)
        elif choice == "3":
            circle(robot)
        elif choice == "4":
            cone_motion(robot)
        elif choice == "5":
            estop(robot)
        elif choice == "6":
            print("Available tools:")
            print("1. Gripper")

            choice = input("\nEnter your choice: ")
            print()
            if choice == "1":
                robot.tool_changer.attach_tool(robot.tools.gripper)
            else:
                print("Invalid choice. Please try again.")

        elif choice == "7":
            robot.tool_changer.detach_tool()
        else:
            print("Invalid choice. Please try again.")

    print("Exiting...")
    robot.shutdown()


if __name__ == "__main__":
    main()
