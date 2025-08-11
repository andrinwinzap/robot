from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Get URDF via xacro
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([
            FindPackageShare("robot_description"),
            "urdf",
            "robot.urdf",
        ]),
    ])
    robot_description = {"robot_description": robot_description_content}

    controller_manager_config = PathJoinSubstitution([
        FindPackageShare("robot_bringup"), "config", "robot_controllers.yaml"
    ])

    rviz_config_file = PathJoinSubstitution([
        FindPackageShare("robot_description"),
        "rviz",
        "view_robot.rviz"
    ])

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[controller_manager_config],
        output="both",
    )

    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
    )

    joint_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller"],
    )

    robot_motion_node = Node(
        package="robot_motion",
        executable="robot_motion_node",
        output="both",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
    )

    return LaunchDescription([
        control_node,
        robot_state_pub_node,
        joint_state_broadcaster_spawner,
        joint_trajectory_controller_spawner,
        robot_motion_node,
        rviz_node
    ])
