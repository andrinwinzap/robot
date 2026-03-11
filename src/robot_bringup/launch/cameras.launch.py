from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    cam0 = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        namespace='camera0',
        name='v4l2_camera',
        parameters=[{
            'video_device': '/dev/video0',
            'image_width': 640,
            'image_height': 480,
            'framerate': 30
        }]
    )

    cam1 = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        namespace='camera1',
        name='v4l2_camera',
        parameters=[{
            'video_device': '/dev/video8',
            'image_width': 640,
            'image_height': 480,
            'framerate': 30
        }]
    )

    return LaunchDescription([cam0, cam1])