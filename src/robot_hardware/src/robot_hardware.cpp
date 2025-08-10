// Copyright 2023 ros2_control Development Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "robot_hardware/robot_hardware.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <chrono>

namespace robot_hardware
{

  CallbackReturn RobotSystem::on_init(const hardware_interface::HardwareInfo &info)
  {
    if (hardware_interface::SystemInterface::on_init(info) != CallbackReturn::SUCCESS)
    {
      return CallbackReturn::ERROR;
    }

    // Initialize atomic flag
    should_stop_.store(false);

    try
    {
      // Create ROS 2 node for hardware interface
      node_ = rclcpp::Node::make_shared("robot_hardware_interface");

      // Create executor
      executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
      executor_->add_node(node_);

      const size_t num_joints = info_.joints.size();

      // Resize vectors
      joint_positions_.resize(num_joints, 0.0);
      joint_velocities_.resize(num_joints, 0.0);
      joint_commands_.resize(num_joints, 0.0);
      command_publishers_.resize(num_joints);
      position_subscribers_.resize(num_joints);
      velocity_subscribers_.resize(num_joints);

      RCLCPP_INFO(node_->get_logger(), "Initializing hardware interface with %zu joints", num_joints);

      // Create publishers and subscribers for each joint
      for (size_t i = 0; i < num_joints; i++)
      {
        const std::string &joint_name = info_.joints[i].name;

        std::string cmd_topic = "/robot/" + joint_name + "/send_command";
        command_publishers_[i] = node_->create_publisher<std_msgs::msg::Float32MultiArray>(cmd_topic, 10);

        // Subscriber to micro-ROS get_position topic
        std::string get_position_topic = "/robot/" + joint_name + "/get_position";
        position_subscribers_[i] = node_->create_subscription<std_msgs::msg::Float32>(
            get_position_topic, 10,
            [this, i, joint_name](std_msgs::msg::Float32::SharedPtr msg)
            {
              std::lock_guard<std::mutex> lock(this->joint_state_mutex_);
              this->joint_positions_[i] = static_cast<double>(msg->data);
              RCLCPP_DEBUG(this->node_->get_logger(),
                           "Received position for joint %s: %f", joint_name.c_str(), msg->data);
            });

        // Subscriber to micro-ROS get_velocity topic
        std::string get_velocity_topic = "/robot/" + joint_name + "/get_velocity";
        velocity_subscribers_[i] = node_->create_subscription<std_msgs::msg::Float32>(
            get_velocity_topic, 10,
            [this, i, joint_name](std_msgs::msg::Float32::SharedPtr msg)
            {
              std::lock_guard<std::mutex> lock(this->joint_state_mutex_);
              this->joint_velocities_[i] = static_cast<double>(msg->data);
              RCLCPP_DEBUG(this->node_->get_logger(),
                           "Received velocity for joint %s: %f", joint_name.c_str(), msg->data);
            });

        RCLCPP_INFO(node_->get_logger(),
                    "Created publisher/subscriber for joint '%s' (topics: %s, %s, %s)",
                    joint_name.c_str(), cmd_topic.c_str(), get_position_topic.c_str(), get_velocity_topic.c_str());
      }
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(rclcpp::get_logger("RobotSystem"), "Failed to initialize: %s", e.what());
      return CallbackReturn::ERROR;
    }

    RCLCPP_INFO(node_->get_logger(), "Hardware interface initialized successfully");
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn RobotSystem::on_configure(const rclcpp_lifecycle::State & /*previous_state*/)
  {
    RCLCPP_INFO(node_->get_logger(), "Configuring hardware interface");

    // Reset joint states and commands
    {
      std::lock_guard<std::mutex> lock(joint_state_mutex_);
      for (auto &pos : joint_positions_)
        pos = 0.0;
      for (auto &vel : joint_velocities_)
        vel = 0.0;
      for (auto &cmd : joint_commands_)
        cmd = 0.0;
    }

    // Reset ros2_control interfaces
    for (const auto &[name, descr] : joint_state_interfaces_)
    {
      set_state(name, 0.0);
    }
    for (const auto &[name, descr] : joint_command_interfaces_)
    {
      set_command(name, 0.0);
    }
    for (const auto &[name, descr] : sensor_state_interfaces_)
    {
      set_state(name, 0.0);
    }

    RCLCPP_INFO(node_->get_logger(), "Hardware interface configured successfully");
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn RobotSystem::on_activate(const rclcpp_lifecycle::State & /*previous_state*/)
  {
    RCLCPP_INFO(node_->get_logger(), "Activating hardware interface");

    should_stop_.store(false);

    // Start the spin thread
    spin_thread_ = std::thread([this]()
                               {
    RCLCPP_INFO(this->node_->get_logger(), "Starting ROS executor thread");
    
    while (!this->should_stop_.load() && rclcpp::ok())
    {
      try 
      {
        this->executor_->spin_once(std::chrono::milliseconds(10));
      }
      catch (const std::exception& e)
      {
        RCLCPP_ERROR(this->node_->get_logger(), "Exception in executor thread: %s", e.what());
        break;
      }
    }
    
    RCLCPP_INFO(this->node_->get_logger(), "ROS executor thread stopped"); });

    RCLCPP_INFO(node_->get_logger(), "Hardware interface activated successfully");
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn RobotSystem::on_deactivate(const rclcpp_lifecycle::State & /*previous_state*/)
  {
    RCLCPP_INFO(node_->get_logger(), "Deactivating hardware interface");

    // Signal thread to stop
    should_stop_.store(true);

    // Wait for thread to finish
    if (spin_thread_.joinable())
    {
      spin_thread_.join();
      RCLCPP_INFO(node_->get_logger(), "Executor thread joined successfully");
    }

    RCLCPP_INFO(node_->get_logger(), "Hardware interface deactivated successfully");
    return CallbackReturn::SUCCESS;
  }

  return_type RobotSystem::read(const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
  {
    // Copy current micro-ROS joint positions and velocities into ros2_control joint state interfaces
    std::lock_guard<std::mutex> lock(joint_state_mutex_);

    for (size_t i = 0; i < info_.joints.size(); i++)
    {
      const std::string pos_name = info_.joints[i].name + "/" + hardware_interface::HW_IF_POSITION;
      const std::string vel_name = info_.joints[i].name + "/" + hardware_interface::HW_IF_VELOCITY;

      set_state(pos_name, joint_positions_[i]);
      set_state(vel_name, joint_velocities_[i]);

      // For debugging - can be removed in production
      RCLCPP_DEBUG(node_->get_logger(),
                   "Joint %s position: %f, velocity: %f",
                   info_.joints[i].name.c_str(), joint_positions_[i], joint_velocities_[i]);
    }

    return return_type::OK;
  }

  return_type RobotSystem::write(const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
  {
    // Send ros2_control joint commands (position and velocity) to micro-ROS topics
    for (size_t i = 0; i < info_.joints.size(); i++)
    {
      // Fetch commands from ros2_control for position and velocity
      const std::string pos_cmd_name = info_.joints[i].name + "/" + hardware_interface::HW_IF_POSITION;
      const std::string vel_cmd_name = info_.joints[i].name + "/" + hardware_interface::HW_IF_VELOCITY;
      double position_cmd = get_command(pos_cmd_name);
      double velocity_cmd = get_command(vel_cmd_name);

      // Pack commands into a Float32MultiArray
      std_msgs::msg::Float32MultiArray msg;
      msg.data.resize(2);
      msg.data[0] = static_cast<float>(position_cmd);
      msg.data[1] = static_cast<float>(velocity_cmd);

      // Publish to the joint's topic
      command_publishers_[i]->publish(msg);

      // For debugging
      RCLCPP_DEBUG(node_->get_logger(),
                   "Commanding joint %s: position %.3f, velocity %.3f",
                   info_.joints[i].name.c_str(), position_cmd, velocity_cmd);
    }

    return return_type::OK;
  }

} // namespace robot_hardware

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(
    robot_hardware::RobotSystem, hardware_interface::SystemInterface)