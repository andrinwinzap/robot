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

#ifndef robot_HARDWARE__robot_HARDWARE_HPP_
#define robot_HARDWARE__robot_HARDWARE_HPP_

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"
#include <std_msgs/msg/float32_multi_array.hpp>

using hardware_interface::return_type;

namespace robot_hardware
{
  using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

  class RobotSystem : public hardware_interface::SystemInterface
  {
  public:
    CallbackReturn on_init(const hardware_interface::HardwareInfo &info) override;

    CallbackReturn on_configure(const rclcpp_lifecycle::State &previous_state) override;

    CallbackReturn on_activate(const rclcpp_lifecycle::State &previous_state) override;

    CallbackReturn on_deactivate(const rclcpp_lifecycle::State &previous_state) override;

    return_type read(const rclcpp::Time &time, const rclcpp::Duration &period) override;

    return_type write(const rclcpp::Time &time, const rclcpp::Duration &period) override;

  private:
    // ROS 2 Node handle for publishers and subscribers
    rclcpp::Node::SharedPtr node_;

    // Publishers and subscribers for each joint
    std::vector<rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr> command_publishers_;
    std::vector<rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr> state_subscribers_;

    // Internal arrays to hold joint states and commands
    std::vector<double> joint_positions_;
    std::vector<double> joint_velocities_;
    std::vector<double> joint_commands_;

    // Mutex to protect concurrent access to joint_positions_ and joint_velocities_
    std::mutex joint_state_mutex_;

    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
    std::atomic<bool> simulation_mode_;

    // Thread for spinning the ROS node
    std::thread spin_thread_;
    std::atomic<bool> should_stop_;

    // Executor for handling callbacks
    rclcpp::executors::SingleThreadedExecutor::SharedPtr executor_;
  };

} // namespace robot_hardware

#endif // robot_HARDWARE__robot_HARDWARE_HPP_