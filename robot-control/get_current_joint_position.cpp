// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <cmath>
#include <iostream>
 
#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/robot_state.h>


int main(int argc, char** argv) {

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <robot-hostname> " << std::endl;
    return -1;
  }
 
  try {
    franka::Robot robot(argv[1]);
 
    franka::RobotState robot_state = robot.readOnce();
    std::array<double, 7> current_position = robot_state.q;

    std::cout << "Current joint positions: ";
    for (const double& joint : current_position) {
      std::cout << joint << " ";
    }
    std::cout << std::endl;
    


  } catch (franka::Exception const& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }
 
  return 0;
}