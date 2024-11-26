// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <cmath>
#include <iostream>
#include <sstream>
 
#include <franka/exception.h>
#include <franka/robot.h>
 
#include "examples_common.h"


int main(int argc, char** argv) {

  if (argc != 9) {
    std::cerr << "Usage: " << argv[0] << " <robot-hostname> q_0 q_1 q_2 q_3 q_4 q_5 q_6" << std::endl;
    return -1;
  }

  try {
    franka::Robot robot(argv[1]);
    setDefaultBehavior(robot);
    

    franka::RobotState robot_state = robot.readOnce();
    std::array<double, 7> initial_position = robot_state.q;

    std::array<double, 7> q_goal;
    for (size_t i = 0; i < 7; i++) {
      q_goal[i] = std::stod(argv[i + 2]);
    }

    std::cout << "Initial joint positions (radians): ";
    for (const double& joint : initial_position) {
      std::cout << joint << " ";
    }
    std::cout << std::endl;

    std::cout << "Target joint positions (radians): ";
    for (const double& joint : q_goal) {
        std::cout << joint << " ";
    }
    std::cout << std::endl;

    
    // speed factor [0,1]
    MotionGenerator motion_generator(0.5, q_goal);
    std::cout << "WARNING: This program will move the robot! "
              << "Please make sure to have the user stop button at hand!" << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    robot.control(motion_generator);
    std::cout << "Finished moving to goal joint configuration." << std::endl;


  } catch (const franka::Exception& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }
 
  return 0;
}