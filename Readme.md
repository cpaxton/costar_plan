# CoSTAR Task Planner (CTP)

[![Build Status](https://travis-ci.com/cpaxton/costar_plan.svg?token=13PmLzWGjzrfxQvEyWp1&branch=master)](https://travis-ci.com/cpaxton/costar_plan)

The CoSTAR Planner is part of the larger [CoSTAR project](https://github.com/cpaxton/costar_stack/). It integrates some learning from demonstration and task planning capabilities into the larger CoSTAR framework in different ways..

Specifically it is a project for creating task and motion planning algorithms that use machine learning to solve challenging problems in a variety of domains. This code provides a testbed for complex task and motion planning search algorithms. The goal is to describe example problems where actor must move around in the world and plan complex interactions with other actors or the environment that correspond to high-level symbolic states.

To run deep learning examples, you will need TensorFlow and Keras, plus a number of Python packages. To run robot experiments, you'll need a simulator (Gazebo or PyBullet), and ROS Indigo or Kinetic. Other versions of ROS may work but have not been tested. If you want to stick to the toy examples, you do not need to use this as a ROS package.

Contents:
  - [0. Introduction](docs/introduction.md)
  - [1. Installation Guide](docs/install.md)
    - [1.1 Docker Instructions](docs/docker_instructions.md)
    - [1.2 Application domains](docs/domains.md)
  - [2. Approach](docs/approach.md): about CTP
    - [2.1 Software Design](docs/design.md): high-level notes
  - [3. Machine Learning Models](docs/learning.md): using the command line tool
    - [3.1 Data collection](docs/collect_data.md): data collection with a real or simulated robot
    - [3.2 MARCC instructions](docs/marcc.md): learning models using JHU's MARCC cluste
    - [3.3 Generative Adversarial Models](docs/learning_gan.md)
  - [4. Creating and training a custom task](docs/task_learning.md): overview of task representations
    - [4.1 Generating a task from data](docs/generate_task_model.md): generate task from demonstrations
    - [4.2 Task Learning](docs/task_learning_experiments.md): specific details
  - [5. CoSTAR Simulation](docs/simulation.md): simulation intro
    - [5.1 Simulation Experiments](docs/simulation-experiments.md): information on experiment setup
    - [5.2 PyBullet Sim](docs/pybullet.md): an alternative to Gazebo that may be preferrable in some situations
    - [5.3 costar_bullet quick start](docs/costar_bullet.md): How to run tasks, generate datasets, train models, and extend costar_bullet with your own components.
    - [5.4 Adding a robot to the ROS code](docs/add_a_robot.md): NOT using Bullet sim
  - [6. Husky robot](docs/husky.md): Start the APL Husky sim
  - [7. TOM robot](docs/tom.md): introducing the TOM robot from TUM
    - [7.1 TOM Data](docs/tom_data.md): data necessary for TOM
    - [7.2 The Real TOM](docs/tom_real_robot.md): details about parts of the system for running on the real TOM

Package/folder layout:
  - [CoSTAR Simulation](costar_simulation/Readme.md): Gazebo simulation and ROS execution
  - [CoSTAR Task Plan](costar_task_plan/Readme.md): the high-level python planning library
  - [CoSTAR Gazebo Plugins](costar_gazebo_plugins/Readme.md): assorted plugins for integration
  - [CoSTAR Models](costar_models/Readme.md): tools for learning deep neural networks
  - [Costar Task Plan - Tom](ctp_tom/Readme.md): specific bringup and scenarios for the TOM robot from TU Munich
  - `setup`: contains setup scripts
  - `slurm`: contains SLURM scripts for running on MARCC
  - `command`: contains scripts with example CTP command-line calls
  - `docs`: markdown files for information that is not specific to a particular ROS package but to all of CTP
  - `photos`: example images
  - `learning_planning_msgs`: ROS messages for data collection when doing learning from demonstration in ROS
  - Others are temporary packages for various projects

Many of these sections are a work in progress; if you have any questions shoot me an email (`cpaxton@jhu.edu`).
## Contact

This code is maintained by Chris Paxton (cpaxton@jhu.edu).

