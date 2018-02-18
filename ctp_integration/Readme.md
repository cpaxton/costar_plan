# CTP Integration

Code for testing Costar Task Planner on the real UR5 robot in our lab. This integrates with the [CoSTAR Stack](https://github.com/cpaxton/costar_stack).

## Quick Start

```
roslaunch ctp_integration bringup.launch
rosrun ctp_integration run.py --iter 1000
```

## Guidelines

  - python: contain specific source code
  - nodes: ROS nodes
  - launch: contains launch files

