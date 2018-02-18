
# Real Robot Experiments with CoSTAR

These all require the [CoSTAR stack](git@github.com:cpaxton/costar_stack.git).

## Starting Real Robot

Just run:
```
roslaunch ctp_integration bringup.launch
```

This is configured specifically to work with the JHU UR5, and may not work (or may require some adaptation) if you want to use it with a different robot.

## Debugging

### Debugging CoSTAR Arm

The `CostarArm` class manages activity like the SmartMoves we use to manipulate objects.

You can just kill the UR5 server script and restart it with:
```
rosnode kill /simple_ur5_driver_node
rosrun costar_robot_manager ur_driver.py
```

This can be useful if you want to debug any particular aspect of the Arm, such as the SmartMoves.
