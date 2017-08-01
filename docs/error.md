```
-----------------------------------------------------------
Optimizer = adam
Learning Rate =  0.001
Clip Norm =  100
===========================================================
Traceback (most recent call last):
  File "/home/jhuapl/costar_ws/src/costar_plan/costar_bullet/scripts/start", line 27, in <module>
    main(args)
  File "/home/jhuapl/costar_ws/src/costar_plan/costar_bullet/scripts/main.py", line 28, in main
    model.train(**agent.data)
  File "/home/jhuapl/costar_ws/src/costar_plan/costar_task_plan/python/costar_task_plan/models/multi_unsupervised1_model.py", line 108, in train
    self._makeModel(features, arm, gripper, arm_cmd, gripper_cmd)
  File "/home/jhuapl/costar_ws/src/costar_plan/costar_task_plan/python/costar_task_plan/models/multi_unsupervised1_model.py", line 66, in _makeModel
    post_tiling_layers=3,
  File "/home/jhuapl/costar_ws/src/costar_plan/costar_task_plan/python/costar_task_plan/models/robot_multi_models.py", line 327, in GetAlbert1Encoder
    padding='same'))(x)
  File "/usr/local/lib/python2.7/dist-packages/keras/engine/topology.py", line 596, in __call__
    output = self.call(inputs, **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/keras/layers/wrappers.py", line 177, in call
    y = self.layer.call(inputs)  # (num_samples * timesteps, ...)
  File "/usr/local/lib/python2.7/dist-packages/keras/layers/convolutional.py", line 164, in call
    dilation_rate=self.dilation_rate)
  File "/usr/local/lib/python2.7/dist-packages/keras/backend/tensorflow_backend.py", line 3138, in conv2d
    data_format='NHWC')
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/nn_ops.py", line 639, in convolution
    input_channels_dim = input.get_shape()[num_spatial_dims + 1]
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/tensor_shape.py", line 500, in __getitem__
    return self._dims[key]
IndexError: list index out of range
```
