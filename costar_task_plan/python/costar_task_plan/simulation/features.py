from costar_models import GetAvailableFeatures
from costar_task_plan.abstract.features import AbstractFeatures
from costar_task_plan.simulation.camera import ImageData

import numpy as np


def GetFeatures(features):
    '''
    Returns a particular task definition in the simulation.
    '''
    try:
        return {
            '': EmptyFeatures(),
            'null': EmptyFeatures(),
            'empty': EmptyFeatures(),
            'depth': DepthImageFeatures(),
            'joint_state': JointStateFeatures(),
            'rgb': RgbImageFeatures(),
            'multi': ImagePlusFeatures(),
            'pose': PoseFeatures(),
            'grasp_segmentation': GraspSegmentationFeatures(),
        }[features]
    except KeyError, e:
        raise NotImplementedError(
            'Feature function not implemented: %s', str(e))


class EmptyFeatures(AbstractFeatures):

    '''
    This is a very simple set of features. It does, well, nothing at all. It is
    super fast, though, which makes it good for running execution tests.
    '''

    def compute(self, world, state):
        return np.array([0])

    def updateBounds(self, world):
        pass

    def getBounds(self):
        return np.array([0]), np.array([0])


class DepthImageFeatures(AbstractFeatures):

    '''
    The only features we return are the depths associated with each camera pixel.
    So we get 2.5D data here.
    '''

    def compute(self, world, state):
        return world.cameras[0].capture().depth

    def updateBounds(self, world):
        raise Exception('feature.updateBounds not yet implemented!')

    def getBounds(self):
        raise Exception('feature.getBounds not yet implemented!')


class JointStateFeatures(AbstractFeatures):

    def compute(self, world, state):
        return np.append(state.arm, state.gripper)

    def updateBounds(self, world):
        raise Exception('feature.updateBounds not yet implemented!')

    def getBounds(self):
        raise Exception('feature.getBounds not yet implemented!')


class RgbImageFeatures(AbstractFeatures):

    '''
    The only feature data we return will be a single RGB image from the first
    camera placed in the world, where ever that may be.
    '''

    def compute(self, world, state):
        return world.cameras[0].capture().rgb


class ImagePlusFeatures(AbstractFeatures):

    '''
    Include arm, state, and gripper features.
    '''

    def compute(self, world, state):
        img = world.cameras[0].capture().rgb
        return [img[:, :, :3], state.arm, state.gripper]

    @property
    def description(self):
        return ["features", "arm", "gripper"]

class PoseFeatures(AbstractFeatures):
    '''
    Get object poses only. Only makes sense on tasks with a consistent object
    list; otherwise things will not work!
    '''

    def compute(self, world, state):
        '''
        Note that since we are iterating over the keys -- these features will
        all be in the same order, which ends up working very nicely for us.
        '''

        object_translation_rotation = []
        for name, oid in world.id_by_object.items():
            obj = world.actors[oid].getState()
            object_translation_rotation += [obj.T.p]
            object_translation_rotation += list(obj.T.M.GetQuaternion())
        return [np.array(object_translation_rotation),
                state.arm,
                state.gripper]

    @property
    def description(self):
        return ["poses", "arm", "gripper"]

class GraspSegmentationFeatures(AbstractFeatures):

    '''
    This set of features includes data helpful for training segmentation.
    object_translation_rotation, state.arm, state.gripper, image_data, object_surface_points

    object_surface_points is where a ray cast from the camera to the object struck the first
    surface. If the ray never hit any surface

    For instructions to use this feature see `segmentation.md`.

    This also represents all objects in the world as a single vector. This
    means that we need to have a constant size world, where we always have the
    same objects in the same order.
    '''

    def compute(self, world, state):
        import pybullet as pb
        object_translation_rotation = []
        # camera.py ImageData namedtuple
        camera = world.cameras[0]
        image_data = camera.capture()
        image_data_arrays = [np.array(value) for value in image_data] + \
            [camera.matrix, camera.projection_matrix]
        # 'camera_view_matrix' namedtuple index is 4
        # TODO(ahundt) ensure camera matrix translation component is from world origin to camera origin
        # camera ray is from the origin of the camera
        camera_transform_array = np.transpose(image_data[4]).reshape((4, 4))
        # use the more conveniently formatted transform for writing out
        image_data_arrays[4] = camera_transform_array
        camera_translation = camera_transform_array[0:3, 3].tolist()
        # print("TEST IMAGEDATA named tuple img matrix: \n", )
        # print("TEST IMAGEDATA named tuple img matrix translation: ", camera_translation)
        camera_ray_from = [camera_translation] * len(world.id_by_object.items())
        # camera_ray_to is the center of each object
        camera_ray_to = []
        for name, oid in world.id_by_object.items():
            # print("oid type:", str(type(oid)))
            # print("actor type:", str(type(world.actors[oid])))
            obj = world.actors[oid].getState()
            p = obj.T.p
            q = obj.T.M.GetQuaternion()
            one_t_r = np.array([p[0], p[1], p[2], q[0], q[1], q[2], q[3]], dtype=np.float32)
            object_translation_rotation.append(one_t_r)
            camera_ray_to.append(list(obj.T.p))

        # print("lengths: ", len(camera_ray_from), len(camera_ray_to))
        object_surface_points = []
        # TODO(ahundt) allow multiple simulations to run
        raylist = pb.rayTestBatch(camera_ray_from, camera_ray_to)
        for i, (uid, linkidx, hitfrac, hitpos, hitnormal) in enumerate(raylist):
            if uid is -1:
                # if the object wasn't hit, use its origin
                name, oid = world.id_by_object.items()[i]
                obj = world.actors[oid].getState()
                object_surface_points += [obj.T.p]
            else:
                object_surface_points += hitpos

        return [np.array(object_translation_rotation),
                np.array(state.arm),
                np.array(state.gripper)] + image_data_arrays[1:] + [np.array(object_surface_points)]

    @property
    def description(self):
        return ["object_translation_rotation", "arm", "gripper"] + \
                list(ImageData._fields)[1:] + \
                ['camera_view_matrix', 'camera_projection_matrix'] + \
                ["camera_to_object_surface_points"]

  def getBounds(self):
    raise Exception('feature.getBounds not yet implemented!')


class JointStateFeatures(AbstractFeatures):

  def compute(self, world, state):
      return np.append(state.arm, state.gripper)
      

  def updateBounds(self, world):
    raise Exception('feature.updateBounds not yet implemented!')

  def getBounds(self):
    raise Exception('feature.getBounds not yet implemented!')
