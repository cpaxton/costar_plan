#!/usr/bin/env python

from __future__ import print_function

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose, Twist
import numpy as np
from costar_models.datasets.npz import NpzDataset
import tf
from tf import TransformListener

from scipy.misc import imresize
import cv2
from cv_bridge import CvBridge, CvBridgeError
import random




gazeboModelsTopic = "/gazebo/model_states"
overheadImageTopic = "/overhead/camera/image_raw"
huskyGoalTopic = "/move_base_simple/goal"
huskyCmdVelTopic = "/husky_velocity_controller/cmd_vel"

dumpsterPose = Pose()
dumpsterPose.position.x = -1.135
dumpsterPose.position.y = -3.856
dumpsterPose.position.z = 0
dumpsterPose.orientation.x = 0
dumpsterPose.orientation.y = 0
dumpsterPose.orientation.z = .959
dumpsterPose.orientation.w = -0.2847

barrierPose = Pose()
barrierPose.position.x = 1.325
barrierPose.position.y = -0.89
barrierPose.position.z = 0
barrierPose.orientation.x = 0
barrierPose.orientation.y = 0
barrierPose.orientation.z = .459
barrierPose.orientation.w = 0.888

constructionBarrelPose = Pose()
constructionBarrelPose.position.x = 1.0088
constructionBarrelPose.position.y = -4.96
constructionBarrelPose.position.z = 0
constructionBarrelPose.orientation.x = 0
constructionBarrelPose.orientation.y = 0
constructionBarrelPose.orientation.z = -0.5729
constructionBarrelPose.orientation.w = 0.81957

fireHydrantPose = Pose()
fireHydrantPose.position.x = 4.02
fireHydrantPose.position.y = -5.114
fireHydrantPose.position.z = 0
fireHydrantPose.orientation.x = 0
fireHydrantPose.orientation.y = 0
fireHydrantPose.orientation.z = 0.8945
fireHydrantPose.orientation.w = -0.447


#store the positions as ints, have a dictionary with keys, robots pose, orientation, and action, sequence (trial number), top down image is 5
#randomize the ordering of targets
poseDictionary = {
        'Dumpster': dumpsterPose,
        'Barrier' : barrierPose,
        'Barrel' :  constructionBarrelPose,
        'Fire Hydrant': fireHydrantPose}

#globals
name_to_dtypes = {
	"rgb8":    (np.uint8,  3),
	"rgba8":   (np.uint8,  4),
	"rgb16":   (np.uint16, 3),
	"rgba16":  (np.uint16, 4),
	"bgr8":    (np.uint8,  3),
	"bgra8":   (np.uint8,  4),
	"bgr16":   (np.uint16, 3),
	"bgra16":  (np.uint16, 4),
	"mono8":   (np.uint8,  1),
	"mono16":  (np.uint16, 1),}

def imageToNumpy(msg):
	if not msg.encoding in name_to_dtypes:
		raise TypeError('Unrecognized encoding {}'.format(msg.encoding))
	
	dtype_class, channels = name_to_dtypes[msg.encoding]
	dtype = np.dtype(dtype_class)
	dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
	shape = (msg.height, msg.width, channels)

	data = np.fromstring(msg.data, dtype=dtype).reshape(shape)
	data.strides = (
		msg.step,
		dtype.itemsize * channels,
		dtype.itemsize
	)

	if channels == 1:
		data = data[...,0]
	return data

class HuskyDataCollector(object):
    def __init__(self):
        self.gazeboModels = None
        self.img_np = None
        self.pose = Odometry()
        self.action = Twist()

        rospy.Subscriber(overheadImageTopic, Image, self.imageCallback)
        rospy.Subscriber(huskyCmdVelTopic, Twist, self.cmdVelCallback)

        self.listener = tf.TransformListener()
        self.writer = NpzDataset('husky_data')

        self.trans = None
        self.rot = None
        self.roll, self.pitch, self.yaw = 0., 0., 0.

        self.trans_threshold = 0.25

    def imageCallback(self, data):
        img_np_bk = imageToNumpy(data)
        #print img_np_bk.shape
        self.img_np = imresize(img_np_bk, (64, 64))
        #print self.img_np.shape
    
    def cmdVelCallback(self, data):
        self.action = data

    def modelsCallback(self, data):
        self.gazeboModels = data    

    def write(self, *args, **kwargs):
        self.writer.write(*args, **kwargs)
    
    # returns true if goal has been reached, else false    
    def goalReached(self, finalPose):
        goal_xyz = np.array([finalPose.position.x,
                finalPose.position.y,
                finalPose.position.z])
        xyz = np.array(self.trans)
        dist = np.linalg.norm(goal_xyz - xyz)

        return dist < self.trans_threshold
      
    def getObjectList(self):
        return list(poseDictionary.keys())

    def getOverheadCamera(self):
        pass

    def moveToObject(self, objectName):
        pass

    def finishCurrentExample(example, max_label=-1):
        '''
        Preprocess this particular example:
        - split it up into different time windows of various sizes
        - compute task result
        - compute transition points
        - compute option-level (mid-level) labels
        '''
        # ============================================================
        # Split into chunks and preprocess the data.
        # This may require setting up window_length, etc.
        next_list = ["reward", "done", "example", "label",
                "image",
                "pose",
                "action"]
        # -- NOTE: you can add other features here in the future, but for now
        # we do not need these. Label gets some unique handling.
        prev_list  = []
        first_list = ["image", "pose"]
        goal_list = ["image", "pose"]
        length = len(current_example['label'])

        # Create an empty dict to hold all the data from this last trial.
        data = {}
        data["prev_label"] = []

        # Compute the points where the data changes from one label to the next
        # and save these points as "goals".
        switches = []
        count = 1
        label = current_example['label']
        for i in xrange(length):
            if i+1 == length:
                switches += [i] * count
                count = 1
            elif not label[i+1] == label[i]:
                switches += [i+1] * count
                count = 1
            else:
                count += 1
        #print (len(switches),  len(current_example['example']))
        assert(len(switches) == len(current_example['example']))

        # ============================================
        # Set up total reward
        total_reward = np.sum(current_example["reward"])
        data["value"] = [total_reward] * len(current_example["example"])

        # ============================================
        # Loop over all entries. For important items, take the previous frame
        # and the next frame -- and possibly even the final frame.
        prev_label = max_label
        for i in xrange(length):
            i0 = max(i-1,0)
            i1 = min(i+1,length-1)
            ifirst = 0

            # We will always include frames where the label changed. We may or
            # may not include frames where the 
            if current_example["label"][i0] == current_example["label"][i1] \
                    and not i0 == 0 \
                    and not i1 == length - 1 \
                    and not np.random.randint(2) == 0:
                        continue

            # ==========================================
            # Finally, add the example to the dataset
            for key, values in current_example.items():
                if not key in data:
                    data[key] = []
                    if key in next_list:
                        data["next_%s"%key] = []
                    if key in prev_list:
                        data["prev_%s"%key] = []
                    if key in first_list:
                        data["first_%s"%key] = []
                    if key in goal_list:
                        data["goal_%s"%key] = []

                # Check data consistency
                if len(data[key]) > 0:
                    if isinstance(values[0], np.ndarray):
                        assert values[0].shape == data[key][0].shape
                    if not type(data[key][0]) == type(values[0]):
                        print(key, type(data[key][0]), type(values[0]))
                        raise RuntimeError('Types do not match when' + \
                                           ' constructing data set.')

                # Append list of features to the whole dataset
                data[key].append(values[i])
                #if key == "label":
                #    data["prev_%s"%key].append(prev_label)
                #    prev_label = values[i]
                if key in prev_list:
                    data["prev_%s"%key].append(values[i0])
                if key in next_list:
                    data["next_%s"%key].append(values[i1])
                if key in first_list:
                    data["first_%s"%key].append(values[ifirst])
                if key in goal_list:
                    data["goal_%s"%key].append(values[switches[i]])

        collector.write(data, current_example['example'][0], total_reward)

    def tick(self):
        try:
            (trans,rot) = self.listener.lookupTransform('/map', '/base_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return False

        self.trans, self.rot = trans, rot
        quaternion = (rot[0], rot[1], rot[2], rot[3])
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.roll = euler[0]
        self.pitch = euler[1]
        self.yaw = euler[2]
        return True
    
if __name__ == '__main__':
    rospy.init_node('costar_husky_data_collection')
    collector = HuskyDataCollector()
    goalPub = rospy.Publisher(huskyGoalTopic, PoseStamped, queue_size=1)
    seqNumber = 0
    prev_label = -1
    print("================================================")
    print("Starting HUSKY data collector!")
    print("---")
    print("This will dump all examples into the 'husky_data' folder.")
    print("If you are having trouble, make sure ROS is not having trouble ")
    print("synchronizing time; there is a rospy.sleep() here to help with")
    print("that.")
    rospy.sleep(0.5)

    try:
        rate = rospy.Rate(10) # 10hz
        
        # Loop as long as we are runnin
        #for objectName in poseDictionary:
        while not rospy.is_shutdown():

            print("================================================")
            print("EXAMPLE NUMBER = ", seqNumber)
            
            current_example = {}
            current_example['reward'] = list()
            current_example['done'] = list()
            current_example['example'] = list()
            current_example['label'] = list()
            current_example['prev_label'] = list()
            current_example['image'] = list()  
            current_example['pose'] = list()
            current_example['action'] = list()
        
            # Make sure we are getting poses
            while not collector.tick():
                rate.sleep()
            
            objectName = None
            prevObjectName = None
            failed = False
            for i in range(3):
                # select random objectName
                while objectName == prevObjectName:
                    objectName = random.sample(poseDictionary.keys(), 1)[0]
                prevObjectName = objectName
                print("Random objective:", objectName)
                
                poseStampedMsg = PoseStamped()
                poseStampedMsg.header.frame_id = "map"
                poseStampedMsg.header.stamp = rospy.Time.now()
                poseStampedMsg.pose = poseDictionary[objectName]
                
                print ("position is ", poseStampedMsg.pose)
                goalPub.publish(poseStampedMsg)

                # Loop until destination has been reached
                max_iter = 150
                iterations = 0
                while (iterations < max_iter):

                    at_goal = collector.goalReached(poseStampedMsg.pose)
                    # get global variables and write
                    if at_goal:
                        print(" ---> SUCCESS!")
                        current_example['reward'].append(10)
                    elif not at_goal and iterations == max_iter -1:
                        print(" ---> FAILED!")
                        current_example['reward'].append(-100)
                        failed = True
                    else:
                        current_example['reward'].append(0)

                    if iterations == max_iter - 1 or at_goal:
                        current_example['done'].append(1)
                    else:
                        current_example['done'].append(0)
                    current_example['example'].append(seqNumber)
                    current_example['image'].append(collector.img_np)
                    current_example['pose'].append([
                        collector.trans[0], collector.trans[1], collector.trans[2],
                        collector.roll, collector.pitch, collector.yaw])
                    
                    action = collector.action
                    current_example['action'].append([
                        action.linear.x, action.linear.y, action.linear.z,
                        action.angular.x, action.angular.y, action.angular.z])
                    
                    current_example['label'].append(poseDictionary.keys().index(objectName))
                    current_example['prev_label'].append(prev_label)
                    
                    iterations = iterations + 1
                    if not collector.tick():
                        raise RuntimeError("collection lost contact with TF for some reason")

                    if at_goal:
                        break

                    rate.sleep()

                if failed:
                    break
                
            print ("writing sample")
            collector.finishCurrentExample(current_example)
            seqNumber = seqNumber + 1
            print ("prev_label", prev_label)
            prev_label = poseDictionary.keys().index(objectName)
            print ("prev_label after update", prev_label)

    except rospy.ROSInterruptException as e:
        pass
    
