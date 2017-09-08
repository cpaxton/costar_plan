import rospy

from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose
from cv_bridge import CvBridge, CvBridgeError
import cv2


gazeboModelsTopic = "/gazebo/model_states"
overheadImageTopic = "/overhead/camera/image_raw"
huskyGoalTopic = "/move_base_simple/goal"

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


poseDictionary = {'Dumpster': dumpsterPose, 'Barrier' : barrierPose, 'Barrel' :  constructionBarrelPose, 'Fire Hydrant': fireHydrantPose}

dumpsterPose = None
barrierPose = None

bridge = CvBridge()

gazeboModels = None


def imageCallback(data):
    
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg 
        cv2.imwrite('camera_image.jpeg', cv2_img)

def modelsCallback(data):
    global gazeboModels
    
    gazeboModels = data    
    
#    dumpster_index = -1
#    jersey_barrier_index = -1
#    
#    global objectList 
#    objectList = data.name
#    
#    for i in range(len(data.name)):
#        if data.name[i] == "Dumpster_0":
#            dumpster_index = i
#        elif data.name[i] == "jersey_barrier_15":
#            jersey_barrier_index = i
#        
#            
#    if dumpster_index == -1:
#        print "dumpster not found!"
#        return
#    
#    if jersey_barrier_index == -1:
#        print "jersey barrier not found!"
#        return
#
#
#    dumpsterPose = data.pose[dumpster_index]
#    barrierPose = data.pose[jersey_barrier_index]    
#    #print "dumpster position ", dumpsterPose.position.x, ", ", dumpsterPose.position.y
#    #print "barrier position ", barrierPose.position.x, ", ", barrierPose.position.y
        
    

def listener():
    # setup subscribers
    rospy.init_node('costar_simulator_husky', anonymous=True)
    
    #rospy.Subscriber(gazeboModelsTopic, ModelStates, modelsCallback)
    rospy.Subscriber(overheadImageTopic, Image, imageCallback)

    # object names and poses

    # overhead camera

    

    

def getObjectList():
    #global gazeboModels
    #if gazeboModels != None:
    #    return gazeboModels.name
    #else:
    #    return None

    return list(poseDictionary.keys())

def getOverheadCamera():
    pass

def moveToObject(objectName):
    pass


if __name__ == '__main__':
    listener()
    global gazeboModels
    goalPub = rospy.Publisher(huskyGoalTopic, PoseStamped, queue_size=1)
    
    while not rospy.is_shutdown():
        if getObjectList() != None:

            print "select object from ", getObjectList()
            objectName = raw_input()
            if objectName == '':
                continue
            
            #objectIndex = gazeboModels.name.index(objectName)  
            poseStampedMsg = PoseStamped()
            poseStampedMsg.header.frame_id = "map"
            poseStampedMsg.header.stamp = rospy.Time.now()
            #poseStampedMsg.pose = gazeboModels.pose[objectIndex]
            poseStampedMsg.pose = poseDictionary[objectName]
            #poseStampedMsg.pose.position.x = gazeboModels.pose[objectIndex].position.y
            #poseStampedMsg.pose.position.y = gazeboModels.pose[objectIndex].position.x
            #poseStampedMsg.pose = gazeboModels.pose[objectIndex]
            
            print "position is ", poseStampedMsg.pose           
            goalPub.publish(poseStampedMsg)
         
            
    
    
    #rospy.spin()
    