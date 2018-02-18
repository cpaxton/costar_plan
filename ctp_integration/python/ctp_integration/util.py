
'''
This function contains some helpful functions for different purposes.
'''

from std_srvs.srv import Empty as EmptySrv

import rospy

def GetDetectObjectsService(srv='/costar_perception/segmenter'):
    '''
    Get a service that will update object positions

    Parameters:
    ----------
    srv: service, defaults to the correct name for costar
    '''
    rospy.wait_for_service(srv)
    return rospy.ServiceProxy(srv, EmptySrv)

def GetCloseGripperService(srv="/costar/gripper/close"):
    rospy.wait_for_service(srv)
    return rospy.ServiceProxy(srv, EmptySrv)

def GetCloseGripperService(srv="/costar/gripper/open"):
    rospy.wait_for_service(srv)
    return rospy.ServiceProxy(srv, EmptySrv)
