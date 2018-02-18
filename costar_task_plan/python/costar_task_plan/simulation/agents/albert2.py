from abstract import AbstractAgent
import pybullet as pb
import PyKDL as kdl
from costar_task_plan.simulation.world import *

class Albert2Agent(AbstractAgent):
    '''
    Really simple test agent that just generates a random set of positions to
    move to.
    '''

    name = "albert2"

    def __init__(self, env, *args, **kwargs):
        super(Albert2Agent, self).__init__(*args, **kwargs)
        self.env = env

    def fit(self, num_iter):
        keylist = []
        sizeCounter = 0
        #a = pb.getKeyboardEvents()
        
        for i in xrange(num_iter):
            print "---- Iteration %d ----"%(i+1)
            #self.env.reset()
            a = pb.getKeyboardEvents()

            while not self._break:
                token = 0
                state = self.env.world.actors[0].state
                control = SimulationRobotAction(arm_cmd=None, gripper_cmd=None)

                a = pb.getKeyboardEvents()
                if a != {}: print a
                # y opens the gripper
                if 121 in a:  
                    print "y detected"
                    keylist.append("y")
                    # arm = 
                    token = 121
                    gripper_cmd = state.robot.gripperOpenCommand()
                    control = SimulationRobotAction(arm_cmd=None, gripper_cmd=gripper_cmd)
                    
                elif 122 in a:  
                    # arm =
                    print "z detected"
                    keylist.append("z")
                    token = 122
                    gripper_cmd = state.robot.gripperCloseCommand()
                    control = SimulationRobotAction(arm_cmd=None, gripper_cmd=gripper_cmd)
                    
       #############################Forwards##########################
    
                #if 119 in a:  
                    #print "w detected"
                    #token = 119
                if 49 in a:
                    print "1 detected"
                    keylist.append("1")
                    token = 49
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0.05,0,0))
                    T_arm = origin * move
                    #Transformed arm
                    gripper_cmd = None
                    #state.arm = joints
                    invarm = state.robot.ik(T_arm, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd)
                    print control
                    
                #if 97 in a:  
                    #print "a detected"
                    #token = 97
                if 51 in a:
                    print "3 detected"
                    keylist.append("3")
                    token = 51
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0,0.05,0))
                    T_arm = origin * move
                    #Transformed arm
                    gripper_cmd = None
                    #state.arm = joints
                    invarm = state.robot.ik(T_arm, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd)
                
                #if 114 in a:  
                    #print "r detected"
                    #token = 114
                if 53 in a:
                    print "5 detected"
                    keylist.append("5")
                    token = 53
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0,0,0.05))
                    T_arm = origin * move
                    #Transformed arm
                    gripper_cmd = None
                    #state.arm = joints
                    invarm = state.robot.ik(T_arm, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd)
            
            
    #############################Rotation#################################

                #if 113 in a:  
                    #print "q detected"
                    #token = 113
                if 55 in a:
                    print "7 detected"
                    keylist.append("7")
                    token = 55
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.RotX(0.5), kdl.Vector.Zero())
                    T_arm = origin * move
                    #Transformed arm
                    gripper_cmd = None
                    #state.arm = joints
                    invarm = state.robot.ik(T_arm, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd)
                    
    
                if 57 in a:
                    print "9 detected"
                    keylist.append("9")
                    token = 57
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.RotY(0.5), kdl.Vector.Zero())
                    T_arm = origin * move
                    #Transformed arm
                    gripper_cmd = None
                    #state.arm = joints
                    invarm = state.robot.ik(T_arm, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd)
                
                #if 114 in a:  
                    #print "r detected"
                    #token = 114
                if 45 in a:
                    print "= detected"
                    keylist.append("-")
                    token = 53
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.RotZ(0.5), kdl.Vector.Zero())
                    T_arm = origin * move
                    #Transformed arm
                    gripper_cmd = None
                    #state.arm = joints
                    invarm = state.robot.ik(T_arm, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd)
      
    ############################Backwards#########################
    
                #if 115 in a:  
                    #print "s detected"
                    #token = 115
                if 50 in a:
                    print "2 detected"
                    keylist.append("2")
                    token = 50
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(-0.05,0,0))
                    T_arm = origin * move
                    #Transformed arm
                    gripper_cmd = None
                    #state.arm = joints
                    invarm = state.robot.ik(T_arm, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd)
                
                
                #if 100 in a:  
                    #print "d detected"
                    #token = 100
                if 52 in a:
                    print "4 detected"
                    keylist.append("4")
                    token = 52
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0,-0.05,0))
                    T_arm = origin * move
                    #Transformed arm
                    gripper_cmd = None
                    #state.arm = joints
                    invarm = state.robot.ik(T_arm, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd)
                
                #if 102 in a:  
                    #print "f detected"
                    #token = 102
                if 54 in a:
                    print "6 detected"
                    keylist.append("6")
                    token = 54
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0,0,-0.05))
                    T_arm = origin * move
                    #Transformed arm
                    gripper_cmd = None
                    #state.arm = joints
                    invarm = state.robot.ik(T_arm, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd)
  

#########################################INV_Rotations##########################################

                #if 101 in a:  
                    #print "e detected"
                    #token = 101
                if 56 in a:
                    print "8 detected"
                    keylist.append("8")
                    token = 56
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.RotX(-0.5), kdl.Vector.Zero())
                    T_arm = origin * move
                    #Transformed arm
                    gripper_cmd = None
                    #state.arm = joints
                    invarm = state.robot.ik(T_arm, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd)
                    
                    
                if 48 in a:
                    print "0 detected"
                    keylist.append("0")
                    token = 48
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.RotY(-0.5), kdl.Vector.Zero())
                    T_arm = origin * move
                    #Transformed arm
                    gripper_cmd = None
                    #state.arm = joints
                    invarm = state.robot.ik(T_arm, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd)
                    
                if 61 in a:
                    print "= detected"
                    keylist.append("=")
                    token = 61
                    origin = state.T
                    move = kdl.Frame(kdl.Rotation.RotZ(-0.5), kdl.Vector.Zero())
                    T_arm = origin * move
                    #Transformed arm
                    gripper_cmd = None
                    #state.arm = joints
                    invarm = state.robot.ik(T_arm, state.arm)
                    control = SimulationRobotAction(arm_cmd=invarm, gripper_cmd=gripper_cmd)
                    
                if 104 in a:
                    print("---------Help requested--------------")
                    print(" ------------------------------------");
                    print("| Costar Keyboard Teleop Help          |");
                    print("|------------------------------------|*");
                    print("| Current Mode: Arm Control          |*");
                    print("|------------------------------------|*");
                    print("| 1/2 : forward/backward translation |*");
                    print("| 3/4 : left/right translation       |*");
                    print("| 5/6 : up/down translation          |*");
                    print("| 7/8 : roll                         |*");
                    print("| up/down : pitch                    |*");
                    print("| left/right : yaw                   |*");
                    print("| 2 : switch to Finger Control       |*");
                    print(" ------------------------------------**");
                    print("  *************************************");
                
                if control is not None:
                    features, reward, done, info = self.env.step(control)
                    '''
                    self._addToDataset(self.env.world,
                            control,
                            features,
                            reward,
                            done,
                            i,
                            token)
                            #i,
                            #names[plan.idx])
                    '''
                    if done: #and len(keylist) > sizeCounter:
                        #sizeCounter+=1
                        break

                else:
                    break
            
            print "end statement reached"
            #self.env.step(cmd)

            if self._break:
                return
