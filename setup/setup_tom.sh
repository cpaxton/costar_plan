
export ROS_DISTRO=kinetic
export CTP_FILE=setup_kinetic.sh\?token\=ADImZ3UI07E04DzgwetRl93sHErSv9yyks5aL5uBwA%3D%3D

wget https://raw.githubusercontent.com/cpaxton/costar_stack/master/install_indigo.sh
chmod +x install_indigo.sh
source ./install_indigo.sh

wget https://raw.githubusercontent.com/cpaxton/costar_plan/master/setup/setup_kinetic.sh?token=ADImZ3UI07E04DzgwetRl93sHErSv9yyks5aL5uBwA%3D%3D I
chmod +x $CTP_FILE
source ./$CTP_FILE

export COSTAR_WS="$HOME/costar_ws"
export COSTAR_PLAN_DIR="$HOME/costar_ws/src/costar_plan"

cd $COSTAR_WS/src/
git clone git@github.com:cpaxton/tom_robot.git
git clone git@github.com:cpaxton/costar_plan.git

# Disable things that won't work
touch $COSTAR_WS/src/ur_modern_driver/CATKIN_IGNORE
touch $COSTAR_PLAN_DIR/costar_gazebo_plugins/CATKIN_IGNORE
touch $COSTAR_WS/src/robotiq/robotiq_s_model_articulated_gazebo_plugins/CATKIN_IGNORE
touch $COSTAR_WS/src/costar_stack/costar_perception/sp_segmenter/CATKIN_IGNORE
touch $COSTAR_WS/src/costar_stack/costar_predicator/predicator_8020_module/CATKIN_IGNORE

