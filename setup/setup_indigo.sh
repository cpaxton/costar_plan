#!/usr/bin/env bash

export ROS_DISTRO=indigo
export ROS_CI_DESKTOP="`lsb_release -cs`"  # e.g. [precise|trusty|...]
export CI_SOURCE_PATH=$(pwd)
export CATKIN_OPTIONS="$CI_SOURCE_PATH/catkin.options"
export ROS_PARALLEL_JOBS='-j8 -l6'
export CATKIN_WS="$HOME/costar_ws"
export COSTAR_PLAN_DIR="$HOME/costar_ws/src/costar_plan"

sudo apt-get update -qq

echo "======================================================"
echo "ROS"
sudo apt-get install -y python-catkin-pkg python-rosdep python-wstool \
  python-catkin-tools ros-$ROS_DISTRO-catkin ros-$ROS_DISTRO-ros-base
echo "--> source ROS setup in /opt/ros/$ROS_DISTRO/setup.bash"
source /opt/ros/$ROS_DISTRO/setup.bash
sudo rosdep init
rosdep update

#echo "======================================================"
#echo "PYTHON"
#echo "Installing python dependencies:"
#echo "Installing basics from apt-get..."
#sudo apt-get -y install python-pygame python-dev
#echo "Installing libraries and drivers..."
#sudo apt-get -y install -y build-essential autoconf libtool pkg-config python-opengl python-imaging python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev libssl-dev
#sudo apt-get -y install -y libx11-dev libpq-dev python-dev libxml2-dev libxslt1-dev libldap2-dev libsasl2-dev libffi-dev mongodb
#echo "Installing smaller libraries from pip..."
#sudo -H pip install --no-binary numpy
#sudo -H pip install h5py keras keras-rl sympy matplotlib pygame gmr networkx \
#  dtw pypr gym PyPNG pybullet numba

# TODO(cpaxton): come up with a better way to install tensorflow here. We want
# to ensure that everything is configured properly for tests.
#if [ nvidia-smi ]
#then
#  sudo -H pip install tensorflow
#else
#  sudo -H pip install tensorflow
#fi

echo "======================================================"
echo "CATKIN"
echo "Create catkin workspace..."
mkdir -p $CATKIN_WS/src
cd $CATKIN_WS
source /opt/ros/$ROS_DISTRO/setup.bash
catkin init
cd $CATKIN_WS/src
echo $PYTHONPATH
catkin build --verbose

#echo "======================================================"
#echo "COSTAR"
#git clone https://github.com/cpaxton/hrl-kdl.git  --branch indigo-devel
#git clone https://github.com/cburbridge/python_pcd.git
#git clone https://github.com/jhu-lcsr/costar_objects.git
#git clone https://github.com/cpaxton/dmp.git --branch indigo
#git clone https://github.com/cpaxton/robotiq_85_gripper.git
##git clone https://github.com/cpaxton/costar_plan.git
#rosdep install -y --from-paths ./ --ignore-src --rosdistro $ROS_DISTRO
#cd $CATKIN_WS/src
#catkin build --verbose
source $CATKIN_WS/devel/setup.bash

