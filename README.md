# DDPG approach for Ar Drone Autonomy using dualbooted Linux
This project was built upon the package ['sjtu-drone'](https://github.com/tahsinkose/sjtu-drone.git) and it implements DDPG for both Navigation and Landing using Ar Drone 2.0

## ROS and Gazebo Install and Setup
```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt install ros-noetic-desktop-full
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Dependencies Setup
```
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
sudo apt install python3-catkin-tools python3-rospy python3-pip libignition-math4-dev
sudo rosdep init
rosdep update
pip3 install tensorflow==2.2.0
pip3 install protobuf==3.20.*
sudo apt-get update
```

## Workspace Setup
```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/tahsinkose/sjtu-drone.git
git clone https://github.com/FabianReister/gazebo_aruco_box.git
cd sjtu-drone/src
git clone https://github.com/Ahmed-S-Ahmed/DDPG-approach-for-Ar-Drone-Autonomy.git
cd catkin_ws
catkin init
catkin build
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
roslaunch sjtu_drone simple.launch
```

## To utilize Nvidia GPU for Tensorflow
- Download GPU drivers from [Nvidia](https://www.nvidia.com/Download/index.aspx?lang=en-us)
```
chmod +x <file>
sudo ./<file>
sudo apt install nvidia-cuda-toolkit
```
- Reboot your system
- Make an Nvidia account and sign up to developer program 
Download cuDNN v7.5.1 for CUDA 10.1 from the [archive](https://developer.nvidia.com/rdp/cudnn-archive)
- Go into your cuDNN file location
```
sudo dpkg -i libcudnn7_7.5.1.10-1+cuda10.1_amd64.deb
sudo apt-get update
```