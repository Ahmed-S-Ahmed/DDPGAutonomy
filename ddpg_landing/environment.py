#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import cv2, rospy, math, time
from cv_bridge import CvBridge
from threading import *

from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Bool, Empty
from std_srvs.srv import Empty as em

import message_filters
from sensor_msgs.msg import Image
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from ar_detect import get_features

class Environment:
  def __init__(self, reward_constants, max_time, max_dist, change_target_period, bad_ep_threshold):
    self.bridge = CvBridge()
    self.start_time = rospy.Time.now()
    self.image_sample = 1
    self.crash = 0
    self.images = []
    self.s = Semaphore()
    self.cs = Semaphore()
    self.ps = Semaphore()
    self.dx = 0
    self.dy = 0
    self.dz = 0
    self.inframe = True
    self.ref = ['sjtu_drone','ground_plane']
    self.cur_dist = 0
    self.prev_dist = self.cur_dist
    self.reward_constants = reward_constants
    self.max_time = max_time
    self.max_dist = max_dist
    self.change_target_period = change_target_period
    self.target = []
    self.cum_dist_diff = 0
    self.bad_ep_threshold = bad_ep_threshold
    self.success_ep = 0

    self.respawn = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    self.pubTakeOff = rospy.Publisher("/drone/takeoff", Empty, queue_size=10)
    self.pubLand = rospy.Publisher("/drone/land", Empty, queue_size=10)
    self.pubCmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    self.pubVelMode = rospy.Publisher("/drone/vel_mode", Bool, queue_size=10)
    self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_world', em)

    bool_msg = Bool()
    bool_msg.data = 1
    self.pubVelMode.publish(bool_msg)
    self.isFlying = False
    self.msg = Twist()

    self.t1 = Thread(target=self.image_thread)
    self.t2 = Thread(target=self.pos_thread)
    
    self.t1.start()
    self.t2.start()

  def pos_thread(self):
    rospy.Subscriber('/drone/gt_pose', Pose, callback=self.update_pos)
  
  def image_thread(self):
    self.down_camera = message_filters.Subscriber('/drone/down_camera/image_raw', Image)

    ts = message_filters.ApproximateTimeSynchronizer([self.down_camera], queue_size=10, slop=0.25)
    ts.registerCallback(self.update_image)

  def update_image(self, msg):
    self.s.acquire()
    image = self.bridge.imgmsg_to_cv2(msg)
    [self.dx, self.dy, self.inframe] = get_features(image)
    self.s.release()

  def update_pos(self, cords):
    self.ps.acquire()
    self.dz = cords.position.z
    self.ps.release()
 
  def get_dist_between_two_coord(self, a, b):
    dist = math.sqrt(pow(a[0]-b[0],2)+pow(a[1]-b[1],2))
    return dist

  def get_current_coords(self):
    self.ps.acquire()
    dz = self.dz
    self.ps.release()
    return [dz]

  def get_state(self):
    self.s.acquire()
    dx = self.dx
    dy = self.dy
    inframe = self.inframe
    self.s.release()
    return [dx, dy, inframe]

  def move(self, lx, ly):
    print("- taken action -")
    print("(lx, ly)")
    print(lx, ly)
    self.msg.linear.x = lx*0.5
    self.msg.linear.y = ly*0.5
    self.msg.linear.z = -0.25
    self.msg.angular.x = 0
    self.msg.angular.y = 0
    self.msg.angular.z = 0
    self.pubCmd.publish(self.msg)

  def calculate_reward(self, state):
    reward = 0
    done = False
    dx = state[0] / 320
    dy = state[1] / 180
    inframe = state[2]
    dz = self.get_current_coords()[0]

    # reach goal
    if dz <= 1.5:
      self.success_ep += 1
      reward += 10
      done = True
      print(reward, "landed")
      return reward, done

    # dist reward
    dist = math.sqrt(dx**2 + dy**2)
    self.cur_dist = dist
    print(dist, "distance") # something here for punishment
    dist_reward = 1 - ((dx/0.6667)**2+(dy/0.5)**2)

    self.cum_dist_diff += dist_reward

    # reward
    reward = (self.reward_constants[0] * dist_reward)

    # time consumed
    time_consumed = rospy.Time.now().to_sec() - self.start_time.to_sec()    
    if not inframe: # some time duration # or time_consumed >= self.max_time 
      print("time's up")
      reward -= 20
      done = True
    print("- dist info -")
    print(dx, dy, dz, dist_reward)
    print("- time -", time_consumed)
    print("- reward info -")
    print("total", reward)
    print("dist reward", dist_reward)
    self.prev_dist = self.cur_dist
    return reward, done

  def execute_action(self, action):
    self.move(action[0], action[1])  
    next_state = self.get_state()
    reward, done = self.calculate_reward(next_state)
    if done: x = 1
    else: x = 0
    return (reward, next_state, x)
  
  def reset_pos(self):
    self.reset_simulation()

    self.msg.linear.x = 0
    self.msg.linear.y = 0
    self.msg.linear.z = 0
    self.msg.angular.x = 0
    self.msg.angular.y = 0
    self.msg.angular.z = 0
    self.pubCmd.publish(self.msg)

    state_msg = ModelState()
    state_msg.model_name = self.ref[0]
    state_msg.pose.position.x = self.origin[0]
    state_msg.pose.position.y = self.origin[1]
    state_msg.pose.position.z = 10
    state_msg.pose.orientation.x = 0
    state_msg.pose.orientation.y = 0
    state_msg.pose.orientation.z = 0
    state_msg.pose.orientation.w = 1
    self.respawn(state_msg)

  def callran(self, v):    
    x = np.random.randint(-v, v)
    y = np.random.randint(-v, v)
    return [x,y]

  def initiate_episode(self, t=0, eval=False):
    print("================================= new episode ======================================")
    self.cur_dist = 0
    self.prev_dist = self.cur_dist

    num = self.callran(1)
    self.origin = [num[0], num[1], 0]

    self.reset_pos()
    self.start_time = rospy.Time.now()
    self.pubTakeOff.publish(Empty())
    self.isFlying = True

    return self.get_state()