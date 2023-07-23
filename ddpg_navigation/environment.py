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
from sensor_msgs.msg import Range
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from tf.transformations import euler_from_quaternion

class Environment:
  def __init__(self, reward_constants, max_time, max_dist, change_target_period):
    self.bridge = CvBridge()
    self.start_time = rospy.Time.now()
    self.image_sample = 3
    self.crash = 0
    self.images = []
    self.s = Semaphore()
    self.cs = Semaphore()
    self.ps = Semaphore()
    self.dx = 0
    self.dy = 0
    self.dz = 0
    self.az = 0
    self.ref = ['sjtu_drone','ground_plane']
    self.cur_dist = 0
    self.prev_dist = self.cur_dist
    self.reward_constants = reward_constants
    self.max_time = max_time
    self.max_dist = max_dist
    self.change_target_period = change_target_period
    self.target = [74, -4, 20]
    self.cum_dist_diff = 0

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

    self.t2 = Thread(target=self.pos_thread)
    self.t3 = Thread(target=self.crash_thread)
    self.t2.start()
    self.t3.start()

  def pos_thread(self):
    rospy.Subscriber('/drone/gt_pose', Pose, callback=self.update_pos)

  def crash_thread(self):
    rospy.Subscriber('/drone/sonar', Range, callback=self.iscrash)
  
  def update_pos(self, cords):
    self.ps.acquire()
    self.dx = cords.position.x
    self.dy = cords.position.y
    self.dz = cords.position.z
    q = np.array([cords.orientation.x, cords.orientation.y, cords.orientation.z, cords.orientation.w])
    r = euler_from_quaternion(q)
    self.az = r[2]
    self.ps.release()

  def iscrash(self, ran):
    self.cs.acquire()
    self.crash = ran.range
    self.cs.release()
 
  def get_dist_between_two_coord(self, a, b):
    dist = math.sqrt(pow(a[0]-b[0],2)+pow(a[1]-b[1],2))
    return dist

  def get_current_coords(self):
    self.ps.acquire()
    dx = self.dx
    dy = self.dy
    dz = self.dz
    az = self.az
    self.ps.release()
    return [dx, dy, dz, az]
  
  def expand_dims(self, input):
    output = []
    for i in input:
      i = np.expand_dims(i, axis=-1)
      i = np.expand_dims(i, axis=-1)
      output.append(i)
    return output

  def get_state(self):
    coords = self.get_current_coords()
    
    c_x = coords[0]
    c_y = coords[1]
    t_x = self.target[0]
    t_y = self.target[1]
    dx = min(t_x - c_x, self.max_dist) / self.max_dist
    dy = min(t_y - c_y, self.max_dist) / self.max_dist
    
    az = coords[3] / math.pi
    if dx == 0:
      if dy > 0:
        taz = 0.5
      else: taz = -0.5
    else:
      angle = dy / dx
      taz = math.atan(angle) / math.pi
      if dy < 0 and dx < 0:
        taz -= 1
      elif dx < 0: 
        taz += 1

    self.cs.acquire()
    crash = self.crash / 5
    self.cs.release()

    print("- state -")
    print("(dx, dy, az, taz, crash)")
    print(dx, dy, az, taz, crash)
    return [dx, dy, az, taz, crash]

  def move(self, lx, az):
    print("- taken action -")
    print("(lx, az)")
    print(lx, az)
    self.msg.linear.x = lx
    self.msg.linear.y = 0
    self.msg.linear.z = 0
    self.msg.angular.x = 0
    self.msg.angular.y = 0
    self.msg.angular.z = az
    self.pubCmd.publish(self.msg)

  def calculate_reward(self, state):
    reward = 0
    done = False
    # detect crashing
    self.cs.acquire()
    crash = self.crash
    self.cs.release() 
    coords = self.get_current_coords()
    alt = np.round(coords[2], decimals=2)
  
    if crash <= 0.05:
      reward = -20
      print("crashing")
      done = True
      print(reward, "r")
      return reward, done
    
    dist = np.round(self.get_dist_between_two_coord(self.target, coords), decimals=3)
    self.cur_dist = dist
    # reach goal
    if dist <= 10: # some threshold that indicates the successful completion of the task
      reward += 20
      done = True
      print(reward, "congrats you reached your goal")
      return reward, done

    # danger level
    crash_reward = - 1 + min(crash, 5) / 5

    # angle effectiveness
    dx = state[0]
    dy = state[1]
    if dx == 0:
      if dy > 0:
        taz = 0.5
      else: taz = -0.5
    else:
      angle = dy / dx
      taz = math.atan(angle) / math.pi
      if dy < 0 and dx < 0:
        taz -= 1
      elif dx < 0: 
        taz += 1
    caz = coords[3] / math.pi
    
    angle_diff = abs(taz - caz)
    if angle_diff > 1:
      angle_reward = -(2 - angle_diff)
    else: angle_reward = -angle_diff

    # distance from goal
    if self.prev_dist == 0:
      dist_reward = 0
    else: dist_reward = self.prev_dist - self.cur_dist

    self.cum_dist_diff += dist_reward

    # time reward 
    time_reward = -0.01

    # reward
    reward = (self.reward_constants[0] * crash_reward) + (self.reward_constants[1] * dist_reward) + (self.reward_constants[2] * angle_reward) + (self.reward_constants[3] * time_reward)
    # time consumed
    time_consumed = rospy.Time.now().to_sec() - self.start_time.to_sec()    
    if time_consumed >= self.max_time or self.cum_dist_diff <= -20: # some time duration
      print("time's up")
      reward -= 20
      done = True
    print("- dist info -")
    print(self.cur_dist, self.prev_dist, dist_reward)
    print("- time -", time_consumed)
    print("- reward info -")
    print("total", reward)
    print("dist reward", dist_reward)
    print("crash reward", crash_reward)
    print("angle reward", angle_reward)
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
    state_msg.pose.position.z = 5
    state_msg.pose.orientation.x = 0
    state_msg.pose.orientation.y = 0
    state_msg.pose.orientation.z = 0
    state_msg.pose.orientation.w = 1
    self.respawn(state_msg)

  def callran(self,t=0):
    
    nx = np.random.randint(-70, -30)
    px = np.random.randint(30, 70)
    if abs(nx) > px: x = nx 
    else: x = px
    
    ny = np.random.randint(-70, -30)
    py = np.random.randint(30, 70)
    if abs(ny) > py: y = ny 
    else: y = py

    return [x,y]

  def initiate_episode(self, t=0, eval=False):
    print("================================= new episode ======================================")
    self.cur_dist = 0
    self.prev_dist = self.cur_dist
    self.cum_dist_diff = 0
    
    if (t % self.change_target_period == 0 and not eval) or t == 0:
      num = self.callran(t)
      self.target = [num[0], num[1], 5]

      num = self.callran(t)
      self.origin = [num[0], num[1], 0]

    self.reset_pos()
    self.start_time = rospy.Time.now()
    self.pubTakeOff.publish(Empty())
    self.isFlying = True

    return self.get_state()
