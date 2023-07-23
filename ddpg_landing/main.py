#!/usr/bin/env python3

import numpy as np
from agent import Agent
from environment import Environment
import rospy, time, pickle
from utils import plot_learning_curve
import tensorflow as tf
import os

# ############### MAIN LOOP ################
class Main:
  def __init__(self, episodes, state_shape, action_dim, counter):
    self.episodes = episodes
    self.agent = Agent(state_shape, action_dim, TAU, alpha, beta, gamma, memory_size, batch_size, init_e, final_e, decay_steps)
    self.env = Environment(r_constants, max_time, max_dist, change_target_period, bad_ep_threshold)
    
    self.best_score = -100000

    self.figure_file = "learning_curve.png"
    self.score_history = []
    self.steps_in_episode = []

    self.folder_path = "metrics_" + str(counter)
    pickle.dump(counter, open("counter", "wb"))
  
  def train(self):
    for episode in range(self.episodes):
      state = self.env.initiate_episode(episode)
      terminal = False
      score = 0
      step = 0
      print("")
      while not terminal:
        start_time = rospy.Time.now()
        print("")
        print("episode n => ", episode, " ---------------------------- step => ", step)
        print("target => ", self.env.target)
        action = self.agent.select_action(state) 

        reward, next_state, done = self.env.execute_action(action)
        reward = np.round(reward, decimals=4)
        self.agent.store_transition(state, action, reward, next_state, done)     

        score += reward
        step += 1
        self.agent.learn(done)
        self.agent.epsilon.decay_epsilon()
        state = next_state
        if done == 1:
          terminal = True
        duration = rospy.Time.now() - start_time
        rospy.sleep(rospy.Duration(1) - duration)
      self.score_history.append(score)
      self.steps_in_episode.append(step)
      score_avg = np.mean(self.score_history[-100:])
      if score_avg > self.best_score and episode >= 100:
        self.agent.save_models()
        self.best_score = score_avg
    x = [i+1 for i in range(self.episodes)]
    plot_learning_curve(x, self.score_history, self.figure_file, "avg reward over 100 timesteps")
    
  def evaluate(self):
    for episode in range(100):
      state = self.env.initiate_episode(episode, eval=True)
      terminal = False
      score = 0
      step = 0
      print("")

      while not terminal:
        start_time = rospy.Time.now()
        print("")
        print("episode n => ", episode, " ---------------------------- step => ", step)
        print("target => ", self.env.target)
        state = tf.convert_to_tensor([state], dtype=tf.float32)

        action = self.agent.actor(state)[0] 
        reward, next_state, done = self.env.execute_action(action)
        score += reward
        state = next_state
        if done == 1:
          terminal = True
        step += 1
        duration = rospy.Time.now() - start_time
        rospy.sleep(rospy.Duration(1) - duration)
      self.score_history.append(score)
      self.steps_in_episode.append(step)
    result = np.mean(np.array(self.score_history))
    print(f"avg score for 100 episode is: {result}")
    x = [i+1 for i in range(100)]
    plot_learning_curve(x, self.score_history, "evaluate_learning_curve", "average reward over 100 timesteps")
    


if __name__ == "__main__":

  state_shape = (3, )
  action_dim = 2

  #=====================================#
  #====# edit every parameter here #====#
  #=====================================#
  
  ## agent hyperparamteres
  alpha=0.0001
  beta=0.0001

  TAU=0.001
  gamma=0.99

  memory_size=5000
  batch_size=256

  ## noise
  init_e = 0.9
  final_e = 0.005
  decay_steps = 60000


  ## enviornment
  change_target_period = 1 # change target every how many episodes 
  bad_ep_threshold = 20
  max_time = 150
  max_dist = 95 

  ## reward constants
  # (crash, dist, angle, time)
  r_constants = [1]

  if not os.path.exists("counter"):
    counter = 0 
  else:
    counter = pickle.load(open("counter", "rb"))
    counter += 1
  try:
    os.mkdir("metrics_"+str(counter))
  except Exception as e:
    print(e)
  try:
    rospy.init_node("ddpg_agent")
    m = Main(10000, state_shape, action_dim, counter)
    time.sleep(0.5)
    m.train()
    # m.evaluate()
  except Exception as e:
    print(e)
  finally:
    print(m.env.success_ep)
    pickle.dump(m.score_history, open(m.folder_path+"/score_history", "wb"))
    pickle.dump(m.agent.actor_loss, open(m.folder_path+"/actor_loss", "wb"))
    pickle.dump(m.agent.critic_loss, open(m.folder_path+"/critic_loss", "wb"))
    pickle.dump(m.steps_in_episode, open(m.folder_path+"/steps_in_episodes", "wb"))
    pickle.dump(sum(m.steps_in_episode), open(m.folder_path+"/total_steps", "wb"))
    pickle.dump(m.env.success_ep, open(m.folder_path+"/success_ep", "wb"))
    print("metrics saved")