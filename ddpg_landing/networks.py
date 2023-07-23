#!/usr/bin/env python3
import os
from tensorflow import keras
keras.models
import tensorflow as tf

class Actor(keras.Model):
  def __init__(self, action_dim, hidden_layers=(512, 512, 200, 200), folder_path="tmp/ddpg", name="actor"):
    super(Actor, self).__init__()
    self._hidden_layer = hidden_layers
    self.check_point_file = os.path.join(folder_path+"_"+name, name)

    self.dns1 =  keras.layers.Dense(self._hidden_layer[0], activation="relu")
    self.dns2 = keras.layers.Dense(self._hidden_layer[1], activation="relu")
    self.out = keras.layers.Dense(action_dim, activation="tanh")
  
  def call(self, s):
    x = self.dns1(s)
    x = self.dns2(x)    
    x = self.out(x)
    return x


class Critic(keras.Model):
  def __init__(self, hidden_layers=(512, 512, 512, 100), folder_path="tmp/ddpg", name="critic"):
    super(Critic, self).__init__()
    self._hidden_layer = hidden_layers
    self.check_point_file = os.path.join(folder_path+"_"+name, name)

    self.dns1 = keras.layers.Dense(self._hidden_layer[0], activation="relu")
    self.dns2 = keras.layers.Dense(self._hidden_layer[1], activation="relu")
    self.q = keras.layers.Dense(1, activation=None)
    
  def call(self, s, action):
    s = tf.concat([s, action], axis=-1)
    x = self.dns1(s)    
    x = self.dns2(x)
    x = self.q(x)
    return x