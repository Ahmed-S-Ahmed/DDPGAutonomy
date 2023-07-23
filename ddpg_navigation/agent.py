#!/usr/bin/env python3
import os, random
import tensorflow as tf
from tensorflow import keras
from networks import Actor, Critic
from memory import Memory
# import keras

class DecayedEpsilonGreedy:
  def __init__(self, epsilon, epsilon_min, decay_steps):
    self.epsilon = epsilon
    self.epsilon_min = epsilon_min
    self.decay_steps = decay_steps
    self.decay_rate = (self.epsilon - self.epsilon_min) / self.decay_steps
  
  def decay_epsilon(self):
    if self.epsilon > self.epsilon_min:
      self.epsilon -= self.decay_rate


class Agent:
  def __init__(self, state_shape, action_dim, TAU=0.001, alpha=0.00001, beta=0.00003, gamma=0.99, memory_size=500000, batch_size=512, init_e=0.9, final_e=0.05, decay_steps=600000):
    self.TAU = TAU
    self.gamma = gamma
    self.action_dim = action_dim
    self.memory = Memory(state_shape, action_dim, memory_size)
    self.batch_size = batch_size
    self.epsilon = DecayedEpsilonGreedy(init_e, final_e, decay_steps)

    self.actor = Actor(action_dim)
    self.critic = Critic()
    self.target_actor = Actor(action_dim,  name="actor_target")
    self.target_critic = Critic(name="critic_target")

    self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=alpha))
    self.critic.compile(optimizer=keras.optimizers.Adam(learning_rate=beta))
    self.target_actor.compile(optimizer=keras.optimizers.Adam(learning_rate=alpha))
    self.target_critic.compile(optimizer=keras.optimizers.Adam(learning_rate=beta))

    self.updagte_target_networks(1)

    self.actor_loss = []
    self.critic_loss = []

    if os.path.exists("tmp"):
      self.load_models()
  
  def updagte_target_networks(self, tau=None):
    if tau is None:
      tau = self.TAU

    weights = []
    targets = self.target_actor.weights
    for i, weight in enumerate(self.actor.weights):
      weights.append(weight*tau + targets[i]*(1-tau))
    self.target_actor.set_weights(weights)

    weights = []
    targets = self.target_critic.weights
    for i, weight in enumerate(self.critic.weights):
      weights.append(weight*tau + targets[i]*(1-tau))
    self.target_critic.set_weights(weights)

  def store_transition(self, state, action, reward, next_state, done):
    self.memory.store_transition([state, action, reward, next_state, done])

  def save_models(self):
    print("Saving models...")
    self.actor.save_weights(self.actor.check_point_file)
    self.critic.save_weights(self.critic.check_point_file)
    self.target_critic.save_weights(self.target_critic.check_point_file)
    self.target_actor.save_weights(self.target_actor.check_point_file)

  def load_models(self):
    print("loading models")
    self.actor.load_weights(self.actor.check_point_file)
    self.critic.load_weights(self.critic.check_point_file)
    self.target_actor.load_weights(self.target_actor.check_point_file)
    self.target_critic.load_weights(self.target_critic.check_point_file)

  def select_action(self, state, evaluate=False):
    state = tf.convert_to_tensor([state], dtype=tf.float32)
    print("output from network")
    action = self.actor(state)
    print(action)
    if not evaluate:
      if random.uniform(0, 1) < self.epsilon.epsilon:
        action = tf.random.uniform((1, self.action_dim), -1, 1)

    return action[0]
  
  def learn(self):
    if self.memory.counter < self.batch_size:
      return
    
    state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

    states = tf.convert_to_tensor(state, dtype=tf.float32)
    actions = tf.convert_to_tensor(action, dtype=tf.float32)
    rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
    next_states = tf.convert_to_tensor(next_state, dtype=tf.float32)

    print("learning--------------------------------*()*()*()()(*()*()(*()(*()(*()(*()*()(*()(*())))))))")
    # critic update
    with tf.GradientTape() as tape:
      t_actions = self.target_actor(next_states)
      target_critic_value = tf.squeeze(self.target_critic(next_states, t_actions), 1)
      critic_q = tf.squeeze(self.critic(states, actions), 1)
      y = rewards + self.gamma * target_critic_value * (1 - done)
      critic_loss = keras.losses.MSE(y, critic_q)
      self.critic_loss.append(critic_loss)

    critic_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
    self.critic.optimizer.apply_gradients(zip(critic_gradient, self.critic.trainable_variables))

    # actor update
    with tf.GradientTape() as tape:
      new_action = self.actor(states)
      actor_loss = -self.critic(states, new_action)
      actor_loss = tf.math.reduce_mean(actor_loss)
      self.actor_loss.append(actor_loss)

    actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
    self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

    self.updagte_target_networks()