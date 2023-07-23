#!/usr/bin/env python3
import numpy as np

class Memory:
  def __init__(self,  state_shape, action_dim, buffer_size): 
    self.buffer_size = buffer_size
    self.counter = 0

    self.state_memory = np.zeros((self.buffer_size, *state_shape))
    self.next_state_memory = np.zeros((self.buffer_size, *state_shape))
    self.action_memory = np.zeros((self.buffer_size, action_dim))
    self.reward_memory = np.zeros(self.buffer_size)
    self.done_memory = np.zeros(self.buffer_size, dtype=np.bool)

  def store_transition(self, time_step):
    indx = self.counter % self.buffer_size
    state, action, reward, next_state, done = time_step

    self.state_memory[indx] = state
    self.action_memory[indx] = action
    self.reward_memory[indx] = reward
    self.next_state_memory[indx] = next_state
    self.done_memory[indx] = done

    self.counter += 1

  def sample_buffer(self, batch_size):
    max_samples = min(self.counter, self.buffer_size)
    indxs = np.random.choice(max_samples, batch_size, replace=False)

    states = self.state_memory[indxs]
    actions = self.action_memory[indxs]
    rewards = self.reward_memory[indxs]
    next_states = self.next_state_memory[indxs]
    done = self.done_memory[indxs]
    
    return (states, actions, rewards, next_states, done)