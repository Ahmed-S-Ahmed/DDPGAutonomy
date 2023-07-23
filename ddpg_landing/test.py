#!/usr/bin/env python3
from utils import plot_learning_curve
import pickle

counter = pickle.load(open("counter", "rb"))

score_history = pickle.load(open(f"metrics_{counter}/score_history", "rb"))
actor_loss = pickle.load(open(f"metrics_{counter}/actor_loss", "rb"))
critic_loss = pickle.load(open(f"metrics_{counter}/critic_loss", "rb"))

x = [i+1 for i in range(len(score_history))]
xa = [i+1 for i in range(len(actor_loss))]
xc = [i+1 for i in range(len(critic_loss))]

plot_learning_curve(x, score_history, f"metrics_{counter}/learning_curve", "average reward over 100 timesteps")
plot_learning_curve(xa, actor_loss, f"metrics_{counter}/actor_loss_graph", "actor loss per1 timestep")
plot_learning_curve(xc, critic_loss, f"metrics_{counter}/critic_loss_graph", "critic loss per timestep")