import gym
import random

class Env:

	def __init__(self):
		self.env = gym.make('Breakout-v0')

		self.