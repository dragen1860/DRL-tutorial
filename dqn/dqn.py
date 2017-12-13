import tensorflow as tf
import gym
import random


class DQN():

	def __init__(self):
		pass


def main():
	env = gym.make('Breakout-v0')
	cfg = tf.ConfigProto()
	cfg.gpu_options.allow_growth = True

	with tf.Session(config = cfg) as sess:
		pass




if __name__ == '__main__':
	main()