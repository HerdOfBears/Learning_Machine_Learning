"""

Solves the cartpole problem using a random search over weights of a
linear model

"""

import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt


def play_episode(w):

	done = False
	s = env.reset()

	num = 0
	while not done:
		num+=1
		observation, r, done, _ = env.step()

	return num

def get_act(observation,w):
	y = np.dot(observation,w)[0]
	if y>=0:
		action = 1
	else:
		action = 0
	return action
# roughly 22 random steps before episode is done
def average_rand_acts(N=100):

	tot=0
	for i in range(N):

		tot+=play_episode()
	print(tot/N)


def main():

#	env = gym.make('CartPole-v1')

	w = np.ones((4,1)) - 10
	max_avg = -1
	for i in range(10):
		new_w = np.random.rand(4,1) # 4x1 dimensions
		
		avg = 0
		num = 0
		for j in range(10):

				done = False
				env = gym.make('CartPole-v0')
				s = env.reset()
				a = get_act(s,new_w)

				while not done:
					if j==1:
						env.render()
						pass
					num+=1
					observation, r, done, _ = env.step(a)
					a = get_act(observation, new_w)
				if j==1:
					env.close()
					pass
		avg = (1/10)*(num)
		if avg > max_avg:
			w = new_w
			max_avg = avg

	#env = wrappers.Monitor(env, "C:/Users/Jyler/Documents/ProgrammingProjects/reinforcement/")
	done = False
	env = gym.make('CartPole-v0')

	s = env.reset()
	a = get_act(s,w)
	num = 0
	tot_r = 0
	while not done:
		env.render()
		num += 1
		observation, r, done, _ = env.step(a)
		a = get_act(observation, w)
		tot_r += r
	env.close()
	print("max_avg = ",max_avg, " num= ",num, "tot reward = ",tot_r)
	print("best weights:\n",w)