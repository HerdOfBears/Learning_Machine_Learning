
import pandas as pd
import gym
import numpy as np
import matplotlib.pyplot as plt

GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = [0,1]

def random_action(a, epsilon):

	p = np.random.rand()
	if p < epsilon:
		action = np.random.choice(ALL_POSSIBLE_ACTIONS)
	else:
		action = a
	return action


def max_dict(dictionary, s):
	# dictionary --> dictionary of tuples (s,x)
	# s --> element
	# This function will find argmax[x]{ dictionary(s,x) }
	max_x = None
	max_val = float("-inf")
	"""
	for tup in dictionary:
		if s==tup[0]:
			if dictionary[tup] >= max_val:
				max_val = dictionary[tup]
				max_x = tup[1]
	"""
	if dictionary[(s,0)] >dictionary[(s,1)]:
		max_x = 0
	else:
		max_x = 1 
	return max_x


def epsilon_greedy_on(Q,s,epsilon):

	max_A = max_dict(Q,s)
	action = random_action(max_A,epsilon)
	
	return action


def cut_into_interals(observation, minim, maxim, number_of_intervals,obs_idx):
	# minim --> minimum box boundary, but we check to infinity on either side.
	# maxim --> maximum box boundary, ""
	# This will cut the continuous valued observation into intervals, then
	# return which interval the observation is in.
	# Ex: -1.5
	# (-inf -2.4], (-2.4,-1.2],(-1.2,0] (0, 1.2] (1.2,2.4] [2.4,+inf)
	# 		1			2			3		4		5			6
	# -1.5 is in the 2nd interval
	# return 2
	delta = (maxim - (-1)*(maxim))/number_of_intervals
	v_delta = np.arange(minim+delta,maxim,delta)
	v_delta_shape = v_delta.shape[0]
	v_delta = v_delta.reshape(1,v_delta_shape)
	#print(obs_idx, " : ",v_delta)
	for i in range(v_delta_shape-1):
		if observation[0][obs_idx] <= v_delta[0][0]:
			s1 = 1
			break
		if (v_delta[0][i+1]>=observation[0][obs_idx]) and (observation[0][obs_idx]>v_delta[0][i]):
			s1 = i+1+1
			break
	if observation[0][obs_idx]>=v_delta[0][-1]:
		s1 = number_of_intervals

	return s1


def get_state(observation):
	# observation --> vector of observations
	# This function uses the observations and the binning to get the state 
	# in a 4D box.
	# Things we know: the episode ends if the angle of the pole is >= 15 degrees
	# 				the episode ends if the position of the cart is >= 2.4
	# observations = [position, velocity, angle, rotation rate]

	# Position interval: (-2.4, 2.4)
	pos1 = cut_into_interals(observation,-2.4,2.4,10,0)

	# velocity interval
	vel1 = cut_into_interals(observation,-2,2,10,1)

	# angle interval
	ang1 = cut_into_interals(observation,-0.4,0.4,10,2)

	# rotation rate interval
	rot1 = cut_into_interals(observation,-3.5,3.5,10,3)

	state = int( str(pos1)+str(vel1)+str(ang1)+str(rot1) )
	return state


def play_episode(env,Q,episode_idx):
			done = False
			#env = gym.make('CartPole-v1')

			# Chance of exploration
			#epsilon = 0.1/((episode_idx/5000)+1)
			# when /10, and when /1, ~1000 episodes is when improvement started to wane
			# multiplying episode_idx by 4.5 resulted in a nice learning curve
			epsilon = 1.0/((np.sqrt((episode_idx) + 1))**(4/3))
			# Start position
			observation = env.reset()
			observation = observation.reshape(1,4)			
			s = get_state(observation)

			# Starting action
			a = epsilon_greedy_on(Q,s,epsilon)
			
			num = 0
			tot_r = 1
			alpha = 0.1
			while not done:
				num+=1
				
				observation, r, done, _ = env.step(a)
				observation = observation.reshape(1,4)
				s_prime = get_state(observation)

				a_prime = epsilon_greedy_on(Q,s,epsilon)				

				if done and num < 199:
					r = -400

				#max_A = max_dict(Q,s_prime)
				#max_Q = Q[(s_prime, max_A)]
				if Q[(s_prime,0)] >= Q[(s_prime,1)]:
					max_Q = Q[(s_prime,0)]
				else:
					max_Q = Q[(s_prime,1)]
				Q[(s,a)] = Q[(s,a)] + (alpha)*(r + GAMMA*max_Q - Q[(s,a)])

				a = a_prime
				s = s_prime
				tot_r += r

			return Q, num+1
			# reaching 200 reward is solving the game
			# Rather than using each time step as a +1 reward, we redefine
			# the reward as -[large number] upon failing
#			if tot_r < 200:
#				r = -500


def main(N=100):

	# Initialize the action-value function Q, and an arbitrary policy
	# There are 10**4=10,000 possible states
	# There are 2 possible actions
	# So a total of 20,000 possible (s,a) pairs
	Q = {}
	policy = {}
	for i in range(1,11,1):
		for j in range(1,11,1):
			for k in range(1,11,1):
				for l in range(1,11,1):
					st = int( str(i)+str(j)+str(k)+str(l) )
					Q[(st,0)] = np.random.uniform(-1,1)
					Q[(st,1)] = np.random.uniform(-1,1)
					policy[st] = 0
	
	# Practice/Train 
	env = gym.make("CartPole-v0")
	tota = 0
	y_vals = []
	x_vals = []
	for i_episode in range(N):
		if (i_episode%100 == 0) and (i_episode!=0):
			x_vals.append(i_episode)
			y_vals.append(tota/100)
			print("episode = ",i_episode)
			print(tota/100)
			tota = 0

		Q, totR = play_episode(env, Q,i_episode)
		tota += totR

	plt.plot(x_vals,y_vals)
	plt.show()	
	
	ans = input("continue to the test?")
	if ans.lower() in ["y","yes"]:
		pass
	else:
		raise ValueError
	
	# Make optimal policy
	for i in range(1,11,1):
		for j in range(1,11,1):
			for k in range(1,11,1):
				for l in range(1,11,1):
					st = int( str(i)+str(j)+str(k)+str(l) )
					if Q[(st,0)] >= Q[(st,1)]:
						policy[st]=0
					else:
						policy[st]=1

	# Test
	tot_r=0
	for i in range(100):
		env = gym.make("CartPole-v0")

		done = False

		# Start position
		observation = env.reset()
		observation = observation.reshape(1,4)			
		s = get_state(observation)

		# Starting action
		a = policy[s]
				
		num = 0
		while not done:
			if i ==1:
				env.render()
			num+=1				
					
			observation, r, done, _ = env.step(a)
			observation = observation.reshape(1,4)

			s_prime = get_state(observation)
			a_prime = policy[s_prime]

			a = a_prime
			s = s_prime
			tot_r += r
	#tot_r = (1/10)*tot_r
		if i == 1:
			env.close()
	print("tot reward = ",tot_r/100)
	if (0):#tot_r/100 > 195:
		df = pd.DataFrame()
		df = df.from_dict(policy,orient="index").reset_index()
		df.to_csv("C:/Users/Jyler/Documents/ProgrammingProjects/reinforcement/cart_pole_bins_solved.csv",index=False)
		print("Saved")
	#return tot_r

def is_learning():

	df = pd.read_csv("C:/Users/Jyler/Documents/ProgrammingProjects/reinforcement/cart_pole_bins_solved.csv")
	policy = df.set_index("index").T.to_dict("list")
	#return policy
	#print(df.head())

	env = gym.make("CartPole-v1")

	done = False

	# Start position
	observation = env.reset()
	observation = observation.reshape(1,4)			
	s = get_state(observation)

	# Starting action
	a = policy[s][0]
	tot_r = 0	
	num = 0
	while not done:
		env.render()
		num+=1				
					
		observation, r, done, _ = env.step(a)
		observation = observation.reshape(1,4)

		s_prime = get_state(observation)
		a_prime = policy[s_prime][0]

		a = a_prime
		s = s_prime
		tot_r += r

	env.close()
	print(tot_r)
