"""
author: Jyler
date: March 2018
Monte Carlo method of discovering value function (i.e. prediction problem).
Next chapter (chpt. 5) of Reinforcement Learning: An Introduction by Sutton and Barto (1998)


Chapter 4 (value iteration, iterative policy evaluation, etc) assume full knowledge of the environment
and actions' transition probabilities. This is a heavy requirement.
Monte Carlo evaluation is model-free. It does not require full knowledge of the environment. However,
we do assume the environment functions episodically. Hence, Monte Carlo methods are incremental in an episode-by-episode by sense.


"""



import numpy as np
from reinforcement.gridworld import standard_grid, negative_grid
from reinforcement.iterative_policy_eval import print_policy, print_IPE_result


def reverse(lst):
	# input: list
	# output: the reverse of the list
	temp_list = []
	L = len(lst)
	for i in range((L-1),-1,-1):
		temp_list.append(lst[i])
	return temp_list


def sample_mean(lst):
	n = len(lst)
	tot = sum(lst)
	x_bar = (1.0/n)*tot
	return x_bar

def epsilon_greedy(s,V,g):
	# s --> current state
	# V --> state-value function
	# g --> grid_world

	epsilon = 0.1
	max_V = None
	pass


def play_episode(policy):
	
	global g

	# Start at random non-terminal, non-wall position
	possible_starts = g.actions.keys()
	start_idx = np.random.randint(len(possible_starts))
	g.set_state(list(possible_starts)[start_idx])

	s = g.current_state()
	states_and_rewards = [(s,0)]
	while not g.game_over():
		action = policy[s]
		r = g.move(action)		
		s = g.current_state()
		states_and_rewards.append((s,r))

	# Game is over. Now compute the actual return.
	G = 0
	states_and_returns = []
	# Start at the last state and reward because the return is calculated 
	# from the final to the initial state 
	# (it depends on the reward of the final state)
	for s,r in reverse(states_and_rewards):
		states_and_returns.append((s,G))
		G = r + gamma*G
	states_and_returns.reverse()
	return states_and_returns
	pass


def first_visit_monte_carlo_pred(P,N,V):
	# P --> policy
	# N --> Number of iterations
	# V --> state-value function
	global g

	all_returns = {}
	for s in g.all_states():
		all_returns[s] = []

	for i in range(N):
		seen_before = []
		# states_AND_returns is a list of tuples [(s,G)]
		states_AND_returns = play_episode(P)
		for s,ret in states_AND_returns:
			if s in seen_before:
				continue
			seen_before.append(s)
			# This could be optimized. Rather than re-calculating the sample mean 
			# every iteration, we could update the (n-1)-th sample mean with the new
			# data point
			all_returns[s].append(ret)
			V[s] = sample_mean(all_returns[s])
	return V


def main():

	global g
	#g = negative_grid()
	g = standard_grid()

	global gamma
	gamma = 0.9

	# initial policy instructs the agent to move towards the +1 terminal state.
	policy = {
	(0,0):"R",
	(0,1):"R",
	(0,2):"R",
	(1,0):"U",
	(1,2):"R",
	(2,0):"U",
	(2,1):"R",
	(2,2):"R",
	(2,3):"U"
	}

	states = g.all_states()

	# initialize the state-value function
	V = {}
	for s in states:
		if s in g.actions:
			V[s] = np.random.rand() 
		else:
			V[s] = 0

	V = first_visit_monte_carlo_pred(policy, 100, V)
	print_IPE_result(g,V)




