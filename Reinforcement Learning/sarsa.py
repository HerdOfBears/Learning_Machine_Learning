

import numpy as np
from reinforcement.gridworld import standard_grid, negative_grid
from reinforcement.the_cliff import standard_cliff, negative_cliff
from reinforcement.iterative_policy_eval import print_policy, print_IPE_result

ALL_POSSIBLE_ACTIONS = ["U","D","L","R"]
GAMMA = 0.9

#g = standard_grid()
#g = negative_grid()
g = standard_cliff()
#g = negative_cliff()

def random_action(a, epsilon=0.1):

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
	for tup in dictionary:
		if s==tup[0]:
			if dictionary[tup] >= max_val:
				max_val = dictionary[tup]
				max_x = tup[1]

	return max_x

def epsilon_greedy_on(Q,s,idx):

	max_A = max_dict(Q,s)
	action = random_action(max_A)
	
	return action


def do_action(a):
	r = g.move(a)
	s_prime = g.current_state()
	return s_prime, r


def main(N=100):
	

	# initial policy instructs the agent to move towards the +1 terminal state.
	"""
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
	"""
	Q = {}
	policy = {}
	a0 = 0.1
	counts = {}
	for s in g.all_states():
		for a in ALL_POSSIBLE_ACTIONS:
			Q[(s,a)] = 0
			counts[(s,a)] = 0
		if s in g.actions:
			policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

	print("Initial policy:")
	print_policy(policy,g)
	biggest_change = 0
	small_enough4convergence = 1e-3
	lst = []
	# N = number of episodes
	for i in range(N):

		# Play game while performing SARSA
		s = (2,0) 
		g.set_state(s)
		a = epsilon_greedy_on(Q,s,i)

		while not g.game_over():
			s_prime, r = do_action(a)
			a_prime = epsilon_greedy_on(Q,s_prime,i)
			Q[(s,a)] = Q[(s,a)] + (0.1)*(r + GAMMA*Q[(s_prime,a_prime)] - Q[(s,a)])
			counts[(s,a)] += 1
			# (a0/(counts[(s,a)]))
			s = s_prime
			a = a_prime


		# takes argmax[a]{ Q(s,a) }
		for s in policy:
			max_a = max_dict(Q,s) 
			policy[s] = max_a
	#print(Q)
	print("After practicing")
	print_policy(policy,g)
	# Next, control problem using TD(0). 
