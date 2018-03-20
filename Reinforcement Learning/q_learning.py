


import numpy as np
from reinforcement.gridworld import standard_grid, negative_grid
from reinforcement.the_cliff import standard_cliff, negative_cliff
from reinforcement.iterative_policy_eval import print_policy, print_IPE_result
from reinforcement.sarsa import random_action, max_dict

ALL_POSSIBLE_ACTIONS = ["U","D","L","R"]
GAMMA = 0.9

#g = standard_grid()
#g = negative_grid() 
g = standard_cliff()
#g = negative_cliff()

def epsilon_greedy_on(Q,s,idx):

	max_A = max_dict(Q,s)
	action = random_action(max_A,epsilon=0.1)
	
	return action


def do_action(a):
	r = g.move(a)
	s_prime = g.current_state()
	return s_prime, r

def main(N=100):

	states = g.all_states()
	Q = {}
	policy = {}
	counts = {}
	for s in states:
		# Initialize the action-value function
		for a in ALL_POSSIBLE_ACTIONS:
			Q[(s,a)] = 0
			counts[(s,a)] = 0
		# initialize a random policy
		if s in g.actions:	
			policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)


	for i in range(N):
		#print("episode = ",i+1)

		# start position
		s = (2,0)
		g.set_state(s)
		a = epsilon_greedy_on(Q,s,i)
		
		while not g.game_over():
			s_prime, r = do_action(a)
			a_prime = epsilon_greedy_on(Q,s_prime,i)
			#print("a_prime = ",a_prime," s_prime = ",s_prime)
			max_A = max_dict(Q,s_prime)
			max_Q = Q[(s_prime, max_A)]
			Q[(s,a)] = Q[(s,a)] + (0.1)*(r + GAMMA*max_Q - Q[(s,a)])

			s = s_prime
			a = a_prime

	# takes argmax[a]{ Q(s,a) }
	for s in policy:
		max_a = max_dict(Q,s) 
		policy[s] = max_a	
	
	print(Q)
	print_policy(policy,g)

