"""
author: Jyler Menard
date: March 2018
One algorithm written while learning reinforcement learning
Used Sutton and Barto's Reinforcement Learning: An Introduction (1998) as reference

"""


import numpy as np
from reinforcement.gridworld import negative_grid
from reinforcement.iterative_policy_eval import print_policy

def value_iteration():
	"""
	This algorithm finds the optimal policy.
	This improves upon cycling between policy improvement and iterative
	policy evaluation because the I-E-I-E-.. requires evaluating nested
	loops. This algorithm requires only one (main) loop. 
	"""
	# g --> grid_world object	
	g = negative_grid()

	possible_actions = ["U","D","L","R"]
	states = g.all_states()

	V = {}
	for s in states:
		V[s] = 0
	
	policy = {}
	epsilon = 1e-4
	gamma = 0.9
	num = 0
	while True:
		delta = 0
		num += 1

		for s in states:

			prev_V = V[s]
			max_V = float("-inf")
			temp_a = None
			g.set_state(s)

			if s in g.actions:

				for action in possible_actions:
					r = g.move(action)
					s_prime = g.current_state()
					temp_V = r + gamma*V[s_prime]
					if temp_V>max_V:
						max_V = temp_V
# THIS MIGHT NOT GIVE THE BEST POLICY. I MIGHT NEED TO MAKE ANOTHER LOOP THAT FINDS THE BEST POLICY AFTER FINDING THE OPTIMAL VALUE FUNCTION
						temp_a = action 
					g.set_state(s)
				###
				V[s] = max_V
				policy[s] = temp_a
				delta = max(delta, np.abs(max_V - prev_V))

		if delta < epsilon:
			break
		###
	###
	print_policy(policy,g)
	print("Iterations = ",num)
	print(V)
	return policy

