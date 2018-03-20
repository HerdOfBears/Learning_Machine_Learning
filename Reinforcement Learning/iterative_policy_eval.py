

import numpy as np
from reinforcement.gridworld import standard_grid, negative_grid




def iterative_policy_evaluation(g):
	# g --> gridworld object
	# P --> pi(a|s); the policy of choosing action a given the state s.
	# Probability distribution. But in lots of cases pi(a|s) can equal 1.
	# Default P == None because I want to be able to switch from using a 
	# deterministic to a random policy. Deterministic in the sense that
	# pi(a|s) = 1 for all actions and for all states.

	# We are going to use a dictionary so that we don't need
	# to set up a mapping from a 2-tuple to an integer, which would
	# be needed to use an array.
	V_s = {} 

	# initialize all states' values to zero
	for s in g.all_states():
		V_s[s] = 0.0

	# Now start updating the states' values
	# Iterative policy eval converges only in the limit. So we iterate
	# until the difference in an updated state's value and its previous 
	# value is negligible.
	epsilon = 10e-4
	num = 0
	while True:
		delta = 0
		num += 1
#		if num > 5:
#			break
		for s in g.all_states():
			prev_V = V_s[s]

			gamma = 1.0 # No discount factor this time.

			g.set_state(s)
			current_state = g.current_state()

			# We have to check each action in the set
			# A(s); the possible actions from the state
			# Check if terminal state. Terminal states have no state-value
			if g.is_terminal(current_state):
				V_s[current_state] = 0.0
			else:
				temp_V_s = 0.0
				order_A = len(g.actions[s])
				prob_a = (1.0/order_A)

				for action in g.actions[s]:
					r = g.move(action)
					s_prime = g.current_state()
					temp_V_s += (prob_a * (r + gamma*V_s[s_prime]))
					g.undo_move(action)
					# I want to sum over all of the actions possible
					# from state s1, so after performing state transition,
					# to check the next action from s1, we have to return to
					# s1.
				###	
				V_s[s] = temp_V_s
				delta = max(delta, np.abs(V_s[current_state] - prev_V))
			###
		#print("Delta = ",delta)
		#print("Number of whiles = ", num)
		if delta < epsilon:
			break

	return V_s

def iterative_fixed_policy_evaluation(P,g, V_s):
	# P --> fixed policy
	# g --> grid_world object
	# V_s --> state-value dictionary. 

	epsilon = 10e-4
	num = 0
	while True:
		delta = 0
		num += 1
		for s in g.all_states():
			prev_V = V_s[s]

			gamma = 0.9 # No discount factor this time.

			g.set_state(s)
			current_state = g.current_state()

			# We have to check each action in the set
			# A(s); the possible actions from the state
			# Check if terminal state. Terminal states have no state-value
			if g.is_terminal(current_state):
				V_s[current_state] = 0
			else:
				temp_V_s = 0
				order_A = len(g.actions[s])
				prob_a = (1.0/order_A)

				action = P[s]
				r = g.move(action)
				s_prime = g.current_state()
				temp_V_s = (r + gamma*V_s[s_prime])
				g.set_state(s)
				#g.undo_move(action) # this was error-prone when using randomly generated policies
				###	
				V_s[s] = temp_V_s
			###
			delta = max(delta, abs(V_s[current_state] - prev_V))

		if delta < epsilon:
			break

	return V_s

def main():

	g = standard_grid()
	values_s = iterative_policy_evaluation(g)
	
	#g = negative_grid()
	#values_s = iterative_policy_evaluation(g)
	print(values_s)
	print_IPE_result(g,values_s)
	if(0):
		print("")
		Policy = {
			(0,0):'R',
			(0,1):'R',
			(0,2):'R',
			(1,0):'U',
			(1,2):'R',
			(2,0):'U',
			(2,1):'R',
			(2,2):'R',
			(2,3):'U'
			}
		g = standard_grid()
		values_s = iterative_fixed_policy_evaluation(Policy,g)
		
	#g = negative_grid()
	#values_s = iterative_policy_evaluation(g)
	#print(values_s)
	#print_IPE_result(g,values_s)

def print_IPE_result(g,value_s):
	# g --> grid_world object
	# value_s --> dictionary of states and their values

	width = g.width + 1
	height = g.height + 1
	for i in range(height):
		print("--------------------------------")
		temp_line = ''
		for j in range(width):
			if (i,j) in value_s:
				if value_s[(i,j)]>=0:
					temp_line += " %.3f"%(value_s[(i,j)]) + " "
				else:
					temp_line += "%.3f"%(value_s[(i,j)]) + " "
			else:
				temp_line += " 0.000 "
		print(temp_line)
	print("--------------------------------")

def print_policy(P,g):
	# g --> grid_world object
	# value_s --> dictionary of states and their values

	width = g.width + 1
	height = g.height + 1
	for i in range(height):
		print("--------------------------------")
		temp_line = ''
		for j in range(width):
			if (i,j) in P:
				temp_line += " "+(P[(i,j)]) + " "
			else:
				temp_line += " 0 "
		print(temp_line)
	print("--------------------------------")

def policy_improvement():
	"""
	policy = {
		(0,0):'D',
		(0,1):'R',
		(0,2):'R',
		(1,0):'U',
		(1,2):'R',
		(2,0):'U',
		(2,1):'R',
		(2,2):'R',
		(2,3):'U'
		}
	"""
	POSSIBLE_ACTIONS = ["D","L","R","U"]

	#g = standard_grid()
	g = negative_grid()
	states = g.all_states()

	# INITIALIZE RANDOM POLICY. 
	# Our algorithm should be able to learn an optimal policy.
	policy = {}
	for s in g.actions:
		policy[s] = POSSIBLE_ACTIONS[np.random.randint(4)]

	# INITIALIZE STATE-VALUE DICTIONARY
	V_s = {} 
	# initialize all states' values to zero. Could also have done random; it doesn't matter.
	for s in g.all_states():
		V_s[s] = 0


	# POLICY IMPROVEMENT 
	policy_changed = False
	print("Initial policy:\n")
	print_policy(policy,g)
	print("")
	
	values_s = iterative_fixed_policy_evaluation(policy,g, V_s)
	num = 0
	while True:
		num+=1
		policy_changed = False

		## Check if we can find a better policy. Control problem
		for s in states:
			g.set_state(s)
			current_state = g.current_state()

			gamma = 0.9

			# If s is not a terminal state
			if s in g.actions:
				previous_a = policy[s]
				max_V = values_s[current_state]
				
				for a in POSSIBLE_ACTIONS:
					r = g.move(a)
					s_prime = g.current_state()
					temp_V = (r + gamma*values_s[s_prime])
					if temp_V > max_V:
						max_V = temp_V
						policy[s] = a
						policy_changed = True
					g.set_state(s)

		#		values_s[s] = max_V

		# state-value improvement if policy changed.
		if policy_changed:
			print("iteration = ",num)
			if (0):# num == 200:
				print(policy)
				input("continue?")
			values_s = iterative_fixed_policy_evaluation(policy,g,values_s)			
		else:
			break

	print("New policy:\n")
	print_policy(policy,g)


