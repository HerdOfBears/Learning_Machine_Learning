
import numpy as np
from reinforcement.gridworld import standard_grid, negative_grid
from reinforcement.iterative_policy_eval import print_policy, print_IPE_result

ALL_POSSIBLE_ACTIONS = ["U","D","R","L"]
GAMMA = 0.9


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


def play_episode(policy):
	# start --> tuple: (s, a), s is a tuple (i,j). So, ( (i,j), a )
	global g

	random_start = list(g.actions.keys())[np.random.choice(len(g.actions.keys()))]
	random_action = np.random.choice(ALL_POSSIBLE_ACTIONS)
	start = (random_start, random_action)
	
	g.set_state(start[0])
	s0 = g.current_state()
	a0 = start[1]
	states_actions_and_rewards = [(s0,a0,0)]
	prev_s = s0
	prev2_s = None 

	s = g.current_state()
	action = a0
	seen_states = [s0]
	# NOTE:
	# r(t) refers to the reward acquired from performing a(t-1) in state s(t-1)
	while True:
		prev_s = s		
		r = g.move(action)
		s = g.current_state()
#		print("prev_s = ", prev_s, " s = ", s," action = ", action)

		if s == prev_s:
			states_actions_and_rewards.append((s,None,-100))
			break
		elif s in seen_states:
#			print("Cycle")
			r = -100
			states_actions_and_rewards.append((s,None,r))
			break
		elif g.game_over():
			states_actions_and_rewards.append((s,None,r))
			break
		else:
			action = policy[s]
			states_actions_and_rewards.append((s,action,r))
		seen_states.append(s)

#		prev_s = s
	print("Check")

	# Game is over. Now compute the actual return.
	G = 0
	states_actions_and_returns = []
	# Start at the last state and reward because the return is calculated 
	# from the final to the initial state 
	# (it depends on the reward of the final state)
	first = True
	for s,a,r in reverse(states_actions_and_rewards):
		
		if first:
			first = False
		else:
			states_actions_and_returns.append((s,a,G))
	
		G = r + GAMMA*G
	states_actions_and_returns.reverse()

	return states_actions_and_returns


def monte_carlo_ES(P,N,Q):
	# P --> policy
	# N --> Number of iterations
	# Q --> state-value function
	global g
	states = g.all_states()

	returns = {}
	for s in g.all_states():
		if s not in g.actions:
			continue
		for a in ALL_POSSIBLE_ACTIONS:
			returns[(s,a)] = []


	deltas = []
	biggest_change = 0

	for i in range(N):
		
		# states_actions_returns is a list of 3-tuples [(s,a,G)]; 
		# the state, the action taken, and the return
		seen_before = []
		states_actions_returns = play_episode(P)
		
		for s,a,G in states_actions_returns:
			if (s,a) not in seen_before:
				seen_before.append((s,a))
				old_Q = Q[(s,a)]
				returns[(s,a)].append(G)
				Q[(s,a)] = sample_mean(returns[(s,a)])
				biggest_change = max(biggest_change, np.abs(old_Q - Q[(s,a)]))
				deltas.append(biggest_change)


		#max_a = None
		for s in list(P.keys()):
			max_Q = float("-inf")
			prev_P = P[s]
			max_a = P[s]

			# Get the argmax[a]{ Q(s,a) }
			for tup in list(Q.keys()):
				if s == tup[0]:
					a = tup[1]
					if Q[(s,a)]>max_Q:
						max_Q = Q[(s,a)]
						max_a = a
			P[s] = max_a

	return P,Q


def main():

	global g
	g = negative_grid()
	#g = standard_grid()

	global gamma
	gamma = 0.9

	states = g.all_states()

	# initialize the action-value function
	Q = {}
	policy={}
	for s in states:
		if s in g.actions:
			for a in ALL_POSSIBLE_ACTIONS:
				Q[(s,a)] = 0#np.random.rand() 
			policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
		else:
			pass
			#for a in ALL_POSSIBLE_ACTIONS:
			#	Q[(s,a)] = 0

	optimal_policy,Q = monte_carlo_ES(policy, 4000, Q)
	print_policy(optimal_policy,g)
	#print(deltas)
	for s in g.actions:
		print('s=',s,' ',Q[s,optimal_policy[s]])
	print(Q[(2,3),"L"])





"""
Grid:

. . . +1
. x . -1
s . .  .



"""