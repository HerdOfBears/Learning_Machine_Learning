

import numpy as np
from reinforcement.gridworld import standard_grid, negative_grid
from reinforcement.iterative_policy_eval import print_policy, print_IPE_result

ALL_POSSIBLE_ACTIONS = ["U","D","L","R"]
GAMMA = 0.9

g = standard_grid()
#g = negative_grid()

def random_action(a, epsilon=0.1):

	p = np.random.rand()
	if p < epsilon:
		action = np.random.choice(ALL_POSSIBLE_ACTIONS)
	else:
		action = a
	return action

def play_game(policy,start):
	# Play game. Perform temporal-difference method of finding
	# value function given a policy (prediction problem)
	states_rewards = []
	g.set_state(start)
	s = g.current_state()

	states_rewards.append((s,0))
	while not g.game_over():

		a = policy[s]
		a = random_action(a)
		r = g.move(a)
		s_prime = g.current_state()

		states_rewards.append((s_prime,r))
		s = s_prime

	return states_rewards


def main(N=100, alpha=0.1):
	

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

	V = {}
	V2 = {}
	for s in g.all_states():
		V[s] = 0
		V2[s] = 0

	biggest_change = 0
	small_enough4convergence = 1e-3

	# N = number of episodes
	for i in range(N):
#	n = 0
#	while True:
#		n +=1 
#		print(n)
		# initialize starting point
		s = (2,0) 
		states_and_rewards = play_game(policy, s)

		for idx in range(len(states_and_rewards)-1):
			s, _ = states_and_rewards[idx]
			s_prime, r = states_and_rewards[idx+1]
			V[s] = V[s] + alpha*(r + GAMMA*V[s_prime] - V[s])

		# Play game. Perform temporal-difference method of finding
		# value function given a policy (prediction problem)
		
		# Want to see if it matters whether you update after every state
		# change compared to at the end of episode. TURNS OUT IT DOESN'T
		s = (2,0) 
		g.set_state(s)

		while not g.game_over():
			a = policy[s]
			a = random_action(a)
			r = g.move(a)
			s_prime = g.current_state()

			V2[s] = V2[s] + alpha*(r + GAMMA*V2[s_prime] - V2[s])

			s = s_prime
		


	print_IPE_result(g,V)
	print("Now V2 = ")
	print_IPE_result(g,V2)
	# Next, control problem using TD(0). 
