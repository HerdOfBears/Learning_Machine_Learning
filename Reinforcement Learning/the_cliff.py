

import numpy as np
import matplotlib.pyplot as plt


class the_cliff():
	
	def __init__(self, width, height, start):
		# start --> tuple of integers defining agent's start position
		self.width = width
		self.height = height

		# the coodinates of the start position
		self.i = start[0] 
		self.j = start[1]

	def set(self, rewards, actions):
		# both of these are dictionaries.
		# Key --> (i, j)
		
		# value --> reward from being at that position
		self.rewards = rewards 

		# value --> list of possible actions from position (i, j)
		# i.e. A(s), the set of actions from state s.
		self.actions = actions

	def set_state(self, s):
		# sets the agent's new position based on state s
		self.i = s[0]
		self.j = s[1]

	def current_state(self):
		return (self.i, self.j)

	def is_terminal(self, s):
		return s not in self.actions

	def move(self, action):
		# Before moving, check if action is legal
		# Action is one of "U", "D", "L", "R"
		if action in self.actions[(self.i, self.j)]:
			# follows numpy array convention
			if action=="U":
				self.i -= 1
			if action == "D":
				self.i += 1
			if action == "L":
				self.j -= 1
			if action == "R":
				self.j += 1
		
		# Gets the value of the key (self.i, self.j) IF key is in the dictionary, 
		# else return default (in this case 0)
		# I.e. if their is a reward associated with the state (i,j), that reward
		# will be returned; otherwise, a reward of zero is returned.
		return self.rewards.get((self.i, self.j), 0)

	def undo_move(self, action):
		# Takes previous action as argument.
		# If the previous action was "U" and we want to undo that,
		# the input here is also "U"
		if action=="U":
			self.i += 1
		if action == "D":
			self.i -= 1
		if action == "L":
			self.j += 1
		if action == "R":
			self.j -= 1

		# Check if the state we returned to is a possible state. 
		# If it isn't, this will raise an AssertionError
		assert(self.current_state() in self.all_states())
	
	def game_over(self):

		# If there are no actions from the current state, then 
		# the current state is a terminal state and the game is over.
		if (self.i, self.j) not in self.actions:
			return True
		else:
			return False

	def all_states(self):

		# All possible states are those that have actions that move from them
		# and those that have rewards (terminal states)
		# Because there may be some overlap b/w the two,
		# casting it as a set will remove the duplicates
		# Inspired by the Udemy reinforcement learning course
		return set(list(self.actions.keys()) + list(self.rewards.keys()))


def standard_cliff():
	# x --> wall
	# s --> start
	# 0  1  2  3  4  5  6
	# .  .  .  .  .  .  . 0
	# .  .  .  .  .  .  . 1
	# .  .  . -1  .  .  . 2
	# s -1 -1 -1 -1 -1 +1 3
	# Really need a better way of initializing this. 
	# Maybe make one s.t. have to specify walls, terminal states, and start only
	g = the_cliff(6,3,(3,0)) # 0-indexing
	g.set(
		{ 
		(3,1):-1,
		(3,2):-1,
		(3,3):-1,
		(3,4):-1,
		(3,5):-1,
		(3,6):+1,
		(2,3):-1
		},
		{
		(0,0):["D","R"],
		(0,1):["L","R","D"],
		(0,2):["L","R","D"],
		(0,3):["L","R","D"],
		(0,4):["L","R","D"],
		(0,5):["L","R","D"],
		(0,6):["L","D"], ## end of row 0
		(1,0):["R","D","U"],
		(1,1):["L","R","D","U"],
		(1,2):["L","R","D","U"],
		(1,3):["L","R","D","U"],
		(1,4):["L","R","D","U"],
		(1,5):["L","R","D","U"],
		(1,6):["L","D","U"], ## end of row 1
		(2,0):["R","D","U"],
		(2,1):["L","R","D","U"],
		(2,2):["L","R","D","U"],
		#(2,3):["L","R","D","U"],
		(2,4):["L","R","D","U"],
		(2,5):["L","R","D","U"],
		(2,6):["L","D","U"], ## end of row 2
		(3,0):["U"],
		}
	)
	return g

def negative_cliff(step_cost = -0.1):

	# x --> wall
	# s --> start
	#    0  1  2  3  4  5  6
	
	# 0  .  .  .  .  .  .  . 
	# 1  .  .  .  .  .  .  . 
	# 2  .  .  .  x  .  .  . 
	# 3  s -1 -1 -1 -1 -1 +1 


	# This penalizes the agent for every step it takes.
	# Fewer steps --> higher final reward.
	# Lots of steps --> lower final reward

	g = standard_cliff()
	g.rewards.update(
		{
		(0,0):step_cost,
		(0,1):step_cost,
		(0,2):step_cost,
		(0,3):step_cost,
		(0,4):step_cost,
		(0,5):step_cost,
		(0,6):step_cost, ## end of row 0
		(1,0):step_cost,
		(1,1):step_cost,
		(1,2):step_cost,
		(1,3):step_cost,
		(1,4):step_cost,
		(1,5):step_cost,
		(1,6):step_cost, ## end of row 1
		(2,0):step_cost,
		(2,1):step_cost,
		(2,2):step_cost,
		(2,3):step_cost,
		(2,4):step_cost,
		(2,5):step_cost,
		(2,6):step_cost, ## end of row 2
		(3,0):step_cost,
		}
	)
	
	return g


