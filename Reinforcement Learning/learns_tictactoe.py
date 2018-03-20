

"""

Second part of LazyProgrammer's Reinforcement Learning in Python course on Udemy.
Algorithm to learn how to play tic-tac-toe.

Objects needed:
Player object
	- take action function
		- INPUT = Board
		- Places a piece on the board. I.e. changes the environment.
	- update_state_history function
		- INPUT = state of board
		- Keeps track of states/moves made.
			- I.e. records what its action (and the opponent's action) was.
	- update function
		- INPUT = board/environment
		- Performs value function update

Environment object
	- game_over function
		- OUTPUT = boolean for whether the board is full or not.
		- True if over; false otherwise
	- draw_board function
		- Prints the board state
	- get_state function
		- Output the state of the board


"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




class env():

	def __init__(self):
# why the "+ 4"? --> my (simple) idea for uniquely encoding board states is if the board is [ ['x','o',''],['o','','x'],['x','o',''] ], 
# then it will be recoded as [1,2,4,2,4,1,1,2,4] and specifically as the 9 digit integer 124241124. From there, each state can be easily stored
# into a numpy array, and found by using binary search. Also '4' because '0' wouldn't allow easy 9 digit integers (i.e. 000,000,002 would be 2, but so would )
		self.board = np.zeros([3,3]) 

	def update_board(self, i, j, play):
		if (play == 'x') or (play == 1):
			temp_play = 1
		if (play == 'o') or (play == 2):
			temp_play = 2

		self.board[i,j] = temp_play

	def game_over(self):
		# Returns boolean. 1 --> game_over; 0 --> game not over

		board = self.board
		board = board.flatten()
		# Check diagonals for a winner
		if (board[0] == 1) and (board[4] == 1) and (board[8] == 1):
			return 1
		if (board[2] == 1) and (board[4] == 1) and (board[6] == 1):
			return 1
		if (board[0] == 2) and (board[4] == 2) and (board[8] == 2):
			return 1
		if (board[2] == 2) and (board[4] == 2) and (board[6] == 2):
			return 1

		# Check horizontals
		if ( ((board[0] == 1) and (board[1] == 1) and (board[2] == 1)) or
			 ((board[0] == 2) and (board[1] == 2) and (board[2] == 2)) 
			):
			return 1
		if ( ((board[3] == 1) and (board[4] == 1) and (board[5] == 1)) or
			 ((board[3] == 2) and (board[4] == 2) and (board[5] == 2)) 
			):
			return 1
		if ( ((board[6] == 1) and (board[7] == 1) and (board[8] == 1)) or
			 ((board[6] == 2) and (board[7] == 2) and (board[8] == 2)) 
			):
			return 1

		# Check verticals
		if ( ((board[0] == 1) and (board[3] == 1) and (board[6] == 1)) or
			 ((board[0] == 2) and (board[3] == 2) and (board[6] == 2)) 
			):
			return 1
		if ( ((board[1] == 1) and (board[4] == 1) and (board[7] == 1)) or
			 ((board[1] == 2) and (board[4] == 2) and (board[7] == 2)) 
			):
			return 1
		if ( ((board[2] == 1) and (board[5] == 1) and (board[8] == 1)) or
			 ((board[2] == 2) and (board[5] == 2) and (board[8] == 2)) 
			):
			return 1

		# Check for an empty space
		for i in range(9):
			if board[i] == 0:
				return 0

		# If there are no winners and there are no empty spaces, then the game is over and is a tie.
		return 1

	def get_state(self):
		# 'x' --> 1
		# 'o' --> 2
		# empty --> 0
		# recoding the state into an integer works like this:
	# [['x','x','o'],['o','x','o'],['','','x']] --> [[1,1,2],[2,1,2],[0,0,1]] as 112212001
		board = self.board
		temp_board = board.flatten()
		temp_str = ""
		for num in temp_board:
			temp_str += str( int(num) )
		temp = int(temp_str)
		#return board	
		#return temp # max 9 digit integer
		return temp_str

	def draw_board(self):
		board = self.board
		
		row1 = []
		row2 = []
		row3 = []

		for i in range(3):
			for j in range(3):
				if i == 0:
					if board[i,j] == 1:
						row1.append('x')
					if board[i,j] == 2:
						row1.append('o')
					if board[i,j] == 0:
						row1.append('.')
				if i == 1:
					if board[i,j] == 1:
						row2.append('x')
					if board[i,j] == 2:
						row2.append('o')
					if board[i,j] == 0:
						row2.append('.')
				if i == 2:
					if board[i,j] == 1:
						row3.append('x')
					if board[i,j] == 2:
						row3.append('o')
					if board[i,j] == 0:
						row3.append('.')

		print("-------------------")
		print("| ",row1[0]," | ",row1[1], " | ", row1[2]," |")
		print("| ",row2[0]," | ",row2[1], " | ", row2[2]," |")
		print("| ",row3[0]," | ",row3[1], " | ", row3[2]," |")
		print("-------------------")


class player():

	def __init__(self, symbol, learningRate,use_past_values = False):
		# initialize an array for the state history
		# Okay, any game of tic-tac-toe has 9 spaces, and each space has at
		# most 3 possible states, 'x','o',''
		# So there are at most 
		# 3x3x3x3x3x3x3x3x3 = 3^(9) = 19,683
		# but not all of those are possible. So the actual amount of
		# states is much less. 
		# Let's just use a length 20,000 vector, and use the indices to 
		# lookup each state.

		# symbol is either 'x' or 'o'
		# 'x' -- > 1
		# 'o' -- > 2
		if symbol.lower() == 'x':
			self.piece = 1
		if symbol.lower() == 'o':
			self.piece = 2

		self.learning_rate = learningRate
		self.number_wins = 0

		self.state_history = []

		# initalize value function lookup table with random values on all possible states.
		if use_past_values:
			df = pd.read_csv("C:/Users/Jyler/Documents/ProgrammingProjects/reinforcement/player1_learned_values.csv",header=None)
			self.values = df.values
		else:
			self.values = np.zeros([20000,1]) + 0.5 
		
	def update_state_history(self, state):
		# state is an integer of maximum 9 digits
		pass

	def trinary(self, state):
		# state --> int 
		# map 9 digit state to a real number using a similar function
		# to binary
		temp_state = str(state)
		lst = []
		tot = 0

		N = len(temp_state)
		for i in range(N):
			tot += ( (3**i)*( int(temp_state[N-1-i]) ) )

		return tot

	def update(self, state): 
		# state --> max 9 digit integer.
#		state = int(state) #####################

		temp_state = str(state)
		reward = 0
		win_state = str(self.piece)*3
		if temp_state[0:3] == win_state:
			reward = 1
		if temp_state[3:6] == win_state:
			reward = 1
		if temp_state[6:] == win_state:
			reward = 1
		# 'x' 'o' 'o' / 'o' 'x' 'o'/ 'o' 'o' 'x'
		#  0   1   2     3   4   5    6   7   8
		if (temp_state[0] + temp_state[4] + temp_state[8]) == win_state:
			reward = 1
		if (temp_state[2] + temp_state[4] + temp_state[6]) == win_state:
			reward = 1

		if reward == 1:
			self.number_wins += 1

		state = self.trinary(state)

		self.values[state] = reward

		# V(s) = V(s) + alpha*( V(s') - V(s) )
		#learning_rate = 0.1
		learning_rate = self.learning_rate

		s_prime = state
		state_history = self.state_history
		length_histories = len(state_history) - 1
		for s_idx in range(length_histories-1,-1,-1):
			
			# Get the 9 digit state and map it to a real number
			s = state_history[s_idx]
			s = self.trinary(s)
			
			V_s = self.values[s][0]
			V_s_prime = self.values[s_prime]
			self.values[s][0] = V_s + learning_rate*(V_s_prime - V_s)
			s_prime = s

		# empty the state history container.
		# 
		self.state_history = []


	def take_action(self, environment):
		# environment --> env() object
		# board will be env().board
		board = environment.board
		board = list(board.flatten())

		for idx in range(len(board)):
			board[idx] = int(board[idx])

		# first we need to get the set of possible actions from 
		# the current state
		# Essentially we need A(s_j), the set of actions, but the set of actions is dependent on the state
		# If you don't have a rock in hand, throwing a rock isn't in the set of actions you can presently perform.
		possible_next_states = []
		possible_actions = []

		temp_state = ""
		temp_list = None
		for idx in range(len(board)):
			temp_state = ""
			if int(board[idx]) == 0:
				if idx == (len(board) - 1):
					temp_list = board[:idx] + [self.piece]
				else:
					temp_list = board[:idx] + [self.piece] + board[idx+1:]
				
				# put the position we tried in the positions list.
				possible_actions.append(idx)

				for num in temp_list:
					temp_state += str(num)
				#print("temp_state = ",temp_state)
				possible_next_states.append( int(temp_state) )
		###

		# Find the next state with the most value and save the action that leads to it.
		max_V = 0
		max_A = None
		num_half = 0
		for s_prime_idx in range(len(possible_next_states)):
			s_prime = possible_next_states[s_prime_idx]
			s_prime = self.trinary(s_prime)

			value_s_prime = self.values[s_prime][0]

			if value_s_prime == 0.5:
				num_half += 1
			if value_s_prime >= max_V:
				max_V = value_s_prime
				max_A = possible_actions[s_prime_idx]

		# Choose action with some probability epsilon of exploring,
		# rather than exploiting the one with the highest value
		epsilon = 0.1
		num = np.random.rand()
		#print("value_s_prime = ",value_s_prime)
		#print("num_half = ", num_half, " len(possible_next_states) = ", len(possible_next_states))
		if num_half == (len(possible_next_states)):
			action = np.random.randint(len(possible_next_states))
			action = possible_actions[action]
		else:
			if num <= epsilon:
				action = np.random.randint(len(possible_next_states))
				action = possible_actions[action]
		#		print("EPSILON")
			else:
				action = max_A
		#print("possible_next_states = ", possible_next_states)
		#print("possible_actions = ", possible_actions)
		#print("action = ",action)
		#print("max_A = ",max_A)
		#action = possible_actions[action]
		# 1,2,3,4,5,6,7,8,9
		# 0,1,2,3,4,5,6,7,8
		#print("action = ",action)
		if action < 3:
			pos_1 = 0
			pos_2 = action
			environment.update_board(pos_1, pos_2, self.piece)
			return None
		if (action < 6) and (action > 2):
			pos_1 = 1
			pos_2 = (action % 3)
			environment.update_board(pos_1, pos_2, self.piece)
			return None
		if (action < 9) and (action > 5):
			pos_1 = 2
			pos_2 = (action % 3)
			environment.update_board(pos_1, pos_2, self.piece)
			return None
			
		pass



# This is the only function from the Udemy course. 
# All of the rest was written by myself.
def play_game(p1, p2, env, draw = False):


	current_player = None

	starter = np.random.rand()

	if starter > 0.5:
		current_player = p1
	else:
		current_player = p2

	while not env.game_over():

		if starter>1:
			# Alternate b/w players
			if current_player == p1:
				current_player = p2
			else:
				current_player = p1
		else:
			starter = starter + 1
		# Draw the board that the player sees before performing an action
		if draw:
			if (draw == 1) and (current_player == p1):
				env.draw_board()
			if (draw == 2) and (current_player == p2):
				env.draw_board()

		# current player makes a move
		current_player.take_action(env)

		# update state histories
		state = env.get_state()
		p1.state_history.append(state)
		#env.draw_board()

		p2.state_history.append(state)

	if draw:
		env.draw_board()
	
	# update value function
	final_state = env.get_state()

	p1.update(final_state)
	p2.update(final_state)


def main(save_values=False):
	lrn_rate1 = 0.1
	lrn_rate2 = 0
	player1 = player('o', lrn_rate1,True)
	player2 = player('x', lrn_rate2)
	n = 10001

	info = []
	infoP1 = 0 # keep track of wins per k%x
	infoP2 = 0 

	for k in range(1,n):
		print("k = ",k+1)
		envir = env()
		play_game(player1, player2, envir, False)
		
		if (0):#(k % 200) == 0:
			plt.plot(player1.values, linestyle="None", marker="o", color="red",label="P1, alpha="+str(lrn_rate1))
			plt.plot(player2.values, linestyle="None", marker="o", color="blue", label="P2, alpha="+str(lrn_rate2))
			plt.ylim(-.5,1.1)
			plt.legend()
			plt.show()
			print("player1 tot wins = ",player1.number_wins)
			print("player2 tot wins = ",player2.number_wins)
			print("k = ",k)
			info.append(((player1.number_wins- infoP1)/200,(player2.number_wins- infoP2)/200))
			infoP1 = player1.number_wins - infoP1
			infoP2 = player2.number_wins - infoP2
			input("continue?")

	envir = env()
	play_game(player1, player2, envir, 1)

	if save_values == True:
		df = pd.DataFrame(data = player1.values)
		df.to_csv("C:/Users/Jyler/Documents/ProgrammingProjects/reinforcement/player1_learned_values.csv",header=False,index=False)


	plt.plot(player1.values, linestyle="None", marker="o", color="red",label="P1, alpha="+str(lrn_rate1))
	plt.plot(player2.values, linestyle="None", marker="o", color="blue", label="P2, alpha="+str(lrn_rate2))
	plt.ylim(-.5,1.1)
	plt.legend()
	plt.show()
	print("player1 wins = ",player1.number_wins)
	print("player2 wins = ",player2.number_wins)
	print("n = ",n)
	print("info list = ",info)
	input("continue?")
	plt.close('all')
	if len(info) != 0:
		info_p1 = [i[0] for i in info]
		info_p2 = [i[1] for i in info]
		x_vals = list(range(1,len(info)+1))
		plt.plot(x_vals,info_p1, linestyle="None", marker="o", color="red",label="P1, alpha="+str(lrn_rate1))
		plt.plot(x_vals,info_p2, linestyle="None", marker="o", color="blue", label="P2, alpha="+str(lrn_rate2))
		plt.show()


def play_against_cp():
	lrn_rate1 = 0.1
	player1 = player('o', lrn_rate1,True)
	print("You are 'x'")
	environment = env()

	rand_num = np.random.rand()
	if rand_num > 0.5:
		print("The computer is going first")
		first = 1
	else:
		print("You go first")
		first = 0

	while not environment.game_over():
		environment.draw_board()
		if first:
			player1.take_action(environment)
			first = 0
		else:
			print("Choose a postion. Enter as tuple, like '(2,0)' Top left is (0,0). Top right is (0,2)")
			user_act = input()
			if user_act[0] == "(":
				user_act = (int(user_act[1]),int(user_act[3]) )
			environment.update_board(user_act[0],user_act[1],'x')
			first = 1