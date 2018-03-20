

"""
Hard-coded tic-tac-toe.
author:		Jyler
date:		2018/01/06

Before starting reinforcement learning, where we can make a program learn
how to play tic-tac-toe itself, let's start by hard-coding the rules.


"""
import random


class tictactoeBoard():

	
	def __init__(self):

		self.boardState = {
					1:" ",
					2:" ",
					3:" ",
					4:" ",
					5:" ",
					6:" ",
					7:" ",
					8:" ",
					9:" "
					}

		boardState = self.boardState
		self.listX = []
		self.listO = []


	def getBoard(self):

		boardState = self.boardState

		print("-----" * 2 + "---") # | x | o | x |
		print("| " + boardState[1] + " | " + boardState[2] + " | " + boardState[3] + " |")
		print("-----" * 2 + "---")
		print("| " + boardState[4] + " | " + boardState[5] + " | " + boardState[6] + " |")
		print("-----" * 2 + "---")
		print("| " + boardState[7] + " | " + boardState[8] + " | " + boardState[9] + " |")
		print("-----" * 2 + "---")


	def updateBoard(self,pos,new_values):

		boardState = self.boardState
		listX = self.listX
		listO = self.listO
		boardState[pos] = new_values

		# Keep track of moves that have been made.
		if new_values.lower() == 'x':
			listX = listX.append(pos)
		elif new_values.lower() == 'o':
			listO = listO.append(pos)
		else:
			pass

		pass


	def checkBoard(self):
		# Checks if someone has won or if it is a tie

		boardState = self.boardState
		
		# Check what positions have been played.
		listX = self.listX
		listO = self.listO
		#print(listO,listX)

		# check if someone has won
		# 1 = 'o', 2 = 'x', False = no one
		winBool = self.checkWin(listX, listO)
		if winBool == 1:
			print("O wins")
			return True
		elif winBool == 2:
			print("X wins")
			return True
		else:
			pass

		# check if there is a tie
		tieBool = self.checkTie(listX, listO)
		if tieBool:
			print("Tie.")
			return True
		else:
			return False


	def checkWin(self,listX,listO):

		winStates = [
					{1,2,3},{4,5,6},{7,8,9},{1,4,7},
					{2,5,8},{3,6,9},{1,5,9},{3,5,7}
					]

		for winState in winStates:
			#print("X: ",listX, " O: ",listO)
			#print("winState: ",winState)
			if (winState & set(listX)) == winState:
				
				return 2 # 'x' wins
			elif (winState & set(listO)) == winState:
				
				return 1 # 'O' wins
			else:
				pass
		return False


	def checkTie(self,listX,listO):

		combined = listX + listO

		for i in range(1,10):

			if i not in combined:
				#print("no tie")
				return False
			else:
				pass

		#print("tie")
		return True


	def free(self):

		listX = self.listX
		listO = self.listO

		combined = listX + listO
		freeSpaces = [ space for space in range(1,10) if space not in combined ]

		return freeSpaces 



class playgame():


	def __init__(self):

		# randomize who goes first.
		num = random.random()
		if num >= 0.5:
			human_first = True
			print("You go first.")
		else:
			human_first = False

		# User selects what 'piece' they want to use.
		while True:
			player_label = input("Choose x's or o's:\n")
			if player_label.lower() in ['x','o']:
				break
			else:
				print("Choose a correct label. ['x','o']")
		
		machine_label = [i for i in ['x','o'] if i != player_label][0]

		self.board = tictactoeBoard() # Initialize the board
		board = self.board
		board.getBoard()

		#self.listStates = staticArray()
		# Play until someone wins or there is a tie.
		self.playGame(board, player_label, machine_label, human_first)


	def human(self, board, label):
		message = ("Type move position. Free positions are: " + 
					str(board.free()) + "\n")
		player = int(input(message))

		freePos = board.free()
		while True:
			if player in freePos:
				break
			else:
				print("Are you trying to cheat?")
				print("I'll let you choose again.")
			player = int(input(message))

		board.updateBoard(player,label)
		board.getBoard()

		if board.checkBoard():
			return True
		else:
			pass


	def reset(self):
		self.winCont = []
		self.tieCont = []


	def machine(self, board, label):

		currentBoard = dict(board.boardState)
		#freePos = board.free()
		if label == 'x':
			currentTurn = 1 # so that the think function can check if a winState is for the machine or not
			machine_label = 1	# This should actually be changed s.t., if I wanted, I could play the machine against itself.
		elif label == 'o':
			currentTurn = 0
			machine_label = 0
		self.reset()
		self.storeBoardState = dict(board.boardState)		# This is unnecessary.. I think
		freeS = list([ i for i in currentBoard.keys() if currentBoard[i] not in ['x','o'] ])

		self.think(currentBoard, currentTurn, 0, machine_label)

		print("MACHINE DECI",self.winCont)
		print("TIE: ",self.tieCont)
		if len(self.winCont) != 0:
			tempMoves = 100
			for tup in self.winCont:
				#print("tup", tup)
				if tup[1] < tempMoves:
					tempMoves = tup[1]
					chooseMove = tup[0]
				else:
					pass
		elif len(self.tieCont) != 0:
			chooseMove = self.tieCont[0][0]
		else:
			freeS = self.board.free()
			lengthFreeS = len(freeS)
			idx = random.randint(0,lengthFreeS-1)
			chooseMove = freeS[idx]
			print("I know you've won.")

		#print("DEBUG1", chooseMove, " moves I thought of: ", self.winCont, " tie cont: ",self.tieCont)
		board.updateBoard(chooseMove, label)
		board.getBoard()

		if board.checkBoard():
			return True
		else:
			pass


	def playGame(self, board, player_label, machine_label, human_first):

		if human_first:
			while True:
					# 
					hum_val = self.human(board,player_label)
					if hum_val:
						break
					else:
						pass
					print("________________")

					mach_val = self.machine(board,machine_label)
					if mach_val:
						break
					else:
						pass

		else:
			while True:
					# 
					mach_val = self.machine(board,machine_label)
					if mach_val:
						break
					else:
						pass

					hum_val = self.human(board,player_label)
					if hum_val:
						break
					else:
						pass


	def think(self, board, turn, depth, machine_label):

		# turn = either 0 or 1. 0 ='o', 1 = 'x'. Can cycle by using (x+1) mod 2
		# board = the board state
		boardState = dict(board)	
		freePos = list([ i for i in boardState.keys() if boardState[i] not in ['x','o'] ])
		
		# If first move, choose anywhere.
		if len(freePos) == 9:
			posTemp = random.randint(1,9)
			self.winCont.append((posTemp,2))		
			return 0

		# Cycle which turn the machine is considering.
		if turn == 0:
			presentTurn = 'o'
		elif turn == 1:
			presentTurn = 'x'

		winState = self.checkWin(boardState) # returns 2 if 'x' wins, 1 if 'o' wins, else False 
		tieState = self.checkTie(boardState) # return True if tie, else False
		
		if (winState != False):#!= machine_label) and (winState != False):				# Machine loses.

			if int(not(turn)) != machine_label: # Is it a winState for the opponent?
			
				if depth == 2:
					temp1 = 5
				else:
					temp1 = 2
			
			elif depth == 1:		# Has only one move been played?
		#		print("HERERERERE!")
				temp1 = 1
			else:
				temp1 = 0			# WinState
			return temp1

		elif tieState:
			return 3
		else:
			pass

		for pos in range(1,10):

			if pos not in freePos:
				continue

			boardState = dict(board)
			boardState[pos] = presentTurn			

			# recursive step
			temp = self.think(boardState, ((turn+1)%2), depth+1, machine_label)

			if temp == 0:
				if depth == 0:
					self.winCont.append((pos,3))
			elif temp == 1:
				if depth == 0:
					self.winCont.append((pos,2))
			elif temp == 2:
				return 4
			elif temp == 3:
				if depth == 0:		
					self.tieCont.append((pos,1))
			elif temp == 4:
				freePos.remove(pos)
			elif temp == 5:
				return 6
			elif temp == 6:
				if depth == 0:
					freePos.remove(pos)
			else:
				pass

		return 0


	def checkWin(self,boardState):

		listX = [ i for i in boardState.keys() if boardState[i] == 'x' ]
		listO = [ i for i in boardState.keys() if boardState[i] == 'o' ]

		winStates = [
					{1,2,3},{4,5,6},{7,8,9},{1,4,7},
					{2,5,8},{3,6,9},{1,5,9},{3,5,7}
					]

		for winState in winStates:

			if (winState & set(listX)) == winState:
				return 2 # 'x' wins
			elif (winState & set(listO)) == winState:				
				return 1 # 'O' wins
			else:
				pass
		return False


	def checkTie(self,boardState):

		listX = [ i for i in boardState.keys() if boardState[i] == 'x' ]
		listO = [ i for i in boardState.keys() if boardState[i] == 'o' ]
		combined = listX + listO

		for i in range(1,10):

			if i not in combined:
				#print("no tie")
				return False
			else:
				pass

		#print("tie")
		return True



class staticArray():


	def __init__(self):

		self.reset()


	def insert(self, place, data):

		if place == 0:
			self.loss.append(data)
		elif place == 1:
			self.win.append(data)
		elif place == 2:
			self.tie.append(data)
		else:
			pass


	def reset(self):

		self.loss = []
		self.win = []
		self.tie = []


	def search(self, place):

		lossList = self.loss
		winList = self.win
		tieList = self.tie

		if place == 0:
			for i in lossList:
				if len(i) == 2:
					return i
				else:
					pass
		elif place == 1: # winlist
			if len(winList) == 1:
				return True
			else:
				False


	def length(self, place):

		if place == 0:
			y = len(self.loss)
		elif place == 1:
			y = len(self.win)
		elif place == 2:
			y = len(self.tie)
		else:
			pass

		return y