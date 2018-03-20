
from math import factorial
import numpy as np
import pandas as pd




# take the number 3, want to get all permutations of 1,2,3.
# (1,2,3), (1,3,2), (3,1,2), (3,2,1), (2,1,3), (2,3,1)
# or for 4, all permutations of 1,2,3,4
# by using recursion

def swapper(x,y):

	return y,x


def permutor(x, depth):
	# will be list.

	length = len(x)
	copied_list = list(x)
	num1 = depth % length
	num2 = (depth + 1) % length

	copied_list[num1], copied_list[num2] = copied_list[num2], copied_list[num1]
	print("copied_list: ", copied_list, " depth: ",depth)
	print("")
	if copied_list in cont:
		num2 = (depth + 2) % length
		copied_list[num1], copied_list[num2] = copied_list[num2], copied_list[num1]
		print("copied_list2: ",copied_list)
		print("")
		if copied_list in cont:
			return 1
		else:
			cont.append(copied_list)
			permutor(copied_list,depth+1)
	else:
		cont.append(copied_list)
		print("copied_list before r", copied_list)
		r = permutor(copied_list,depth+1)
		if r == 1:
			print(cont)

	return 1

	print('cont:', cont)

#cont = []
def main(x):
	init = []
	# gets first 

	if x == 1:
		return 1

	for i in range(1,x+1):
		init.append(i)

	global cont
	cont = []
	cont.append(init)
	permutor(init,0)
	print("")
	print("End:", cont)
	print("length: ", len(cont)," factorial: ", factorial(x))


		
# 3 - > (3,1,2), (3,2,1)
# 2 - > (2,1,3), (2,3,1)
# 1 - > (1,2,3), (1,3,2)

# (1,2,3) 
# (1,3,2), (2,3,1), (3,2,1)
# (3,1,2), (2,1,3)