

import numpy as np
import matplotlib.pyplot as plt
import random


class bandit():

	def __init__(self, m, initial_val):

		self.true_mean = m
		self.sample_mean = initial_val
		self.N = 0
		self.temp_val = 0

	def pull(self):

		mn = self.true_mean
		val = np.random.normal(mn, 1, None)
		self.update(val)

	def update(self, x):
		self.N += 1

		new_sample_mean = (1 - (1/self.N))*(self.sample_mean) + (1/self.N)*x
		self.sample_mean = new_sample_mean
"""
class bandit_time():

	def __init__(self, m, initial_val):

		self.true_mean = m 
"""
# uses epsilon greedy algorithm to learn
def three_armed_bandit(mn1, mn2, mn3, eps, n):

	N = n
	epsilon = eps
	bandits = [bandit(mn1,0), bandit(mn2,0), bandit(mn3,0)]
	tot = 0
	data = np.ones([N,1])

	for i in range(N):
		p = np.random.rand()

		if p < epsilon:
			# explore bandits
			use = np.random.randint(3)
		else:
			# exploit one with max sample mean
			bandits_xbar = [bandits[0].sample_mean, bandits[1].sample_mean, bandits[2].sample_mean]
			use = np.argmax(bandits_xbar)
		
		bandits[use].pull()
		data[i][0] = bandits[use].sample_mean

	cumulative_avg = np.cumsum(data)/(np.arange(N) + 1)
	x_vals = np.arange(0,N,1)

	return cumulative_avg
#	print("bandit1 = ",bandits[0].sample_mean,"bandit2 = ",bandits[1].sample_mean,"bandit3 = ",bandits[2].sample_mean)

# simply greedy algorithm, but initial value is very large, so it encourage exploration.
def optimistic_initial_val(mn1,mn2,mn3,n):
	N = n
	bandits = [bandit(mn1,10),bandit(mn2,10),bandit(mn3,10)]
	tot = 0
	data = np.ones([N,1])

	for i in range(N):
		bandits_xbar = [bandits[0].sample_mean, bandits[1].sample_mean, bandits[2].sample_mean]
		use = np.argmax(bandits_xbar)		
		bandits[use].pull()
		data[i][0] = bandits[use].sample_mean

	cumulative_avg = np.cumsum(data)/(np.arange(N) + 1)
	x_vals = np.arange(0,N,1)

	return cumulative_avg

# Using Chernoff-Hoefding bounds to learn. Need to do more reading on this.
def UCH_bound(mn1,mn2,mn3,n):
	N = n
	bandits = [bandit(mn1,10),bandit(mn2,10),bandit(mn3,10)]
	tot = 0
	data = np.ones([N,1])

	for i in range(1,N+1):
		bandits_CH_bound = [np.sqrt( (2*np.log(i)/(bandits[0].N + 0.0001)) ),
							np.sqrt( (2*np.log(i)/(bandits[1].N + 0.0001)) ),
							np.sqrt( (2*np.log(i)/(bandits[2].N + 0.0001)) )]
		bandits_xbar = [bandits[0].sample_mean + bandits_CH_bound[0],
						bandits[1].sample_mean + bandits_CH_bound[1],
						bandits[2].sample_mean + bandits_CH_bound[2]]
		use = np.argmax(bandits_xbar)		
		bandits[use].pull()
		data[i-1][0] = bandits[use].sample_mean

	cumulative_avg = np.cumsum(data)/(np.arange(N) + 1)
	x_vals = np.arange(0,N,1)

	return cumulative_avg

# compare different methods
def compare_eps():

	mn1 = 1
	mn2 = 2
	mn3 = 3
	n = 4000

	c_01 = three_armed_bandit(mn1,mn2,mn3,0.1,n)
	c_005 = three_armed_bandit(mn1,mn2,mn3,0.05,n)
	c_001 = three_armed_bandit(mn1,mn2,mn3,0,n)

	# optimistic initial value (works on static problems)
	c_OIV = optimistic_initial_val(mn1,mn2,mn3,n)

	# Chernoff-Hoefding bound as epsilon
	c_UCHB = UCH_bound(mn1,mn2,mn3,n)

	x_vals = np.arange(0,c_01.shape[0],1)

	plt.plot(x_vals,c_001,label="eps = 0.01",color="black")
	plt.plot(x_vals,c_005,label = "eps = 0.05",color="orange")
	plt.plot(x_vals,c_01,label = "eps = 0.1",color="red")
#	plt.plot(x_vals,c_OIV,label = "OIV = 10",color="blue")
	plt.plot(x_vals,c_UCHB,label = "UCHB",color="blue")
	plt.plot(x_vals, np.ones([n,1])*mn1, linestyle="--")
	plt.plot(x_vals, np.ones([n,1])*mn2, linestyle="--")
	plt.plot(x_vals, np.ones([n,1])*mn3, linestyle="--")
	plt.xscale("log")
	plt.legend()
	plt.show()

	plt.plot(x_vals,c_001,label="eps = 0.01",color="black")
	plt.plot(x_vals,c_005,label = "eps = 0.05",color="orange")
	plt.plot(x_vals,c_01,label = "eps = 0.1",color="red")
#	plt.plot(x_vals,c_OIV,label = "OIV = 10",color="blue")
	plt.plot(x_vals,c_UCHB,label = "UCHB",color="blue")			
	plt.plot(x_vals, np.ones([n,1])*mn1, linestyle="--")
	plt.plot(x_vals, np.ones([n,1])*mn2, linestyle="--")
	plt.plot(x_vals, np.ones([n,1])*mn3, linestyle="--")
	plt.legend()
	plt.show()
