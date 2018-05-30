"""
Author: Jyler Menard
Purpose implement a Deep Q Network that uses double Q-learning rather than Q-learning.
Q-learning can easily overestimate the value of an action from a state, resulting in overoptimistic value estimates.
Double Q-learning decouples the action selection step and the action evaluation step. 

"""

import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
#import reinforcement.cart_pole_rbf as cpr
GAMMA = 0.99
ALL_POSSIBLE_ACTIONS = [0,1,2]
GAME = "CartPole-v1"

class NeuralNetwork():

	def __init__(self, n_observations, n_actions):
		# n_observations --> number of input nodes
		# n_actions --> number of output nodes
		self.n_actions = n_actions
		self.n_observations = n_observations
		print("Using Feed-forward Neural Network")
		self.scaler = StandardScaler()

		# MEMORY FOR EXPERIENCE REPLAY
		self.mem = []
		self.mem_min_size = 150
		self.mem_max_size = 10000
		self.mem_full = 0 # Default: False

		self.tester = 0
		##
		# DEFINE NN ARCHITECTURE
		##
		learning_rate = 2.5e-4
		hid1 = 200	#
		hid2 = 200
		#hid3 = 500
		#print("hid1 = ", hid1, " hid2 = ",hid2)
		print("hid1 = ",hid1, " learning_rate = ",learning_rate)
		# DEFINE PLACEHOLDER(S)
		self.x = tf.placeholder(tf.float32, shape=[None,n_observations])
		self.y_true = tf.placeholder(tf.float32, shape=[None,n_actions])
		self.A = tf.placeholder(tf.float32, shape=[None,n_actions])

		# DEFINE VARIABLES
		self.W1 = tf.Variable(tf.truncated_normal([n_observations,hid1],mean=0.0,stddev=0.1))
		self.b1 = tf.Variable(tf.constant(0.1, shape=[hid1]))
		self.W2 = tf.Variable(tf.truncated_normal([hid1,hid2],mean=0.0,stddev=0.1))
		self.b2 = tf.Variable(tf.constant(0.1, shape=[hid2]))
		#self.W3 = tf.Variable(tf.truncated_normal([hid2,hid3],mean=0.0,stddev=0.1))
		#self.b3 = tf.Variable(tf.constant(0.1, shape=[hid3]))
		self.W4 = tf.Variable(tf.truncated_normal([hid2, n_actions],mean=0.0, stddev=0.1))
		self.b4 = tf.Variable(tf.constant(0.1, shape=[n_actions]))

		# DEFINE ARCHITECTURE
		y1 = tf.matmul(self.x, self.W1) + self.b1
		z1 = tf.nn.tanh(y1)
		y2 = tf.matmul(z1, self.W2) + self.b2
		z2 = tf.nn.tanh(y2)
		#y3 = tf.matmul(z2, self.W3) + self.b3
		z3 = z2#tf.nn.relu(y3)
		y_pred = tf.matmul(z3, self.W4) + self.b4	

		# DEFINE OPERATIONS AND COST FUNCTION
		#selected_action_values = tf.reduce_sum(
      		#tf.multiply(y_pred,self.A),
      	#	y_pred * tf.one_hot(self.A, n_actions),
      	#	keepdims=True
      		#reduction_indices=[1]
		#	)
		selected_action_values = y_pred * self.A#tf.one_hot(self.A, n_actions)

		delta = selected_action_values - self.y_true
		#delta = y_pred - self.y_true
		#cost = tf.reduce_sum( delta*delta )
		cost = tf.reduce_sum(tf.square(delta))

		# OPS
		self.train_ops = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
		#self.train_ops = tf.train.AdamOptimizer(learning_rate).minimize(cost)
		self.predict_ops = y_pred

		self.sess = tf.InteractiveSession()
		sess = self.sess
		init = tf.global_variables_initializer()
		sess.run(init)

		self.grad_vals = []
		pass

	def feedfwd(self,X,size):
		input_size = int(X.get_shape()[1])
		y = tf.matmul(X,self.W) + self.b1
		z = tf.nn.relu(y)
		return z

	def update_test(self,num):
		self.tester += 1

	def partial_fit(self,G,X):
		# X --> observations, 1x4 initially
		# G --> vector of returns, 1x2 initially
		
		#print("Shape = ", G.shape, " G = ",G)
		if self.mem_full:
			batch_X, batch_G, batch_A = self.batch_replay(32)
			feed_dictionary = {self.x:batch_X,self.y_true:batch_G, self.A:batch_A}
			self.sess.run(self.train_ops, feed_dict=feed_dictionary)

	def predict(self,X):
		# X --> observations
		if not self.mem_full:
			return np.random.random((1,self.n_actions))
		y = self.sess.run(self.predict_ops, feed_dict={self.x:X})
		#print("predicted y = ",y)
		return y

	def get_state(self,observations):
		shape = observations.shape[0]
		y = observations.reshape((1,shape))

		return y

	def store_in_mem(self,s,a,r,s_prime,G):
		tup_4 = (s,a,r,s_prime,G)
		if self.mem_full:
			if len(self.mem)>=self.mem_max_size:
				self.mem.pop(0)
			self.mem.append(tup_4)
		else:
			self.mem.append(tup_4)
			if len(self.mem) == self.mem_min_size:
				print("Memory full")
				self.mem_full = 1

	def batch_replay(self, batch_size):
		# mem filled with 4-tuples (s,a,r,s')
		# Need to grab random batch of size batch_size
		temp_batches = self.mem.copy()
		np.random.shuffle(temp_batches)
		temp_batches = temp_batches[:batch_size]
		batch_G = np.zeros((batch_size,self.n_actions))
		batch_X = np.zeros((batch_size,self.n_observations))
		batch_A = np.zeros((batch_size,self.n_actions))#,dtype=np.int32)
		#batch_A = []

		for i in range(batch_size):
			s, a, r, s_prime,temp_G = temp_batches[i]
			V_s_prime = self.predict(s_prime)
			#batch_G[i][a] = r + GAMMA*np.max(V_s_prime)
			
			batch_G[i] = temp_G
			batch_X[i] = s
			#batch_X[i] *= batch_A[i]
			batch_A[i][a] = 1
			#batch_A.append(a)
		#print(batch_A)
		return batch_X, batch_G, batch_A


def epsilon_greedy(model,model_2, s, epsilon, env):

	p = np.random.random()
	if p <= epsilon:
		action = env.action_space.sample()#np.random.choice(ALL_POSSIBLE_ACTIONS)
		return action
	
	# Compute the value for each action given the state
	V = model.predict(s)
	V_2 = model_2.predict(s)
	
	return np.argmax(V + V_2)


def get_return(model_1,model_2, s_prime,a,r, target_model):
	
	## target_model says which model is going to make the target, Y, of Y-Y_pred.

	if target_model == 1:
		# model 1 selects act, model 2 evaluates it.
		V_s_prime = model_2.predict(s_prime)
		#print(V_s_prime, V_s_prime.shape)
		V_s_prime_eval_act = model_1.predict(s_prime)
		state_act_val = V_s_prime_eval_act[0][np.argmax(V_s_prime)]
		G = np.zeros((1,V_s_prime.shape[1]))
	else:
		# model 2 selects act, model 1 evaluates it.
		V_s_prime = model_1.predict(s_prime)
		#print(V_s_prime, V_s_prime.shape)
		V_s_prime_eval_act = model_2.predict(s_prime)
		state_act_val = V_s_prime_eval_act[0][np.argmax(V_s_prime)]
		G = np.zeros((1,V_s_prime.shape[1]))	

	G[0][a] = r + GAMMA*state_act_val
	
	return G

def reward_function(observation, target_pos):
	y = (target_pos - observation[0])/(target_pos*3)
	return abs(y * 100)


def play_episode(env, model, model_2, epsilon, tot_acts):

	done = False
	obs = env.reset()
	s = model.get_state(obs)

	num = 0
	run_avg = 0
	prnt = 1
	while not done and num<500:
		num+=1
		if num>300 and prnt==1:
			print("num > 300, performing very well")
			prnt = 0

		tot_acts += 1
		a = epsilon_greedy(model,model_2, s, epsilon,env)
		observation, r, done, _ = env.step(a)

		s_prime = model.get_state(observation)
		
		# FOR CART-POLE GAME
		if done:
			r = -200
		if r >-100:
			run_avg += 1

		# FOR MOUNTAIN CAR
		#if observation[0] > 0:
		#	r = +50
		#r = reward_function(observation, 0.6)

		num_p = np.random.random()
		if num_p >= 0.5:
			G = get_return(model,model_2, s_prime, a,r,2)

			model.store_in_mem(s,a,r,s_prime,G)
			model.partial_fit(G, s)
		else:
			G = get_return(model,model_2, s_prime, a,r,1)

			model_2.store_in_mem(s,a,r,s_prime,G)
			model_2.partial_fit(G, s)
			
		s = s_prime
	
	return run_avg, tot_acts


def main(N=100):

	#env = gym.make("CartPole-v1")
	env = gym.make(GAME)
	D = len(env.observation_space.sample())
	K = env.action_space.n

	model = NeuralNetwork(D,K)
	model_2 = NeuralNetwork(D,K)

	running_average = []
	positions = []
	tot_run_avg = 0
	tot_acts = 0
	for i in range(N):
		epsilon = 1.0/(np.sqrt(i) + 1)
		temp_run_avg, temp_tot_acts = play_episode(env, model, model_2, epsilon, tot_acts)
		tot_run_avg += temp_run_avg
		tot_acts += temp_tot_acts 

		if i%50 == 0 and i!=0:
			tot_run_avg/= 50
			print("episode = ",i, " avg over 50 = ",tot_run_avg)
			running_average.append(tot_run_avg)
			tot_run_avg = 0

	plt.plot(running_average)
	plt.xlabel("No. games (x100)")
	plt.ylabel("50-Game Time Average")
	plt.show()

	input("test?")

	test(model, model_2, env)

def test(model,model_2, env):
	num=0
	alpha = 0.1
	for i in range(10):
		done = False
		obs = env.reset()
		s = model.get_state(obs)

		while not done:
#			if i == 1:
#				env.render()
			a = epsilon_greedy(model,model_2, s, -1,env)
			observation, r, done, _ = env.step(a)
			s_prime = model.get_state(observation)
			

			s = s_prime
			num+=1
#		if i == 1:
#			env.close()

	print("tot = ",num/10)
	
	#env = gym.make("CartPole-v1")
	env = gym.make(GAME)
	done =False
	obs = env.reset()
	s = model.get_state(obs)
	while not done:
		env.render()
		a = epsilon_greedy(model,model_2, s, -1,env)
		observation, r, done, _ = env.step(a)
		s_prime = model.get_state(observation)
		s = s_prime
	env.close()


