"""
Author: Jyler Menard 
Methods required:

CNN for image recognition of the breakout game
Multi-image 'packages' s.t. the agent can discern vectors
Epsilon-greedy
Double Q-learning
Q-learning can easily overestimate the value of an action from a state, resulting in overoptimistic value estimates.
Double Q-learning decouples the action selection step and the action evaluation step, preventing the value estimates from
being overoptimistic.
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from scipy.misc import imresize

GAMMA = 0.99

class NeuralNetwork():

	def __init__(self, image_height, image_width, n_actions, num_stacked_frames = 4):
		# n_actions --> number of output nodes
		self.n_actions = n_actions
		self.imHeight = image_height
		self.imWidth = image_width
		self.num_frames = num_stacked_frames
		#self.n_observations = n_observations
		print("Using a Convolutional Neural Network")
		self.scaler = StandardScaler()
		self.check = 1

		# MEMORY FOR EXPERIENCE REPLAY
		self.mem = []
		self.mem_min_size = 200
		self.mem_max_size = 1500
		self.mem_full = 0 # Default: False

		self.tester = 0
		##
		# DEFINE NN ARCHITECTURE
		##
		learning_rate = 1e-4
		hid1 = 200	#
		hid2 = 200
		print("Learning_rate = ",learning_rate)

		# DEFINE PLACEHOLDER(S)
		self.x = tf.placeholder(tf.float32, 
					shape=[None,self.imHeight, self.imWidth, self.num_frames])

		self.y_true = tf.placeholder(tf.float32, shape=[None,n_actions])
		self.A = tf.placeholder(tf.float32, shape=[None,n_actions])

		# DEFINE VARIABLES
		self.W1 = tf.Variable(tf.truncated_normal([8,8,self.num_frames, 8],mean=0.0,stddev=0.1))
		self.b1 = tf.Variable(tf.constant(0.1, shape=[8]))

		self.W2 = tf.Variable(tf.truncated_normal([8,8,8,16],mean=0.0,stddev=0.1))
		self.b2 = tf.Variable(tf.constant(0.1, shape=[16]))
		
		self.W3 = tf.Variable(tf.truncated_normal([8,8,32,32],mean=0.0,stddev=0.1))
		self.b3 = tf.Variable(tf.constant(0.1, shape=[32]))
		
		self.W4 = tf.Variable(tf.truncated_normal([20*20*16, 200],mean=0.0, stddev=0.1))
		self.b4 = tf.Variable(tf.constant(0.1, shape=[200]))

		self.W5 = tf.Variable(tf.truncated_normal([200,n_actions],mean=0.0, stddev=0.1))
		self.b5 = tf.Variable(tf.constant(0.1, shape=[n_actions]))

		# DEFINE ARCHITECTURE
		convo_1 = self.conv2d(self.x, self.W1) + self.b1
		relu_1 = tf.nn.relu(convo_1)
		convo_1_pool = self.max_pool_2by2(relu_1)
		
		convo_2 = self.conv2d(convo_1_pool, self.W2) + self.b2
		relu_2 = tf.nn.relu(convo_2)
		convo_2_pool = self.max_pool_2by2(relu_2)
		
		convo_2_flat = tf.reshape(convo_2_pool,[-1,20*20*16])
		
		full_layer_1 = tf.matmul(convo_2_flat, self.W4) + self.b4
		relu_3 = tf.nn.relu(full_layer_1)
		full_layer_2 = tf.matmul(relu_3, self.W5) + self.b5
		y_pred = full_layer_2	

		# DEFINE OPERATIONS AND COST FUNCTION
		selected_action_values = y_pred * self.A#tf.one_hot(self.A, n_actions)
		delta = selected_action_values - self.y_true
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

	def conv2d(self,x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
	
	def max_pool_2by2(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
			strides=[1, 2, 2, 1], padding='SAME')

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
		shape_0 = X.shape
		X = X.reshape((1,shape_0[0],shape_0[1],shape_0[2]))
		if not self.mem_full:
			return np.random.random((1,self.n_actions))
		y = self.sess.run(self.predict_ops, feed_dict={self.x:X})
		#print("predicted y = ",y)
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
		"""
		mem filled with 4-tuples (s,a,r,s')
		TURNS OUT OPENAI GYM HAS STOCHASTIC FRAME SKIPPING BUILT-IN.
		So I don't need to make my own frame skipping algo.
		"""

		mem = self.mem # EDIT: THIS USED TO BE self.mem.copy(), TRYING self.mem
		mem_length = len(mem)
		temp_mem = []
		idx = np.random.randint(low=0, high=mem_length-1, size=batch_size)
		
		# kind of a terrible way of doing this
		for i in list(idx):
			temp_mem.append(mem[i])

		batch_G = np.zeros((batch_size,self.n_actions))
		batch_A = np.zeros((batch_size,self.n_actions))#,dtype=np.int32)
		
		premade_batches = []
		for i in range(batch_size):
			s, a, r, s_prime,temp_G = temp_mem[i]
			premade_batches.append(s)

			batch_G[i] = temp_G
			batch_A[i][a] = 1
		try:
			batch_X = np.stack(premade_batches,axis=0) # dimensions: batch_size * image_height * image_width * 4
		except:
			for i in premade_batches:
				print(i.shape)
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
	
	## Double Q-learning switches between two architectures.
	## The target for one architecture is made by the other architecture.
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


def resize_image(observation):
	# resize the observation and average over the color channels.
	resized = imresize(observation[30:195], size=(80,80,3), interp="nearest")
	resized = resized.mean(axis=2).astype(np.uint8)
	resized = resized/255 # normalize image to values between 0 and 1
	return resized


def update_state(s, observation):
	s.append( resize_image(observation) )
	if len(s)>4:
		s.pop(0)

def play_episode(env, model, model_2, epsilon, tot_acts):

#	lst_observations = []
#	lst_actions = []
#	lst_returns = []
	s = []
	prev_s = []

	done = False
	obs = env.reset()
	update_state(s,obs)	

	num = 0
	tot_rew = 0

	# We don't want to let this run for /too/ long
	while not done and num<1000:
		num+=1

		tot_acts += 1

		# GET AN ACTION
		if len(s)<4:
			a = env.action_space.sample()
		else:
			temp_s = np.stack(s,axis=2)
			a = epsilon_greedy(model,model_2, temp_s, epsilon,env)
		
		# COPY PREVIOUS STATE
		prev_s.append(s[-1])
		if len(prev_s)>4:
			prev_s.pop(0)

		# TAKE ACTION
		observation, r, done, _ = env.step(a)

		# UPDATE NEW STATE
		update_state(s,observation)	

		if done:
			r = -200

		# Switch between architectures for double q-learning
		if (len(s)<4) or (len(prev_s)<4):
			pass
		else:
			num_p = np.random.random()
			temp_prev_s = np.stack(prev_s, axis=2)
			temp_s = np.stack(s, axis=2)
			if num_p >= 0.5:
				G = get_return(model, model_2, temp_s, a, r, 2)

				model.store_in_mem(temp_prev_s, a, r, temp_s, G)
				model.partial_fit(G, temp_s)
			else:
				G = get_return(model, model_2, temp_s, a,r,1)

				model_2.store_in_mem(temp_prev_s, a, r, temp_s, G)
				model_2.partial_fit(G, temp_s)

		tot_rew += r
	
	return tot_rew, tot_acts


def main(N=100):

	env = gym.make("Breakout-v0")
	record_bool = input("Record every perfect cube training episode? [Y/n]")
	while True:
		if record_bool not in ["Y","n"]:
			print("Wrong input")
			record_bool = input("Record every perfect cube training episode? [Y/n]")	
		else:
			break
	if record_bool=="Y":
		env = gym.wrappers.Monitor(env, "videos_breakout",force=True)
	else:
		pass
		
	D = len(env.observation_space.sample())
	K = env.action_space.n

	model = NeuralNetwork(80,80,K,num_stacked_frames=4)
	model_2 = NeuralNetwork(80,80,K,num_stacked_frames=4)

	running_average = []
	positions = []
	tot_run_avg = 0
	tot_acts = 0
	for i in range(N):
		#epsilon = 1.0/(np.sqrt(i) + 1)
		if i < 100:
			epsilon = 1 - (i/100)
		else:
			epsilon = 0.1

		temp_run_avg, temp_tot_acts = play_episode(env, model, model_2, epsilon, tot_acts)
		tot_run_avg += temp_run_avg
		tot_acts += temp_tot_acts 
		print("episodo = ",i)
			
		if i%50 == 0 and i!=0:
			tot_run_avg/= 50
			print("episode = ",i, " avg over 50 = ",tot_run_avg)
			running_average.append(tot_run_avg)
			tot_run_avg = 0

	plt.plot(running_average)
	plt.xlabel("No. games (x50)")
	plt.ylabel("50-Game Time Average")
	plt.show()


