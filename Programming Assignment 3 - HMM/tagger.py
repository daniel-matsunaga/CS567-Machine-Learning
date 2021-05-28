import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	# Edit here
	
	# Observations = sentences
	# States = tags
	S = len(tags)

	# Calculate pi and state_dict
	states = {}
	pi = np.zeros([S])
	for tag in tags:
		states[tag] = 0

	count = 0
	for sentence in train_data:
		# Count initial states
		states[sentence.tags[0]] += 1
		# Count number of sequences
		count += 1

	idx = 0
	for state, value in states.items():
		# Calculate pi probabilites
		pi[idx] = float(value / count)
		# Map state to index in pi 
		states[state] = idx
		idx += 1

	# Calculate Transition probablilities
	num_states = {}
	for tag in tags:
		num_states[tag] = 0

	A = np.zeros([S,S])
	for sentence in train_data:
		for i in range(sentence.length - 1):
			# Count total number of transitions starting at each state and add to dictionary
			num_states[sentence.tags[i]] += 1

			# Count transitions from s1->s2
			x = states[sentence.tags[i]]
			y = states[sentence.tags[i+1]]
			A[x,y] += 1

	for state, value in num_states.items():
		for j in range(S):
			# Calculate the probabilites
			x = states[state]
			A[x,j] = float(A[x,j] / value)

	# Calculate Obeservation probabilities and obs dictionary
	num_obs = {}
	obs = {}
	idx = 0
	for sentence in train_data:
		for word in sentence.words:
			# Count the total number of each observation
			if word in num_obs:
				num_obs[word] += 1
			else:
				num_obs[word] = 1
				# Map observation to index
				obs[word] = idx
				idx += 1

	B = np.zeros([S,len(obs)])
	for sentence in train_data:
		# Count observations given state
		for i in range(sentence.length):
			x = states[sentence.tags[i]]
			y = obs[sentence.words[i]]
			B[x,y] += 1
	
	for state in states:
		for o, value in num_obs.items():
			# Calculate the probabilities
			x = states[state]
			y = obs[o]
			B[x,y] = float(B[x,y] / value)

	model = HMM(pi = pi, A = A, B = B, obs_dict = obs, state_dict = states)
	###################################################
	return model

# TODO:
def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	# Edit here


	S = len(tags)
	# Check if there are any new observations
	new_words = []
	for sentence in test_data:
		for word in sentence.words:
			if word not in model.obs_dict:
				new_words.append(word)

	if len(new_words) > 0:

		blen = len(model.B[0])
		newlen = blen + len(new_words)
		newB = np.zeros([S,newlen])

		# Add new columns
		for i in range(S):
			for j in range(newlen):
				if j < blen:
					newB[i,j] = model.B[i,j]
				else:
					newB[i,j] = 10 ** -6

		model.B = newB

		# Add new words to dictionary
		for word in new_words:
			model.obs_dict[word] = blen
			blen += 1


	# Run Viterbi Algorithm
	for sentence in test_data:
		tagging.append(model.viterbi(sentence.words))

	###################################################
	return tagging
