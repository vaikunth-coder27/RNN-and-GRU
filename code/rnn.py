# coding: utf-8
from rnnmath import *
from model import Model, is_param, is_delta

class RNN(Model):
	'''
	This class implements Recurrent Neural Networks.
	
	You should implement code in the following functions:
		predict				->	predict an output sequence for a given input sequence
		acc_deltas			->	accumulate update weights for the RNNs weight matrices, standard Back Propagation
		acc_deltas_bptt		->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time
		acc_deltas_np		->	accumulate update weights for the RNNs weight matrices, standard Back Propagation -- for number predictions
		acc_deltas_bptt_np	->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time -- for number predictions

	Do NOT modify any other methods!
	Do NOT change any method signatures!
	'''
	
	def __init__(self, vocab_size, hidden_dims, out_vocab_size):
		'''
		initialize the RNN with random weight matrices.
		
		DO NOT CHANGE THIS - The order of the parameters is important and must stay the same.
		
		vocab_size		size of vocabulary that is being used
		hidden_dims		number of hidden units
		out_vocab_size	size of the output vocabulary
		'''

		super().__init__(vocab_size, hidden_dims, out_vocab_size)

		# matrices V (input -> hidden), W (hidden -> output), U (hidden -> hidden)
		with is_param():
			self.U = np.random.randn(self.hidden_dims, self.hidden_dims)*np.sqrt(0.1)
			self.V = np.random.randn(self.hidden_dims, self.vocab_size)*np.sqrt(0.1)
			self.W = np.random.randn(self.out_vocab_size, self.hidden_dims)*np.sqrt(0.1)

		# matrices to accumulate weight updates
		with is_delta():
			self.deltaU = np.zeros_like(self.U)
			self.deltaV = np.zeros_like(self.V)
			self.deltaW = np.zeros_like(self.W)

	def predict(self, x):
		'''
		predict an output sequence y for a given input sequence x
		
		x	list of words, as indices, e.g.: [0, 4, 2]
		
		returns	y,s
		y	matrix of probability vectors for each input word
		s	matrix of hidden layers for each input word
		
		'''

		
		# matrix s for hidden states, y for output states, given input x.
		# rows correspond to times t, i.e., input words
		# s has one more row, since we need to look back even at time 0 (s(t=0-1) will just be [0. 0. ....] )
		s = np.zeros((len(x) + 1, self.hidden_dims))
		y = np.zeros((len(x), self.out_vocab_size))

		for t in range(len(x)):
			##########################
			# --- your code here --- #
			##########################
			# if t==0:
			# 	s[0] = sigmoid(np.dot(self.V,make_onehot(x[0],x[0].shape)))
			# else:
			# 
			if t==0:
				s[t] = sigmoid(np.dot(self.V,make_onehot(x[t],self.vocab_size)))
			else:
				s[t] = sigmoid(np.dot(self.V,make_onehot(x[t],self.vocab_size))+np.dot(self.U,s[t-1]))
			
			y[t] = softmax(np.dot(self.W,s[t]))
			
		return y, s
	
	def acc_deltas(self, x, d, y, s):
		'''
		accumulate updates for V, W, U
		standard back propagation
		
		this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
		
		x	list of words, as indices, e.g.: [0, 4, 2]
		d	list of words, as indices, e.g.: [4, 2, 3]
		y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
			should be part of the return value of predict(x)
		s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
			should be part of the return value of predict(x)
		
		no return values
		'''

		for t in reversed(range(len(x))):
			##########################
			# --- your code here --- #
			##########################
			# delta W
			one_d = make_onehot(d[t],self.out_vocab_size)
			deltaout = np.multiply(one_d-y[t],np.ones(self.out_vocab_size))
			self.deltaW+=np.outer(deltaout,s[t])
			# delta V
			sigderivative = np.multiply(s[t],np.ones(s[t].shape)-s[t])
			deltain = np.multiply(np.dot(np.transpose(self.W),deltaout),sigderivative)
			self.deltaV += np.outer(deltain,make_onehot(x[t],self.vocab_size))
			#delta U
			self.deltaU += np.outer(deltain,s[t-1])

	def acc_deltas_np(self, x, d, y, s):
		'''
		accumulate updates for V, W, U
		standard back propagation
		
		this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
		for number prediction task, we do binary prediction, 0 or 1

		x	list of words, as indices, e.g.: [0, 4, 2]
		d	array with one element, as indices, e.g.: [0] or [1]
		y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
			should be part of the return value of predict(x)
		s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
			should be part of the return value of predict(x)
		
		no return values
		'''

		##########################
		# --- your code here --- #
		##########################
		
	def acc_deltas_bptt(self, x, d, y, s, steps):
		'''
		accumulate updates for V, W, U
		back propagation through time (BPTT)
		
		this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
		
		x		list of words, as indices, e.g.: [0, 4, 2]
		d		list of words, as indices, e.g.: [4, 2, 3]
		y		predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
				should be part of the return value of predict(x)
		s		predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
				should be part of the return value of predict(x)
		steps	number of time steps to go back in BPTT
		
		no return values
		'''
		for t in reversed(range(len(x))):
			##########################
			# --- your code here --- #
			##########################
			# delta W
			one_d = make_onehot(d[t],self.out_vocab_size)
			deltaout = np.multiply(one_d-y[t],np.ones(self.out_vocab_size))
			self.deltaW+=np.outer(deltaout,s[t])

			# delta V
			sigderivative = np.multiply(s[t],np.ones(s[t].shape)-s[t])
			deltain = np.multiply(np.dot(np.transpose(self.W),deltaout),sigderivative)
			self.deltaV += np.outer(deltain,make_onehot(x[t],self.vocab_size))
			#delta U
			self.deltaU += np.outer(deltain,s[t-1])


			newdelta = np.multiply(np.dot(np.transpose(self.W),deltaout),np.multiply(s[t],np.ones(len(s[t]))-s[t]))
			for i in range(1,steps+1):
				if t-i>=0:
					newdelta = np.multiply(np.dot(np.transpose(self.U),newdelta),np.multiply(s[t-i],np.ones(len(s[t-i]))-s[t-i]))
					self.deltaV += np.outer(newdelta,make_onehot(x[t-i],self.vocab_size))

					self.deltaU += np.outer(newdelta,s[t-i-1])

			'''for i in range(1,steps+1):
				if t-i<0:
					continue
				sigderivative = np.multiply(s[t-i+1],np.ones(len(s[t-i+1]))-s[t-i+1])
				deltain[t-i]=np.multiply(np.dot(np.transpose(self.U),deltain[t-i+1]),sigderivative)
				self.deltaV = 0.1*np.cross(x[t-i],deltain[t-i])
			# delta V
			if t-steps>0:
				sigderivative = np.dot(s[t-steps],np.ones(len(s[t-steps]))-s[t-steps])
				deltain[t]=np.dot(np.transpose(self.U)*deltain[t-steps+1],sigderivative)
				self.deltaV = 0.1*np.cross(x[t-steps],deltain[t])
			else:
				sigderivative = np.dot(s[0],np.ones(len(s[0]))-s[0])
				deltain[t]=np.dot(np.transpose(self.U)*deltain[1],sigderivative)
				self.deltaV = 0.1*np.cross(x[0],deltain[t])'''
			


	def acc_deltas_bptt_np(self, x, d, y, s, steps):
		'''
		accumulate updates for V, W, U
		back propagation through time (BPTT)

		this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
		for number prediction task, we do binary prediction, 0 or 1

		x	list of words, as indices, e.g.: [0, 4, 2]
		d	array with one element, as indices, e.g.: [0] or [1]
		y		predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
				should be part of the return value of predict(x)
		s		predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
				should be part of the return value of predict(x)
		steps	number of time steps to go back in BPTT

		no return values
		'''

		##########################
		# --- your code here --- #
		##########################
		pass