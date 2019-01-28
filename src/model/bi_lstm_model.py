
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, sys
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from torch.nn.init import xavier_uniform_


class bi_lstm_layer (nn.Module):
	# @num_of_word is vocab size.
	def __init__(self, lstm_input_dim, lstm_dim, batch_size):
		super().__init__()
		self.lstm_dim = lstm_dim
		self.lstm = nn.LSTM(lstm_input_dim, lstm_dim // 2, bidirectional=True, num_layers=1,
												batch_first=True)  # divide 2 because each direction needs "half"
		self.batch_size = batch_size
		self.hidden_state = self.init_hidden()

	def init_hidden(self):
		# Before we've done anything, we dont have any hidden state.
		# http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		# if self.doCuda == True:
		#   return (	Variable(torch.zeros(2, self.batch_size, self.lstm_dim // 2)).cuda() ,
		#             Variable(torch.zeros(2, self.batch_size, self.lstm_dim // 2)).cuda() ) # NOTICE. "2" for num_layers when using bi-directional (with 1 layer indication)
		# else:
		return (	Variable(torch.zeros(2, self.batch_size, self.lstm_dim // 2)).cuda(),
							Variable(torch.zeros(2, self.batch_size, self.lstm_dim // 2)).cuda() )

	def forward(self, embeds): 
    ##  REMEMBER TO CLEAR THE HIDDEN_STATE OTHERWISE, WILL SEE OPTIM ERROR 
		lstm_out, self.hidden_state = self.lstm(embeds, self.hidden_state)
		return lstm_out 
	

class bi_lstm_layer_pack_pad (nn.Module):
	# @num_of_word is vocab size.
	def __init__(self, lstm_input_dim, lstm_dim, batch_size, use_cuda=True):
		super().__init__()
		self.lstm_dim = lstm_dim
		self.lstm = nn.LSTM(lstm_input_dim, lstm_dim // 2, bidirectional=True, num_layers=1,
												batch_first=True)  # divide 2 because each direction needs "half"
		self.batch_size = batch_size
		self.drop = nn.Dropout(p=0.1)
		self.use_cuda = use_cuda
		self.hidden_state = self.init_hidden()

	def init_hidden(self):
		# Before we've done anything, we dont have any hidden state.
		# http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		if self.use_cuda == True:
		  return (	Variable(torch.zeros(2, self.batch_size, self.lstm_dim // 2)).cuda() ,
		            Variable(torch.zeros(2, self.batch_size, self.lstm_dim // 2)).cuda() ) # NOTICE. "2" for num_layers when using bi-directional (with 1 layer indication)
		else:
			return (	Variable(torch.zeros(2, self.batch_size, self.lstm_dim // 2)), # .cuda()
								Variable(torch.zeros(2, self.batch_size, self.lstm_dim // 2)) )

	def forward(self, embeds, seq_lengths): ## @seq_lengths is used to avoid useless lstm passing to end of padding. 

		embeds = self.drop ( embeds )
		
		# @embeds must be sorted by len
		# for padding, we need to sort by len
		seq_lengths, idx_sort = np.sort(seq_lengths)[::-1].copy(), np.argsort(-seq_lengths) ## @.copy() is needed to avoid "uncontiguous" block of numbers
		idx_unsort = np.argsort(idx_sort)

		if self.use_cuda:
			idx_sort = Variable(idx_sort).cuda()  # torch.from_numpy(idx_sort) #.cuda()
		else: 
			idx_sort = Variable(idx_sort)

		embeds = embeds.index_select(0, idx_sort ) ## order the embedding by len

		# Handling padding in Recurrent Networks
		packed_input = pack_padded_sequence(embeds, seq_lengths, batch_first=True)
		##  REMEMBER TO CLEAR THE HIDDEN_STATE OTHERWISE, WILL SEE OPTIM ERROR 
		lstm_out, self.hidden_state = self.lstm(packed_input, self.hidden_state)
		# unpack your output if required
		lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True) ## see the zero as padding 
	
		# Un-sort by length, so we get back the original ordering not sorted by len. 
		# idx_unsort = torch.from_numpy(idx_unsort) #.cuda() 
		if self.use_cuda : 
			lstm_out = lstm_out.index_select(0, Variable(idx_unsort).cuda() )
		else: 
			lstm_out = lstm_out.index_select(0, Variable(idx_unsort) )

		return lstm_out 
	
	
