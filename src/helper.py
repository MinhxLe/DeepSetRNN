from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import xavier_uniform_

import numpy as np
import pickle
import gzip
from copy import deepcopy

from sklearn.metrics import f1_score, hamming_loss


def pad_sentences(vectorized_seqs, do_sort=True):
  # add in 0 padding at the end of the "batch of sentences"
  # get the length of each seq in your batch
  seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))
  # dump padding everywhere, and place seqs on the left.
  # NOTE: you only need a tensor as big as your longest sequence
  seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long()  # .cuda()
  for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
    seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
  perm_idx = None
  if do_sort:
    # SORT YOUR TENSORS BY LENGTH!
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
  # 2d tensor, row=sentence col=word_indexing
  return seq_lengths, seq_tensor, perm_idx


def construct_neighbor_list(data_in, GO_all_info, get_parent=True, entailment=True, do_filter_down=False):
  # take 1 batch of data_in. construct the neighbors
  # CAN BE DONE IN PRE-PROCESSING
  
  neighbor = []
  
  if entailment: 
    # unique name from left-right-hand side
    left_right_hand = list(set(data_in['name'][0] + data_in['name'][1]))
  else: ## use @construct_neighbor_list for label data 
    left_right_hand = list(data_in['label_index_map'].keys())
  
  for go_name in left_right_hand:  # name of the GO in order of appearances 
    if get_parent:  
      if GO_all_info[go_name].parent_info is not None: 
        neighbor = neighbor + GO_all_info[go_name].parent_info['neighbor_name'] ## must not be None
    #
    else: ## get children 
      if GO_all_info[go_name].children_info is not None: ## really rare to get all nodes with no children ?? so we don't need to worry ?? 
        ## append the children to @neighbor 
        nn = GO_all_info[go_name].children_info['neighbor_name'] ## num neighbor 
        if do_filter_down and (len(nn)>10) : 
          nn = np.random.permutation(nn)[ 0:10 ]
          nn = nn.tolist() ## convert back to array 
        neighbor = neighbor + nn
      # print ('go term that fails {}'.format(go_name) )


  if len (neighbor) == 0: ## possible we get all leaf nodes, so we have no children 
    return {}

  neighbor = list(set(neighbor))
  neighbor.sort()

  ## append neighbors
  indexing = []
  for go_name in neighbor:
    indexing.append(GO_all_info[go_name].word_indexing) 
    # len_sent.append( GO_all_info[go_name].sent_len )


  # pass all neighbors into LSTM or CNN, extract them back later.
  seq_lengths, seq_tensor, perm_idx = pad_sentences(indexing, do_sort=False) # @indexing is array of many array 
  neighbor_data = {}
  neighbor_data['index'] = seq_tensor
  neighbor_data['len'] = seq_lengths
  neighbor_data['name'] = neighbor
 
  return neighbor_data


def make_ground_truth_np(data_in_batch, do_vstack=False):
  ground_truth_np = None  # get true labels
  for i in range(len(data_in_batch)):
    if ground_truth_np is None:
      ground_truth_np = data_in_batch[i]['true_label'].data.numpy()
    else:
      if do_vstack:
        ground_truth_np = np.vstack((ground_truth_np, data_in_batch[i]['true_label'].data.numpy()))  # for animo GO
      else:
        ground_truth_np = np.hstack((ground_truth_np, data_in_batch[i]['true_label'].data.numpy()))
  return ground_truth_np


def append_predicted_value(predicted_value_tensor, predicted_value_np): 
  # @predicted_value_tensor is some pytorch format 
  # take output of each batch, then put them in numpy form
  if predicted_value_np is None:
    predicted_value_np = predicted_value_tensor.cpu().data.numpy()
  else:
    predicted_value_np = np.vstack((predicted_value_np, predicted_value_tensor.cpu().data.numpy()))
  return predicted_value_np


def binary_loss_metric(predicted_value_np, ground_truth):
  metrics = { 'f1' : f1_score ( ground_truth, predicted_value_np , average='binary' ) } ## care only about yes/no 
  metrics ['hamming loss'] = hamming_loss( ground_truth, predicted_value_np )
  return metrics


def get_go_bio_type ( go_index, go_bio_type ) : ## get which go terms in which bio type 
  ## return dict of indexing for each type BP, CC, MF 
  index_in_type ={}
  for bio_type in [ 'biological_process', "cellular_component", "molecular_function"] : 
    val = []
    for g in go_index.keys() : # for each of the GO term tested, we see what type it is in.
      if g in go_bio_type[bio_type] : ## found in BP for example .
        val.append ( go_bio_type[bio_type].index(g) ) ## append the indexing to be extracted from the whole 1-hot np.array 
    val.sort() 
    index_in_type [ bio_type ] = np.array (val)
  return index_in_type


