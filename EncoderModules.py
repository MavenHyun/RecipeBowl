import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
from model_utils import *
import pickle
from tqdm import tqdm

# from args import get_parser
from model.blocks import *
import gensim
from env_config import *

VOCAB_SIZE = 30167

def load_word_vectors():
	return gensim.models.KeyedVectors.load_word2vec_format(ROOT_PATH+"vocab.bin", binary=True)

def load_flavorgraph_vectors():
	print("Loaded FlavorGraph vectors!")
	iid2fgvec = pickle.load(open(f'{ROOT_PATH}iid2fgvec_full.pkl', 'rb'))
	fg_indices = iid2fgvec.keys()
	vocab_space = np.zeros((VOCAB_SIZE, 300), np.float)
	for k, v in tqdm(iid2fgvec.items()):
		vocab_space[k, :] = v
	return vocab_space

'''
	###################################
	#                                 #
	#  Ingredients Embedding Modules  #
	#                                 #
	###################################

	SimpleMean - very simple module, just average all the embeddings
	SetTransformer - originally from Reciptor, uses Transformers to compute self-attentions and perform max-pooling

'''

class NullModule(nn.Module):
	def __init__(self):
		super(NullModule, self).__init__()
		self.linear = nn.Linear(300, 1)

	def forward(self, x):
		return x

class BaseModule(nn.Module):
	def __init__(self, args=0):
		super(BaseModule, self).__init__()
		self.food_indices = args.ing_indices
		iid2fgvec = pickle.load(open(f'{ROOT_PATH}iid2fgvec_full.pkl', 'rb'))
		if args.ingred_embed == 'word2vec':
			self.vocab_space = load_word_vectors().vectors
		elif args.ingred_embed == 'flavorgraph':
			self.vocab_space = load_flavorgraph_vectors()
		elif args.ingred_embed == 'random':
			self.vocab_space = load_word_vectors().vectors
		else:
			raise
		vectors = torch.cuda.FloatTensor(self.vocab_space)


		self.u_ingbedding = nn.Sequential(
								nn.Embedding(VOCAB_SIZE, 300, padding_idx=0),
								nn.Linear(300, 300, bias=False), nn.ReLU(), nn.LayerNorm(300))
		self.v_ingbedding = nn.Sequential(
								nn.Embedding(VOCAB_SIZE, 300, padding_idx=0),
								nn.Linear(300, 300, bias=False), nn.ReLU(), nn.LayerNorm(300))

		if args.ingred_embed == 'random':
			nn.init.normal_(self.u_ingbedding[0].weight.data, -1/300, 1/300)
			nn.init.normal_(self.v_ingbedding[0].weight.data, -1/300, 1/300)
		else:
			self.u_ingbedding[0].weight.data.copy_(vectors)
			self.v_ingbedding[0].weight.data.copy_(vectors)
		self.u_ingbedding[0].weight.data[0] = 0.
		self.v_ingbedding[0].weight.data[0] = 0.
		self.u_ingbedding[0].requires_grad = args.fine_tuning
		self.v_ingbedding[0].requires_grad = args.fine_tuning

class SimpleSum(BaseModule):
	def __init__(self, args=0):
		super(SimpleSum, self).__init__(args)
		assert args.hidden_nodes == 300

	def forward(self, batch):
		# batch => [ingrs, num_ingrs, target, tags, iid, rid, rvec, neis]
		B = batch[0].size(0)
		x = self.u_ingbedding(batch[0])
		n = self.v_ingbedding(batch[2].transpose(0,1).repeat(B, 1))
		p = self.v_ingbedding(batch[2]).squeeze(1)
		r = self.v_ingbedding(batch[7])

		sorted_len, sorted_idx = batch[1].sort(0, descending=True)
		batch_max_len = sorted_len.cpu().numpy()[0]
		x = x[:, :batch_max_len, :].sum(1)

		return {
			'ingreds': batch[0],
			'ingred_lengths': batch[1],
			'tags': batch[3],
			'ingred_missing': batch[2],
			'recipe_ids': batch[5],
			'positive_vectors': p,
			'relative_vectors': r,
			'negative_vectors': n,
			'anchor_vectors': x,
			'ingred_relevant': batch[-1],
			'recipe_vectors2': batch[-2]}

class biLSTM(BaseModule):
	'''
		Implementation based on RECIPTOR's biLSTM model
		Citated from paper: An Effective Pretrained Model for Recipe Representation Learning, Li et al., 2020
		Link to citation: https://dl.acm.org/doi/10.1145/3394486.3403223
		Link to github: https://github.com/DiyaLI916/Reciptor
	'''

	def __init__(self, args=0):
		super(biLSTM, self).__init__(args)
		self.irnn = nn.LSTM(input_size=300, hidden_size=int(args.hidden_nodes/2), bidirectional=True, batch_first=True)

	def forward(self, batch):
		# batch => [ingrs, num_ingrs, target, tags, iid, rid, rvec, neis]
		B = batch[0].size(0)
		x = self.u_ingbedding(batch[0])
		n = self.v_ingbedding(batch[2].transpose(0,1).repeat(B, 1))
		p = self.v_ingbedding(batch[2]).squeeze(1)
		r = self.v_ingbedding(batch[7])

		sorted_len, sorted_idx = batch[1].sort(0, descending=True)
		index_sorted_idx = sorted_idx.view(-1,1,1).expand_as(x).cuda()
		sorted_inputs = x.gather(0, index_sorted_idx.long())

		packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
			sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)

		out, hidden = self.irnn(packed_seq)
		_, original_idx = sorted_idx.sort(0, descending=False)

		unsorted_idx = original_idx.view(1,-1,1).expand_as(hidden[0]).cuda()

		x = hidden[0].gather(1, unsorted_idx).transpose(0,1).contiguous()
		x = x.view(x.size(0),x.size(1)*x.size(2))

		return {
			'ingreds': batch[0],
			'ingred_lengths': batch[1],
			'tags': batch[3],
			'ingred_missing': batch[2],
			'recipe_ids': batch[5],
			'positive_vectors': p,
			'relative_vectors': r,
			'negative_vectors': n,
			'anchor_vectors': x,
			'ingred_relevant': batch[-1],
			'recipe_vectors2': batch[-2]}

class DeepSet(BaseModule):
	'''
		Implementation based on Juho Lee's Deep Sets 
		Citated from paper: Deep Sets, Zaheer et al., 2017
		Link to citation: https://dl.acm.org/doi/10.1145/3394486.3403223
		Citated from paper: An Effective Pretrained Model for Recipe Representation Learning, Li et al., 2020
		Link to citation: http://proceedings.mlr.press/v97/lee19d/lee19d.pdf
		Link to github: https://github.com/juho-lee/set_transformer
	'''

	def __init__(self, args=0):
		super(DeepSet, self).__init__(args)
		self.encoder = nn.Sequential(
				nn.Linear(300, args.hidden_nodes, bias=False),
				nn.ReLU(),
				nn.LayerNorm(args.hidden_nodes),
				nn.Linear(args.hidden_nodes, args.hidden_nodes, bias=False),
				nn.ReLU(),
				nn.LayerNorm(args.hidden_nodes),
				nn.Linear(args.hidden_nodes, args.hidden_nodes, bias=False),
				nn.ReLU(),
				nn.LayerNorm(args.hidden_nodes))

		self.decoder = nn.Sequential(
				nn.Linear(args.hidden_nodes, args.hidden_nodes),
				nn.ReLU(),
				nn.LayerNorm(args.hidden_nodes))

	def forward(self, batch):
		# batch => [ingrs, num_ingrs, target, tags, iid, rid, rvec, neis]
		B = batch[0].size(0)
		x = self.u_ingbedding(batch[0])
		n = self.v_ingbedding(batch[2].transpose(0,1).repeat(B, 1))
		p = self.v_ingbedding(batch[2]).squeeze(1)
		r = self.v_ingbedding(batch[7])

		x = self.encoder(x)
		sorted_len, sorted_idx = batch[1].sort(0, descending=True)
		batch_max_len = sorted_len.cpu().numpy()[0]
		x = x[:, :batch_max_len, :].sum(1)
		x = self.decoder(x)

		return {
			'ingreds': batch[0],
			'ingred_lengths': batch[1],
			'tags': batch[3],
			'ingred_missing': batch[2],
			'recipe_ids': batch[5],
			'positive_vectors': p,
			'relative_vectors': r,
			'negative_vectors': n,
			'anchor_vectors': x,
			'ingred_relevant': batch[-1],
			'recipe_vectors2': batch[-2]}


class SetTransformer(BaseModule):
	'''
		Implementation based on Juho Lee's Set Transformer
		Citated from paper: An Effective Pretrained Model for Recipe Representation Learning, Li et al., 2020
		Link to citation: http://proceedings.mlr.press/v97/lee19d/lee19d.pdf
		Link to github: https://github.com/juho-lee/set_transformer
	'''
	def __init__(self, args=0):
		super(SetTransformer, self).__init__(args)
		# Default setting: (M, H, K) = (16, 4, 2)
		self.D = 300
		self.K = args.settf_num_heads  # number of heads
		self.S = args.settf_num_seeds  # number of seed vectors
		self.I = args.settf_num_ipoints
		self.H = args.hidden_nodes
		assert self.D % self.K == 0

		# Model architecture 
		self.layer1 = ISAB(300, self.H, self.K, self.I, ln=True)
		self.layer2 = ISAB(self.H, self.H, self.K, self.I, ln=True)
		self.layer3 = PMA(self.H, self.K, self.S, ln=True)
		self.layer4 = SAB(self.H, self.H, self.K, ln=True)
		self.layer5 = nn.Sequential(
							nn.Linear(self.S*self.H, self.H),
							nn.ReLU(), nn.LayerNorm(self.H))

	def forward(self, batch):
		# batch => [ingrs, num_ingrs, target, tags, iid, rid, rvec, neis]
		B = batch[0].size(0)
		x = self.u_ingbedding(batch[0])
		n = self.v_ingbedding(batch[2].transpose(0,1).repeat(B, 1))
		p = self.v_ingbedding(batch[2]).squeeze(1)
		r = self.v_ingbedding(batch[7])

		sorted_len, sorted_idx = batch[1].sort(0, descending=True)
		batch_max_len = sorted_len.cpu().numpy()[0]
		x = x[:, :batch_max_len, :]
		m1 = x != 0
		m2 = masking_tensor(self.H, batch[1], batch_max_len)

		x = self.layer1(x) 
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x.reshape(-1, self.S*self.H))

		return {
			'ingreds': batch[0],
			'ingred_lengths': batch[1],
			'tags': batch[3],
			'ingred_missing': batch[2],
			'recipe_ids': batch[5],
			'positive_vectors': p,
			'relative_vectors': r,
			'negative_vectors': n,
			'anchor_vectors': x,
			'ingred_relevant': batch[-1],
			'recipe_vectors2': batch[-2]}