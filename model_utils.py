import torch
import torch.nn as nn
from torch.nn.functional import normalize as l2
import math
import torch.nn.functional as F

'''
	##############################
	#                            #
	#    Custom Loss Functions   #
	#                            #
	##############################

'''

class NllLossCosine(nn.Module):
	def __init__(self, tau=1.0):
		super(NllLossCosine, self).__init__()
		self.tau = tau

	def forward(self, anchor, positive, negatives):
		sim = nn.CosineSimilarity(dim=2, eps=1e-6)
		logsoftmax = nn.LogSoftmax(1)
		exp_size = negatives.size(1)
		anchor_expanded = anchor.unsqueeze(1).repeat(1, exp_size+1, 1)
		target_expanded = torch.cat([positive.unsqueeze(1), negatives], 1)
		scores = sim(anchor_expanded, target_expanded) / self.tau
		scores = -1.0 * logsoftmax(scores)[:, 0]

		return scores.mean()

class NllLossEuclidean(nn.Module):
	def __init__(self, tau=1.0):
		super(NllLossEuclidean, self).__init__()
		self.tau = tau

	def forward(self, anchor, positive, negatives):
		logsoftmax = nn.LogSoftmax(1)
		exp_size = negatives.size(1)
		anchor_expanded = anchor.unsqueeze(1).repeat(1, exp_size+1, 1)
		target_expanded = torch.cat([positive.unsqueeze(1), negatives], 1)
		scores = -1.0 * (anchor_expanded - target_expanded).abs().pow(2).sum(2).sqrt() / self.tau
		scores = -1.0 * logsoftmax(scores)[:, 0]

		return scores.mean()
	
class SnnLossCosine(nn.Module):
	def __init__(self, tau=1.0):
		super(SnnLossCosine, self).__init__()
		self.tau = tau

	def forward(self, anchor, positive, relatives, negatives):
		sim = nn.CosineSimilarity(dim=2, eps=1e-6)
		exp_pos = relatives.size(1)
		exp_neg = negatives.size(1)
		anchor_expanded = anchor.unsqueeze(1).repeat(1, exp_pos+1, 1)
		friend_expanded = torch.cat([positive.unsqueeze(1), relatives], 1)
		pos_scores = (sim(anchor_expanded, friend_expanded) / self.tau).exp().sum(1)
		anchor_expanded = anchor.unsqueeze(1).repeat(1, exp_pos+exp_neg+1, 1)
		target_expanded = torch.cat([positive.unsqueeze(1), relatives, negatives], 1)
		neg_scores = (sim(anchor_expanded, target_expanded) / self.tau).exp().sum(1)
		scores = -1.0 * (pos_scores / neg_scores).log()

		return scores.mean()

class SnnLossEuclidean(nn.Module):
	def __init__(self, tau=1.0):
		super(SnnLossEuclidean, self).__init__()
		self.tau = tau

	def forward(self, anchor, positive, relatives, negatives):
		exp_pos = relatives.size(1)
		exp_neg = negatives.size(1)
		anchor_expanded = anchor.unsqueeze(1).repeat(1, exp_pos+1, 1)
		friend_expanded = torch.cat([positive.unsqueeze(1), relatives], 1)
		pos_scores = (-1.0 * (anchor_expanded - friend_expanded).abs().pow(2).sum(2).sqrt() / self.tau).exp().sum(1)
		anchor_expanded = anchor.unsqueeze(1).repeat(1, exp_pos+exp_neg+1, 1)
		target_expanded = torch.cat([positive.unsqueeze(1), relatives, negatives], 1)
		neg_scores = (-1.0 * (anchor_expanded - target_expanded).abs().pow(2).sum(2).sqrt() / self.tau).exp().sum(1)
		scores = -1.0 * (pos_scores / neg_scores).log()

		return scores.mean()

def masking_tensor(input_dim, input_lengths, max_length):
	input_lengths = input_lengths.cpu().tolist() 
	mask_tensor = torch.zeros((len(input_lengths), max_length, input_dim)).cuda()
	for idx, input_length in enumerate(input_lengths):
		mask_tensor[idx, :input_length, :] = 1.
				
	return mask_tensor


''''
	#############################################
	#                                           #
	#    Building Blocks for Set Transformer    #
	#                                           #
	#############################################

	Following implementations are based on Juho Lee's Set Transformer
	Citated from paper: An Effective Pretrained Model for Recipe Representation Learning, Li et al., 2020
	Link to citation: http://proceedings.mlr.press/v97/lee19d/lee19d.pdf
	Link to github: https://github.com/juho-lee/set_transformer
'''

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)