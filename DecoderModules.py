import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from EncoderModules import *
from data_utils import *
from env_config import *
from model_utils import *

# All the predictions will be stored in dictionaries for convenience

class BaseModule(nn.Module):
	def __init__(self, args=0):
		super(BaseModule, self).__init__()
		self.add_tags = args.add_tags
		self.tag_dim = 630 if args.add_tags else 0
		self.recipe_model = args.recipe_decoder

		# Build model architecture
		self.lat2ingred = nn.Sequential(
			nn.Linear(args.hidden_nodes+self.tag_dim, 300),
			nn.ReLU(),
			nn.LayerNorm(300),
			nn.Linear(300, 300))

		self.lat2recipe = nn.Sequential(
			nn.Linear(args.hidden_nodes+self.tag_dim, 300),
			nn.ReLU(),
			nn.LayerNorm(300),
			nn.Linear(300, 600))

		# Get loss function keys
		self.missing_func = args.ingred_loss
		self.recipe_func = args.recipe_loss
		self.recipe_coef = args.recipe_coef
		self.temperature_coef = args.temperature_coef

	def forward(self, batch):
		# Output Computation
		batch['recipe_vectors1'] = torch.ones(batch['ingred_missing'].size()).cuda()
		batch['recipe_vectors2'] = torch.ones(batch['ingred_missing'].size()).cuda()
		batch['anchor_probs'] = None

		# Loss Computation
		batch['total_loss'] = torch.empty(1, requires_grad=True).cuda()
		batch['missing_loss'] = 0.0
		batch['recipe_loss'] = 0.0

		return batch

class ProbabilityLearner(BaseModule):
	def __init__(self, args=0):
		super(ProbabilityLearner, self).__init__(args)

	def forward(self, batch):
		# Output Computation
		x = torch.cat([batch['anchor_vectors'], batch['tags']], 1) if self.add_tags else batch['anchor_vectors']
		batch['anchor_vectors'] = self.lat2ingred(x)
		batch['anchor_probs'] = None
		batch['recipe_vectors1'] = self.lat2recipe(x)
		batch_size = x.size(0)

		# Ingred Loss Computation
		if self.missing_func == 'nll_cos':
			missing_criterion = NllLossCosine(self.temperature_coef)
			batch['missing_loss'] = missing_criterion(
												anchor=batch['anchor_vectors'],
												positive=batch['positive_vectors'],
												negatives=batch['negative_vectors'])
		elif self.missing_func == 'nll_euc':
			missing_criterion = NllLossEuclidean(self.temperature_coef)
			batch['missing_loss'] = missing_criterion(
												anchor=batch['anchor_vectors'],
												positive=batch['positive_vectors'],
												negatives=batch['negative_vectors'])
		elif self.missing_func == 'snn_cos':
			missing_criterion = SnnLossCosine(self.temperature_coef)
			batch['missing_loss'] = missing_criterion(
												anchor=batch['anchor_vectors'],
												positive=batch['positive_vectors'],
												relatives=batch['relative_vectors'],
												negatives=batch['negative_vectors'])
		elif self.missing_func == 'snn_euc':
			missing_criterion = SnnLossEuclidean(self.temperature_coef)
			batch['missing_loss'] = missing_criterion(
												anchor=batch['anchor_vectors'],
												positive=batch['positive_vectors'],
												relatives=batch['relative_vectors'],
												negatives=batch['negative_vectors'])
		else:
			raise

		# Recipe Loss Computation
		if self.recipe_func == 'mse':
			recipe_criterion = nn.MSELoss()
			batch['recipe_loss'] = recipe_criterion(
										input=batch['recipe_vectors1'], 
										target=batch['recipe_vectors2'])
		elif self.recipe_func == 'cos':
			recipe_criterion = nn.CosineSimilarity()
			batch['recipe_loss'] = 1. - recipe_criterion(
											x1=batch['recipe_vectors1'], 
											x2=batch['recipe_vectors2']).mean()	
		elif self.recipe_func == 'dis':
			recipe_criterion = nn.PairwiseDistance(p=2)
			batch['recipe_loss'] = recipe_criterion(
											x1=batch['recipe_vectors1'],
											x2=batch['recipe_vectors2']).mean()
		else:
			raise	

		batch['total_loss'] = batch['missing_loss'] + self.recipe_coef*batch['recipe_loss']
		batch['missing_loss'] = batch['missing_loss'].item()
		batch['recipe_loss'] = batch['recipe_loss'].item()
		if 'mu_vectors' in batch.keys(): batch['total_loss'] += kld_loss(batch)
		assert batch['missing_loss'] == batch['missing_loss']
		assert batch['recipe_loss'] == batch['recipe_loss']

		return batch