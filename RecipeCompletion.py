import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import numpy as np
import EncoderModules as em
import DecoderModules as dm
from data_utils import *
from env_config import *
from prettytable import PrettyTable
from tqdm import tqdm
from sklearn.metrics.pairwise import paired_distances, paired_euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import ndcg_score
from scipy.stats import rankdata
import math
# from torch.utils.tensorboard import SummaryWriter
import wandb 
import pickle 
import adabound
import faiss
from metric import MetricHelper
import timeit

class RecipeCompletion(nn.Module):
	def __init__(self, args):
		super(RecipeCompletion, self).__init__()
		self.project_name = args.project_name
		self.config = args # wandb
		self.num_epochs = args.num_epochs
		self.lr = args.learning_rate
		self.wd = args.weight_decay 
		self.dr = args.dropout_rate
		self.batch_size = args.batch_size

		self.iid_list = sorted(list(pickle.load(open(f'{ROOT_PATH}iids_full.pkl', 'rb'))))
		self.iid_list.remove(1)

		args.ing_indices = self.iid_list
		self.ingredients2latent = get_encoder_model(args)
		# self.recipe = get_recipe_model(args)
		args.vocab_size = 30167
		self.latent2prediction = get_decoder_model(args)

		self.checkpoint = args.load_checkpoint
		self.filepath = args.saved_path + args.session_name
		os.makedirs(self.filepath, exist_ok=True)
		display_hyper_parameters(args, self.filepath)

		self.config.ing_indices = None
		mh_args = (self.filepath, self.ingredients2latent.vocab_space, args.num_neighbors)
		self.helper = MetricHelper(*mh_args)
		if args.ingred_decoder == 'none': self.num_epochs = 0
		args.ing_indices = None

		self.best_tloss, self.best_vloss = 999.999, 999.999
		self.patience = 50

	def check_improvement(self, model, loss_dict):
		vloss, tloss = loss_dict['valid/loss'], loss_dict['train/loss']
		if (vloss - self.best_vloss > 0.0001) or (vloss > tloss):
			self.patience -= 1
		if loss_dict['train/loss'] < self.best_tloss:
			self.best_tloss = loss_dict['train/loss']
		if loss_dict['valid/loss'] < self.best_vloss:
			self.best_vloss = loss_dict['valid/loss']

		return False if self.patience >= 0 else True

	def forward(self, batch):
		batch = self.ingredients2latent(batch)
		batch = self.latent2prediction(batch)

		return batch

	def predict(self, ingredients, tags=None):
		self.helper.update_foodvectors(self)
		tags_dict = pickle.load(open('data/tags.pkl', 'rb'))

		print("Checking Ingredients...")
		ing_vector = list()
		for food in ingredients:
			try: ing_vector.append(self.helper.food2idx[food])
			except:
				print("The following ingredient is invalid!", food)
		ingrs = torch.cuda.LongTensor(ing_vector).view(1,-1)
		num_ingrs = len(ingredients)
		target = torch.cuda.LongTensor([1]).view(1,-1)

		print("Checking Recipe-related Tags...")
		tag_vector = np.zeros(630)
		for tag in tags:
			try: tag_vector[tags_dict[tag]] = 1.0
			except: 
				print("The following recipe tag is invalid!", tag)


		tags = torch.cuda.FloatTensor(np.zeros(630)).view(1,-1)
		iid = None
		rid = None
		rvec = torch.cuda.FloatTensor(np.zeros(600)).view(1,-1)
		neis = torch.cuda.LongTensor(np.ones(10)).view(1,-1)
		# batch => [ingrs, num_ingrs, target, tags, iid, rid, rvec, neis]
		batch = self.forward([ingrs, num_ingrs, target, tags, iid, rid, rvec, neis])
		
		recommended_ingred = batch['anchor_vectors'].detach().cpu().numpy().astype(np.float32)
		recommended_recipe = batch['recipe_vectors1'].detach().cpu().numpy().astype(np.float32)

		X = recommended_ingred
		Y = self.helper.food_vectors.copy()
		R = recommended_recipe
		S = self.helper.recipe_vectors.copy()
		idx = faiss.index_factory(self.helper.food_vectors.shape[1],"Flat",faiss.METRIC_INNER_PRODUCT)
		faiss.normalize_L2(X)
		faiss.normalize_L2(Y)
		idx.add(Y)
		score_matrix, score_ranked = idx.search(X, 100)
		idx = faiss.index_factory(600, "Flat", faiss.METRIC_INNER_PRODUCT)
		faiss.normalize_L2(R)
		faiss.normalize_L2(S)
		idx.add(S)
		recipe_matrix, recipe_ranked = idx.search(R, 100)

		def get_predicted_names(x):
			x = self.helper.food_indices[x]
			return self.helper.idx2food[x]

		def get_predicted_names2(x):
			return self.helper.idx2recipe[x]

		predicted_col = np.vectorize(get_predicted_names)(score_ranked)[0,:5].tolist()
		predicted_Rcol = np.vectorize(get_predicted_names2)(recipe_ranked)[0,:5].tolist()
		rid2title = pickle.load(open('data/rid2title.pkl', 'rb'))
		predicted_Rcol = [rid2title[rid] for rid in predicted_Rcol]

		return predicted_col, predicted_Rcol

	def fit(self, train, valid=None):
		# batch => [instrs, num_instrs, ingrs, num_ingrs, tags, missing, rec_id]
		train_iter, valid_iter, early_stopping = 0, 0, False
		model = self.cuda()
		model.apply(weights_init)
		total_num = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
		print('Total number of trainable parameters: ', total_num)
		opti = adabound.AdaBound(model.parameters(), lr=self.lr, final_lr=0.5, weight_decay=self.wd)
		sche = torch.optim.lr_scheduler.ReduceLROnPlateau(opti, factor=0.7, verbose=True)
		if self.num_epochs < 1: return model 
		for epoch in range(self.num_epochs):
			pbar = tqdm(train)
			for idx, batch in enumerate(pbar):
				model.train()
				opti.zero_grad()
				batch = model(batch)
				batch['total_loss'].backward()
				opti.step()
				if idx == train_iter % len(train):
					self.helper.store_everything(batch, 'train')
				else:
					self.helper.store_losses(batch, 'train')
				del batch 
				torch.cuda.empty_cache()
			self.helper.update_foodvectors(model)
			if not valid: continue
			for idx, batch in enumerate(valid):
				if idx == valid_iter % len(valid):
					model.eval()
					batch = model(batch)
					self.helper.store_everything(batch, 'valid')
					sche.step(batch['total_loss'])
					del batch 
					torch.cuda.empty_cache()
					break
			valid_iter += 1
			train_iter += 1
			loss_dict = self.helper.update_average_loss(epoch+1)
			early_stopping = self.check_improvement(model, loss_dict)
			print(f"epoch: {epoch+1}, train loss: {loss_dict['train/loss']:.5f}, valid loss:{loss_dict['valid/loss']:.5f}")
			self.helper.update_scores(model, 'train', epoch+1)
			self.helper.update_scores(model, 'valid', epoch+1)
			self.helper.reset_everything()
			if early_stopping: break
		
		return model

	def predict_evaluate(self, data, suffix='test', model_type='last'):
		model = self.cuda()
		model.eval()
		for batch in tqdm(data):
			batch = model(batch)
			self.helper.store_everything(batch, suffix)
			del batch 
			torch.cuda.empty_cache()
		self.helper.update_foodvectors(model)

		loss_dict = self.helper.update_average_loss(self.num_epochs+1)
		score_dict = self.helper.update_scores(model, suffix, self.num_epochs+1, 
											save_csv=True, save_numpy=True, model_type=model_type)

		return [loss_dict, score_dict]

def get_encoder_model(args):
	if args.ingred_encoder == 'settf'                : return em.SetTransformer(args)
	elif args.ingred_encoder == 'bilstm'             : return em.biLSTM(args)
	elif args.ingred_encoder == 'deepset'            : return em.DeepSet(args)
	elif args.ingred_encoder == 'sum'                : return em.SimpleSum(args)
	else                                            : raise Exception

def get_decoder_model(args):
	if args.ingred_decoder == 'prob'                : return dm.ProbabilityLearner(args)
	elif args.ingred_decoder == 'none'              : return dm.BaseModule(args)
	else                                            : raise Exception

def display_hyper_parameters(args, filepath):
	x = PrettyTable()
	x.field_names = ["Hyperparameter", args.session_name]
	x.add_row(["Session Name", args.session_name])
	x.add_row(["IMRecipe Dataset", args.dataset])
	x.add_row(["Ingredient(s) Encoder Model", args.ingred_encoder])
	x.add_row(["Ingredient(s) Embedding Vectors", args.ingred_embed])
	x.add_row(["Ingredient(s) Fine Tuning?", args.fine_tuning])
	x.add_row(["Ingredient Decoder Model", args.ingred_decoder])
	x.add_row(["Ingredient Loss Function", args.ingred_loss])
	x.add_row(["Recipe Loss Function", args.recipe_loss])
	x.add_row(["Recipe Loss Coefficient", args.recipe_coef])
	x.add_row(["Add Tag Binary Vectors", args.add_tags])
	x.add_row(["Hidden Dimension Size", args.hidden_nodes])
	x.add_row(["Learning Rate", args.learning_rate])
	x.add_row(["Weight Decay", args.weight_decay])
	x.add_row(["Dropout Rate", args.dropout_rate])
	x.add_row(["Number of Epochs", args.num_epochs])
	x.add_row(["Batch Size", args.batch_size])
	x.add_row(["Number of Targets", args.num_targets])
	x.add_row(["Number of Neighbors", args.num_neighbors])
	print(x.get_string(title=args.session_name))

	with open(filepath + '/hyperparameters.table', 'w') as f:
		f.write(x.get_string(title=args.session_name))
	with open(filepath + '/hyperparameters.pkl', 'wb') as f:
		pickle.dump(args, f) 

def weights_init(m):
	try:
		if not isinstance(m, nn.Embedding):
			nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
	except:
		pass