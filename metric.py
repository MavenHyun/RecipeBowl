import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import pandas as pd
import numpy as np
from data_utils import *
from env_config import *
from prettytable import PrettyTable
from tqdm import tqdm
from sklearn.metrics.pairwise import paired_distances as pdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import ndcg_score
from scipy.stats import rankdata
import math
# from torch.utils.tensorboard import SummaryWriter
import wandb 
import pickle 
import faiss

class MetricHelper:
	def __init__(self, filepath, ingtable, num_rel=20):
		# model_type: last, best_tloss, best_vloss
		self.filepath = filepath
		self.food_indices = sorted(list(pickle.load(open(f'{ROOT_PATH}iids_full.pkl', 'rb'))))
		self.food_indices.remove(1)

		self.widx2fidx = {wdx: fdx for fdx, wdx in enumerate(self.food_indices)}

		self.food_vectors = ingtable[self.food_indices]
		self.idx2food = pickle.load(open(f'{ROOT_PATH}iid2ingr_full.pkl', 'rb'))
		self.food_names = [self.idx2food[i] for i in self.food_indices]

		self.recipe_vectors = pickle.load(open(f'{ROOT_PATH}rid2vec.pkl', 'rb')) # 600 dim, recipe vectors are fixed
		self.recipe_vectors = pd.DataFrame(self.recipe_vectors).transpose()
		self.idx2recipe = list(self.recipe_vectors.index.values)
		self.idx2recipe = {i: self.idx2recipe[i] for i in range(len(self.idx2recipe))}
		self.recipe_vectors = self.recipe_vectors.values.astype(np.float32)

		self.nrp = 1# we will not use neighbor ingredients
		print("Number of Neighbors for Ranking", self.nrp)

		self.loss_dict = {
			'train/loss': [], 'train/mloss': [], 'train/rloss': [], 
			'valid/loss': [], 'valid/mloss': [], 'valid/rloss': [],
			'test/loss': [], 'test/mloss': [], 'test/rloss': []}

		self.numpy_dict = {
			'train/mtrue': [], 'train/mpred': [], 'train/scores': [],
			'train/rtrue': [], 'train/rpred': [], 'train/foods': [],
			'train/recipeids': [], 'train/recipes': [], 'train/foods_related': [], 
			'valid/mtrue': [], 'valid/mpred': [], 'valid/scores': [],
			'valid/rtrue': [], 'valid/rpred': [], 'valid/foods': [],
			'valid/recipeids': [], 'valid/recipes': [], 'valid/foods_related': [], 
			'test/mtrue': [], 'test/mpred': [], 'test/scores': [],
			'test/rtrue': [], 'test/rpred': [], 'test/foods': [],
			'test/recipeids': [], 'test/recipes': [], 'test/foods_related': []}

		self.wandb_dict = dict()

		# for saving best models
		self.best_loss = {'train/loss': 999.999, 'valid/loss': 999.999}

	def update_modelinfo(self, model, epoch):
		for n, p in model.named_parameters():
			try: wandb.log({f'model/params/{n}': wandb.Histogram(numpify(p))})
			except: pass

	def update_foodvectors(self, model):
		table = model.ingredients2latent.v_ingbedding
		all_indices = torch.cuda.LongTensor([i for i in range(table[0].weight.size(0))])
		self.food_vectors = numpify(table(all_indices))[self.food_indices]

	def update_scores(self, model, phase='train', epoch=0, save_csv=False, save_numpy=False, food_table=None, model_type='last'):
		if food_table: self.food_vectors = numpify(food_table.weight)[self.food_indices]

		for n, p in model.named_parameters():
			try: self.wandb_dict[f'model/params/{n}'] = wandb.Histogram(numpify(p))
			except: pass
			if (p.requires_grad) and ("bias" not in n) and p.grad is not None:
				try: self.wandb_dict[f'model/grads/{n}'] = wandb.Histogram(numpify(p.grad))
				except: pass

		self.numpy_dict[f'{phase}/mpred'] = np.vstack(self.numpy_dict[f'{phase}/mpred']).astype(np.float32)
		self.numpy_dict[f'{phase}/mtrue'] = np.vstack(self.numpy_dict[f'{phase}/mtrue']).astype(np.float32)
		self.numpy_dict[f'{phase}/rpred'] = np.vstack(self.numpy_dict[f'{phase}/rpred']).astype(np.float32)
		self.numpy_dict[f'{phase}/rtrue'] = np.vstack(self.numpy_dict[f'{phase}/rtrue']).astype(np.float32)
		self.numpy_dict[f'{phase}/foods'] = np.hstack(self.numpy_dict[f'{phase}/foods'])
		self.numpy_dict[f'{phase}/recipes'] = np.vstack(self.numpy_dict[f'{phase}/recipes'])
		self.numpy_dict[f'{phase}/foods_related'] = np.vstack(self.numpy_dict[f'{phase}/foods_related'])
		num_rows = self.numpy_dict[f'{phase}/mpred'].shape[0]

		print("0] Calculating mean cosine similarity and euclidean distance...")
		self.wandb_dict[f'{phase}/recipe/mcs'] = 1. - pdist(self.numpy_dict[f'{phase}/rpred'], 
															self.numpy_dict[f'{phase}/rtrue'], 
															'cosine').mean()
		self.wandb_dict[f'{phase}/recipe/med'] = pdist(self.numpy_dict[f'{phase}/rpred'],
													   self.numpy_dict[f'{phase}/rtrue'],
													   'euclidean').mean()
		self.wandb_dict[f'{phase}/ingred/mcs'] = 1. - pdist(self.numpy_dict[f'{phase}/mpred'], 
															self.numpy_dict[f'{phase}/mtrue'], 
															'cosine').mean()
		self.wandb_dict[f'{phase}/ingred/med'] = pdist(self.numpy_dict[f'{phase}/mpred'], 
													   self.numpy_dict[f'{phase}/mtrue'],
													   'euclidean').mean()

		print("1] Converting and transforming the 2D arrays...")
		def convert_indices(x):
			return self.widx2fidx[x]
		
		def get_predicted_names(x):
			x = self.food_indices[x]
			return self.idx2food[x]

		def get_predicted_names2(x):
			return self.idx2recipe[x]

		def get_missing_names(x):
			return self.idx2food[x]

		def get_incomplete_names(x):
			return np.asarray((' & ').join([self.idx2food[i] for i in x.tolist() if i > 1]),dtype=object)

		temp_numpy = self.numpy_dict[f'{phase}/foods'].reshape(-1,1)
		# temp_numpy = np.hstack([self.numpy_dict[f'{phase}/foods'].reshape(-1,1), self.numpy_dict[f'{phase}/foods_related']])
		missing_foods = np.vectorize(convert_indices)(temp_numpy)
		# missing_col = np.apply_along_axis(get_incomplete_names, 1, temp_numpy).reshape(-1,1)
		missing_col = np.apply_along_axis(get_incomplete_names, 1, temp_numpy).reshape(-1,1)
		incomplete_col = np.apply_along_axis(get_incomplete_names, 1, self.numpy_dict[f'{phase}/recipes']).reshape(-1,1)
		recipe_col = np.array(self.numpy_dict[f'{phase}/recipeids']).astype(str).reshape(-1,1)

		def get_reciprocal_rank(z, *args):
			P = args[0]
			X = z[:P]
			Y = z[self.nrp:]
			rr_list = [0.]
			for x in X.tolist():
				try: rr_list.append(1 / (np.nonzero(Y==int(x))[0][0] + 1))
				except Exception as e: rr_list.append(1 / len(self.food_indices))

			return max(rr_list)

		def get_recall(x, *args):
			K = args[0][0]
			P = args[0][1]
			target = x[:P].tolist()
			candid = x[self.nrp:K+self.nrp].tolist()

			return len(set(target) & set(candid)) / P

		def get_precision(x, *args):
			K = args[0][0]
			P = args[0][1]
			target = x[:P].tolist()
			candid = x[self.nrp:K+self.nrp].tolist()

			return len(set(target) & set(candid)) / K

		def get_average_precision(x, *args):
			K = args[0][0]
			P = args[0][1]
			if K < P: return -1
			target = x[:P].tolist()
			candid = x[self.nrp:K+self.nrp].tolist()
			AP, i = 0., 1
			for n, c in enumerate(candid):
				if True in (t==c for t in target):
					AP += i / (n+1)
					i += 1

			return AP / len(target)

		def get_hit(x, *args):
			K = args[0][0]
			P = args[0][1]
			target = x[:P].tolist()
			candid = x[self.nrp:K+self.nrp].tolist()
			return 1 if True in (t in candid for t in target) else 0

		# Added Recipe Ranking Results
		metric_dict = {'cosine': faiss.METRIC_INNER_PRODUCT, 'euclidean': faiss.METRIC_L2}
		# for score in ['scores', 'cosine', 'euclidean']:
		for score in ['cosine']:
			print(f"2] Retrieving ranking results based on {score}...")
			X = self.numpy_dict[f'{phase}/mpred'].copy()
			Y = self.food_vectors.copy()
			R = self.numpy_dict[f'{phase}/rpred'].copy()
			S = self.recipe_vectors.copy()
			if 'scores' not in score:
				if score == 'cosine':
					# Too redundant...?
					idx = faiss.index_factory(self.food_vectors.shape[1],"Flat",faiss.METRIC_INNER_PRODUCT)
					faiss.normalize_L2(X)
					faiss.normalize_L2(Y)
					idx.add(Y)
					score_matrix, score_ranked = idx.search(X, 100)
					idx = faiss.index_factory(600, "Flat", faiss.METRIC_INNER_PRODUCT)
					faiss.normalize_L2(R)
					faiss.normalize_L2(S)
					idx.add(S)
					recipe_matrix, recipe_ranked = idx.search(R, 100)
				elif score == 'euclidean':
					idx = faiss.index_factory(self.food_vectors.shape[1],"Flat",faiss.METRIC_L2)
					idx.add(Y)
					score_matrix, score_ranked = idx.search(X, Y.shape[0])
					idx = faiss.index_factory(600, "Flat", faiss.METRIC_L2)
					idx.add(S)
					recipe_matrix, recipe_ranked = idx.search(R, 100)
				else:
					raise
			else:
				try:
					score_matrix = np.vstack(self.numpy_dict[f'{phase}/scores'])
					score_matrix = score_matrix[:,self.food_indices] # d x 3280 (word2vec)
					score_ranked = np.argsort(score_matrix, axis=1)
					idx = faiss.index_factory(600, "Flat", faiss.METRIC_INNER_PRODUCT)
					faiss.normalize_L2(R)
					faiss.normalize_L2(S)
					idx.add(S)
					recipe_matrix, recipe_ranked = idx.search(R, Y.shape[0])
				except:
					continue

			assert score_matrix.shape[1] < 4000
			predicted_col = np.vectorize(get_predicted_names)(score_ranked)
			predicted_Rcol = np.vectorize(get_predicted_names2)(recipe_ranked)
			XY = np.hstack([missing_foods,score_ranked])
			print(XY.shape)
			# assert XY.shape[1] == 101 + self.nrp
			rr_col = None
			hitratio_cols, hitratio_Rcols = [], []
			for p in [1]:
				print(f"3] Calculating Mean Reciprocal Rank Get{p}...")
				mrr = np.apply_along_axis(get_reciprocal_rank, 1, XY, (p)).mean()
				if p == 1: rr_col = np.apply_along_axis(get_reciprocal_rank, 1, XY, (p)).reshape(-1,1)
				self.wandb_dict[f'{phase}/ingred/{score}/mrr/get{p}'] = mrr
				for k in [1, 5, 10, 20]:
					print(f"4-1] Calculating Hit Ratio Get{p}@Top{k}...")
					hit_col = np.apply_along_axis(get_hit, 1, XY, (k,p)).reshape(-1,1)
					self.wandb_dict[f'{phase}/ingred/{score}/hitratio/get{p}/top{k}'] = hit_col.mean()
					if p == 1: hitratio_cols.append(hit_col)

					print(f"4-2] Calculating Mean Recall Get{p}@Top{k}...")
					mrc = np.apply_along_axis(get_recall, 1, XY, (k,p)).reshape(-1,1).mean()
					self.wandb_dict[f'{phase}/ingred/{score}/mrc/get{p}/top{k}'] = mrc

					print(f"4-3] Calculating Mean Precision Get{p}@Top{k}...")
					mpr = np.apply_along_axis(get_precision, 1, XY, (k,p)).reshape(-1,1).mean()
					self.wandb_dict[f'{phase}/ingred/{score}/mpr/get{p}/top{k}'] = mpr

					print(f"4-4] Calculating Mean Average Precision Get{p}@Top{k}...")
					map_ = np.apply_along_axis(get_average_precision, 1, XY, (k,p)).reshape(-1,1).mean()
					self.wandb_dict[f'{phase}/ingred/{score}/map/get{p}/top{k}'] = map_

				print(f"4-5] Calculating Recipe Hit Ratio Get1@Top{k}...")
				hit_Rcol = np.apply_along_axis(get_hit, 1, 
				np.hstack([recipe_col,predicted_Rcol]), (k,1)).reshape(-1,1)
				hr = hit_Rcol.mean()
				self.wandb_dict[f'{phase}/recipe/{score}/hitratio/top{k}'] = hr
				hitratio_Rcols.append(hit_Rcol)

			hitratio_cols = np.hstack(hitratio_cols)
			hitratio_Rcols = np.hstack(hitratio_Rcols)
			sub_filepath = self.filepath + f'/{model_type}'

			if phase == 'train':
				print(self.wandb_dict[f'{phase}/ingred/{score}/mrr/get1'])

			if save_csv:
				print(f"5] Saving Ranking Results to {sub_filepath}...")
				whole_table = np.hstack([recipe_col, incomplete_col, missing_col, rr_col, hitratio_cols, predicted_col[:,:10]])
				mrr = self.wandb_dict[f'{phase}/ingred/{score}/mrr/get1']
				header1 = ['Recipe ID', 'Incomplete Ingredient Set', f'Missing Ingredient (MRR: {mrr:.3f})', 'Reciprocal Rank']
				header2 = []
				for k in [1, 5, 10, 20]:
					hr = self.wandb_dict[f'{phase}/ingred/{score}/hitratio/get1/top{k}']
					mrc = self.wandb_dict[f'{phase}/ingred/{score}/mrc/get1/top{k}']
					mpr = self.wandb_dict[f'{phase}/ingred/{score}/mpr/get1/top{k}']
					map_ = self.wandb_dict[f'{phase}/ingred/{score}/map/get1/top{k}']
					header2.append(f'HIT@Top{k} | Hit Ratio: {hr:.3f} | MAP: {map_:.3f} | Mean Recall: {mrc:.3f} | Mean Precision: {mpr:.3f}')
				header3 = [f'Predicted #{i}' for i in range(10)]
				header = header1 + header2 + header3
				df = pd.DataFrame(whole_table, columns=header)
				os.makedirs(sub_filepath, exist_ok=True)
				df.to_csv(sub_filepath + f'/ranking_results_{phase}_{score}.csv')
				np.save(sub_filepath + f'/score_matrix_{phase}_{score}.npy', score_matrix)

		if save_numpy:
			print(f"6] Saving Numpy Vectors to {sub_filepath}...")
			os.makedirs(sub_filepath, exist_ok=True)
			with open(sub_filepath + f'/numpy_vectors_{phase}.pkl', 'wb') as fw:
				pickle.dump(
					{'pred': self.numpy_dict[f'{phase}/mpred'], 
					 'true': self.numpy_dict[f'{phase}/mtrue'], 
					 'index': self.numpy_dict[f'{phase}/foods'], 
					 'recipe_true': self.numpy_dict[f'{phase}/rtrue'],
					 'recipe_pred': self.numpy_dict[f'{phase}/rpred'],
					 'recipe_ids': self.numpy_dict[f'{phase}/recipeids']}, fw)

		wandb.log(self.wandb_dict, step=epoch) 
		return self.wandb_dict

	def reset_everything(self):
		for key in self.numpy_dict.keys():
			self.numpy_dict[key] = []
		for key in self.loss_dict.keys():
			self.loss_dict[key] = []
		self.wandb_dict = dict()
	
	def store_lists(self, list_, key):
		self.numpy_dict[key] = self.numpy_dict[key] + list(list_)

	def store_numpy(self, numpy, key):
		if numpy is not None: self.numpy_dict[key].append(numpy)

	def store_loss(self, loss, key):
		self.loss_dict[key].append(loss)

	def update_average_loss(self, epoch):
		loss_dict = dict()
		for key in self.loss_dict.keys():
			if len(self.loss_dict[key]) > 0:
				avg = sum(self.loss_dict[key]) / len(self.loss_dict[key])
				loss_dict[key] = avg
				self.wandb_dict[f'{key}_avg'] = avg

		return loss_dict

	def store_embedding_vectors(self):
		return

	def store_losses(self, batch, phase):
		self.store_loss(batch['total_loss'].item(), f'{phase}/loss')
		self.store_loss(batch['missing_loss'], f'{phase}/mloss')
		self.store_loss(batch['recipe_loss'], f'{phase}/rloss')

	def store_everything(self, batch, phase):
		# step = epoch*len(train)+idx
		self.store_loss(batch['total_loss'].item(), f'{phase}/loss')
		self.store_loss(batch['missing_loss'], f'{phase}/mloss')
		self.store_loss(batch['recipe_loss'], f'{phase}/rloss')
		self.store_numpy(numpify(batch['ingred_missing'].view(-1)), f'{phase}/foods')
		self.store_numpy(numpify(batch['ingreds']), f'{phase}/recipes')
		self.store_numpy(numpify(batch['anchor_vectors']), f'{phase}/mpred')
		self.store_numpy(numpify(batch['anchor_probs']), f'{phase}/scores')
		self.store_numpy(numpify(batch['positive_vectors']), f'{phase}/mtrue')
		self.store_numpy(numpify(batch['recipe_vectors1']), f'{phase}/rpred')
		self.store_numpy(numpify(batch['recipe_vectors2']), f'{phase}/rtrue')
		self.store_lists(batch['recipe_ids'], f'{phase}/recipeids')
		# under development
		self.store_numpy(numpify(batch['ingred_relevant']), f'{phase}/foods_related')


def numpify(x):
	try: return x.detach().cpu().numpy()
	except: None