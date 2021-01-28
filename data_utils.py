from __future__ import print_function
import torch.utils.data as data
from torch.utils.data import random_split as split
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
import random
from numpy.random import choice 
import json
from env_config import *
import pickle
from datetime import datetime

class _RepeatSampler(object):
	def __init__(self, sampler):
		self.sampler = sampler

	def __iter__(self):
		while True:
			yield from iter(self.sampler)

class IMRecipeLoader(data.dataloader.DataLoader):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
		self.iterator = super().__iter__()

	def __len__(self):
		return len(self.batch_sampler.sampler)

	def __iter__(self):
		for i in range(len(self)):
			yield next(self.iterator)

class IMRecipeDataset(data.Dataset):
	def __init__(self, args=0):
		self.dataset = pickle.load(open(f'{ROOT_PATH}{args.train_dev_test}_ingrs_{args.dataset}.pkl','rb'))
		# self.iid2triplet = pickle.load(open(f'{ROOT_PATH}iid2triplet_full.pkl', 'rb'))
		# self.iid_list = sorted(list(pickle.load(open(f'{ROOT_PATH}iids_full.pkl', 'rb'))))
		# self.iid2fgvec = pickle.load(open(f'{ROOT_PATH}iid2fgvec_full.pkl', 'rb'))
		# self.iid2ingred = pickle.load(open(f'{ROOT_PATH}iid2ingr_full.pkl', 'rb'))
		# self.rid2iids = pickle.load(open(f'{ROOT_PATH}rid2iids_full.pkl', 'rb'))
		# self.rid2tfidfs = pickle.load(open(f'{ROOT_PATH}rid2tfidf_full.pkl', 'rb'))
		self.rid2info = pickle.load(open(f'{ROOT_PATH}rid2info_full.pkl', 'rb'))
		self.rid2sortediids = pickle.load(open(f'{ROOT_PATH}rid2sorted_iids_full.pkl', 'rb'))
		self.iid2sortedsims = pickle.load(open(f'{ROOT_PATH}iid2sim_sorted_top100_full.pkl', 'rb'))
		del self.iid2sortedsims[1]
		self.idx2recipeid = sorted(self.dataset.keys())

		self.num_targets = args.num_targets
		self.num_neighbors = args.num_neighbors
		self.total_selection = False
		self.random_draw = False

		if self.total_selection:
			raise
			# test_dataset = []
			# for key in tqdm(self.dataset.keys()):
			# 	ingrs_list = self.dataset[key]['ingrs'].tolist()
			# 	ingrs_list = [i for i in ingrs_list if i > 1]
			# 	for missing in ingrs_list:
			# 		temp_dict = {
			# 				'ingrs': np.array(sorted([int(i) if i != missing else 0 for i in self.dataset[key]['ingrs'].tolist()], reverse=True)),
			# 				'missing': int(missing), 'rid': key, 'tags': self.dataset[key]['tags']}
			# 		test_dataset.append(temp_dict)
			# self.dataset = test_dataset
		else:
			new_dataset = []
			for rid in tqdm(self.dataset.keys()):
				ingrs_list = self.dataset[rid]['ingrs'].tolist()
				ingrs_list = [i if i > 1 else 0 for i in ingrs_list]
				p_targets = self.rid2sortediids[rid][:self.num_targets]
				s_targets = self.rid2sortediids[rid][self.num_targets:]
				for p in p_targets:
					ingrs = sorted([int(i) if int(i) != int(p) else 0 for i in ingrs_list], reverse=True)
					new_dataset.append({
						'ingrs': np.array(ingrs),
						'target': int(p),
						'rid': rid,
						'tags': self.dataset[rid]['tags']})
				if len(s_targets) > 0 and self.random_draw:
					new_dataset.append({
						'ingrs': ingrs_list,
						'target': s_targets,
						'rid': rid,
						'tags': self.dataset[rid]['tags']})
			self.dataset = new_dataset

				
	def __getitem__(self, index):
		np.random.seed(int(datetime.now().timestamp()))
		rid = self.dataset[index]['rid']
		# Secondary Targets
		if isinstance(self.dataset[index]['target'], list):
			iid = int(choice(self.dataset[index]['target']))
			ingrs_list = self.dataset[index]['ingrs'][:50]
			ingrs = np.array(sorted([int(i) if int(i) != int(iid) else 0 for i in ingrs_list], reverse=True))
			num_ingrs = (ingrs != 0).sum()
			ingrs = torch.cuda.LongTensor(ingrs)
			target = torch.cuda.LongTensor([iid])
			tags = torch.cuda.FloatTensor(self.dataset[index]['tags'])
		# Primary Targets
		else:
			iid = self.dataset[index]['target']
			ingrs = torch.cuda.LongTensor(self.dataset[index]['ingrs'][:50])
			target = torch.cuda.LongTensor([iid])
			num_ingrs = (self.dataset[index]['ingrs'] != 0).sum()
			tags = torch.cuda.FloatTensor(self.dataset[index]['tags'])

		assert iid > 1
		assert 1 not in self.dataset[index]['ingrs']
		assert ingrs[0] != 0
		assert ingrs[-1] == 0

		nei_candidates = [i for i in self.iid2sortedsims[iid] if i > 1]
		neis = torch.cuda.LongTensor(nei_candidates[:self.num_neighbors])
		rvec = torch.cuda.FloatTensor(self.rid2info[rid]['vec_reciptor']) 

		return [ingrs, num_ingrs, target, tags, iid, rid, rvec, neis]

	def __len__(self):
		return len(self.dataset)