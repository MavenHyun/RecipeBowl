from RecipeCompletion import RecipeCompletion
from data_utils import *
from env_config import *
import argparse
from torch.utils.data import random_split
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler
from torch.utils.data import ConcatDataset
import torch.multiprocessing as multiprocessing
import setproctitle
import pandas as pd
import wandb
import os

REPORT_METRICS = [
	'test/ingred/cosine/mrr/get1',
	'test/ingred/cosine/mrc/get1/top1',
	'test/ingred/cosine/mrc/get1/top5',
	'test/ingred/cosine/mrc/get1/top10']

def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True

def load_dataloaders(args):
	args.train_dev_test = 'train'
	trainloader = IMRecipeLoader(IMRecipeDataset(args), batch_size=args.batch_size, shuffle=True)
	args.train_dev_test = 'valid'
	validloader = IMRecipeLoader(IMRecipeDataset(args), batch_size=args.batch_size)
	args.train_dev_test = 'test'
	testloader = IMRecipeLoader(IMRecipeDataset(args), batch_size=args.batch_size)
	print(f"# of training samples: {len(trainloader)*args.batch_size}")
	print(f"# of validation samples: {len(validloader)*args.batch_size}")
	print(f"# of test samples: {len(testloader)*args.batch_size}")

	return trainloader, validloader, testloader

def load_dataloaders_reshuffled(args):
	args.train_dev_test = 'train'
	trainset = load_dataset(args)
	args.train_dev_test = 'valid'
	validset = load_dataset(args)
	args.train_dev_test = 'test'
	testset = load_dataset(args)
	wholeset = ConcatDataset([trainset, validset, testset])
	# Reshuffle the indices and make new random splits
	# return trainloader, validloader, testloader

parser = argparse.ArgumentParser()

parser.add_argument('--project_name', default='HyperParameter Search', type=str)
parser.add_argument('--session_name', default='maven', type=str)
parser.add_argument('--saved_path', default='./saved/', type=str)
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--best_model', default='last', type=str)
parser.add_argument('--load_checkpoint', default=False, action='store_true')
parser.add_argument('--test_mode', default=False, action='store_true')
parser.add_argument('--quick_mode', default=False, action='store_true')
parser.add_argument('--num_process', default=1, type=int)
parser.add_argument('--dataset', default='full', type=str)
parser.add_argument('--dataset_split', default='preset', type=str) 
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--num_targets', default=1, type=int)
parser.add_argument('--num_neighbors', default=9, type=int)
parser.add_argument('--total_selection', default=False, action='store_true')

# Encoder Module / Decoder Module
parser.add_argument('--fine_tuning', default=False, action='store_true')
parser.add_argument('--ingred_encoder', default='settf', type=str)
parser.add_argument('--ingred_embed', default='flavorgraph', type=str)
parser.add_argument('--ingred_decoder', default='mlp', type=str )
parser.add_argument('--recipe_decoder', default='none', type=str)
parser.add_argument('--add_tags', default=False, action='store_true')

# Set Transformer Related Arguments
parser.add_argument('--settf_num_heads', default=4, type=int)
parser.add_argument('--settf_num_seeds', default=1, type=int)
parser.add_argument('--settf_num_ipoints', default=32, type=int)

# Model Training Arguments
parser.add_argument('--hidden_nodes', default=128, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--num_epochs', default=60, type=int)
parser.add_argument('--dropout_rate', default=0.0, type=float)
parser.add_argument('--recipe_coef', default=0.25, type=float)
parser.add_argument('--ingred_loss', default='nll_euc', type=str)
parser.add_argument('--recipe_loss', default='cos', type=str)
parser.add_argument('--temperature_coef', default=10.0, type=float)

args = parser.parse_args()

def do_session(args, fold=0):
	setup_seed(args.random_seed+fold)
	multiprocessing.current_process().name = f'{args.ingred_embed}-{args.ingred_encoder}-{args.ingred_decoder}-{args.ingred_loss}'
	setproctitle.setproctitle(multiprocessing.current_process().name)

	if args.dataset_split == 'preset':
		trainloader, validloader, testloader = load_dataloaders(args)
	else:
		raise Exception('Random splits will be implemented later if needed')

	args.session_name = f'{args.session_name}'
	net = RecipeCompletion(args)
	if not args.test_mode:
		net = net.fit(trainloader, validloader)
		torch.save(net.state_dict(), net.filepath + '/last.mdl')

	args = pickle.load(open(net.filepath + '/hyperparameters.pkl', 'rb'))
	net = RecipeCompletion(args)
	net.load_state_dict(torch.load(net.filepath + f'/{args.best_model}.mdl'))
	net.eval()

	return net.predict_evaluate(testloader, model_type=args.best_model)

if __name__ == "__main__":
	# Step [1]: Set up the WANDB environment
	wandb.init(project=args.project_name, config=args)
	wandb.run.name = args.session_name

	# Step [2]: Run the experiments
	loss_dict, score_dict = do_session(args)

	# Step [3]: Get the final test scores and display them
	list_summaries, list_metrics = [], [key for key in REPORT_METRICS]
	list_columns = [
			'[0] Model State for Evaluation',
			'[1] Ingredient Word Vectors', 
			'[2] Encoder Module for Ingredients and Sets', 
			'[3-1] Ingredient Loss Function',
			'[3-2] Recipe Loss Function',
			'[3-3] Recipe Coef', 
			'Calculated Loss for Ingredient Prediction',
			'Calculated Loss for Recipe Prediction']
	row1 = [
		args.best_model, 
		args.ingred_embed, 
		args.ingred_encoder, 
		args.ingred_loss, 
		args.recipe_loss, 
		args.recipe_coef,
		loss_dict['test/mloss'], 
		loss_dict['test/rloss']]
	row2 = [round(score_dict[metric], 3) for metric in list_metrics]
	list_summaries.append(row1+row2)
	if args.quick_mode: exit()

	# Step [4]: Save and print the results for the experiment sheeT	
	resultdf = pd.DataFrame(list_summaries, columns=list_columns+list_metrics)
	resultdf.to_csv(f'{args.saved_path}{args.session_name}_{args.best_model}.csv')
	for row in list_summaries: print(*row, sep="\t")
	wandb.finish()
