import torch
import numpy as np
import pickle

from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss

from lib.tools import *
from multiprocessing import Pool, cpu_count
import lib.convert_examples_to_features as convert_examples_to_features

from tqdm import tqdm_notebook, trange
import os
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, AdamW
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
#from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

# for CNN
import torch.nn as nn
import torchtext
from torchtext.vocab import GloVe

from torchtext import data
from torchtext.data import TabularDataset

from torchtext.data import Iterator

from lib.cnnModel import CNN_Text
from lib.readCSV import csvReader
from lib.wordEmbed import Embedding

import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################ CNN PART ####################################

embedSize = 10000
embedDim = 50 
maxLen = 200

parser = argparse.ArgumentParser(description='CNN text classifier')
parser.add_argument('-epoch-num',type=int,default=10,help='training epoch [default : 10]')
parser.add_argument('-kernel-num',type=int,default =100,help='number of each kind of kernel')
parser.add_argument('-kernel-sizes',type=str,default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-dropout',type=float,default=0.5, help='the probability for dropout [default : 0.5]')

args = parser.parse_args()
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.class_num = 2
args.embed_num = maxLen
args.embed_dim = embedDim

devReader = csvReader('data/dev.csv')
devContextList = devReader.getContextList()

wordEmbedding = Embedding(embedDim)
allInputCNN = wordEmbedding.getEmbedTensor(devContextList,200)
cnnModel = CNN_Text(args)
cnnModel.to(device)

######################### FC PART ########################

bertHiddenDim = 768
cnnOutputDim = len(args.kernel_sizes)*args.kernel_num

fcLayer = nn.Linear(bertHiddenDim+cnnOutputDim,args.class_num)
fcLayer.to(device)


###################### BERT PART ########################

DATA_DIR = "data/"

BERT_MODEL = 'fakeNews.tar.gz' 
TASK_NAME = 'fakeNews'

OUTPUT_DIR = f'outputs/{TASK_NAME}/'
REPORTS_DIR = f'reports/{TASK_NAME}_evaluation_report/'
CACHE_DIR = 'cache/'

MAX_SEQ_LENGTH = 128

TRAIN_BATCH_SIZE = 30
EVAL_BATCH_SIZE = 30
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 1 
RANDOM_SEED = 42
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1
OUTPUT_MODE = 'classification'

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

if not os.path.exists(REPORTS_DIR):
	os.makedirs(REPORTS_DIR)

if os.path.exists(REPORTS_DIR) :
	REPORTS_DIR += f'report_{len(os.listdir(REPORTS_DIR))}'
	if not os.path.exists(REPORTS_DIR):
		os.makedirs(REPORTS_DIR)

def get_eval_report(task_name,labels,preds):
	mcc=matthews_corrcoef(labels, preds)
	tn, fp, fn, tp = confusion_matrix(labels,preds).ravel()
	f1 = f1_score(labels, preds,average='binary')
	return {
		"task":task_name,
		"mcc":mcc,
		"tp":tp,
		"tn":tn,
		"fp":fp,
		"fn":fn,
		"f1":f1
	}

def compute_metrics(task_name,labels,preds):
	assert len(preds) == len(labels)
	return get_eval_report(task_name,labels,preds)

tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR+'vocab.txt',do_lower_case=False)

processor = BinaryClassificationProcessor()
eval_examples = processor.get_dev_examples(DATA_DIR)
label_list = processor.get_labels()
num_labels = len(label_list)
eval_examples_len = len(eval_examples)

label_map = {label: i for i, label in enumerate(label_list)}
eval_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in eval_examples]

process_count = cpu_count() - 1
if __name__=="__main__":
	print(f'Preparing to convert {eval_examples_len} examples..')
	print(f'Spawning {process_count} processes..')
	with Pool(process_count) as p:
		eval_features = list(tqdm_notebook(p.imap(convert_examples_to_features.convert_example_to_feature, eval_examples_for_processing), total=eval_examples_len))

all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

if OUTPUT_MODE == "classification":
	all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
elif OUTPUT_MODE == "regression":
	all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,allInputCNN)

eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)

#model = BertForSequenceClassification.from_pretrained(CACHE_DIR+BERT_MODEL, cache_dir = CACHE_DIR, num_labels =len(label_list))
modelBERT = BertModel.from_pretrained(CACHE_DIR)
modelBERT.to(device)

modelBERT.eval()
eval_loss = 0
nb_eval_steps =0 
preds = []

for input_ids, input_mask, segment_ids, label_ids,inputCNN in tqdm_notebook(eval_dataloader,desc='Evaluating'):
	input_ids = input_ids.to(device)
	input_mask = input_mask.to(device)
	segment_ids = segment_ids.to(device)
	label_ids = label_ids.to(device)
	inputCNN = inputCNN.to(device)

	with torch.no_grad():
		outputBERT = modelBERT(input_ids, segment_ids, input_mask)
		lastLayerCLS = outputBERT[0][:,0,:].squeeze()
		cnnOutput = cnnModel(inputCNN)
		logits = fcLayer(torch.cat((lastLayerCLS,cnnOutput),1))
	
	if OUTPUT_MODE == 'classification':
		loss_fct = CrossEntropyLoss()
		tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
	elif OUTPUT_MODE == 'regression':
		loss_fct = MSELoss()
		tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
	
	eval_loss += tmp_eval_loss.mean().item()
	nb_eval_steps += 1
	if len(preds) ==0:
		preds.append(logits.detach().cpu().numpy())
	else:
		preds[0] = np.append(preds[0],logits.detach().cpu().numpy(), axis=0)
	
eval_loss = eval_loss / nb_eval_steps
preds = preds[0]
if OUTPUT_MODE == 'classification':
	preds = np.argmax(preds, axis = 1)
elif OUTPUT_MODE == 'regression':
	preds = np.squeeze(preds)
result = compute_metrics(TASK_NAME,all_label_ids.numpy(),preds)

result['eval_loss'] = eval_loss

output_eval_file = os.path.join(REPORTS_DIR, "eval_results.txt")
with open(output_eval_file,'w') as writer:
	logger.info("**** EVAL RESULTS ****")
	for key in (result.keys()):
		logger.info(" %s=%s", key, str(result[key]))
		writer.write("%s=%s\n" % (key, str(result[key])))
