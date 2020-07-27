# IMPORT LIBRARY

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
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, AdamW, BertConfig

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

# FUNCTION DEFINITION 

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

# ARGUMENTS
parser = argparse.ArgumentParser(description='CNN text classifier')
# GENERAL
parser.add_argument('-task-name',type=str,default='fakeNews', help='task name [default : fakeNews]')
parser.add_argument('-data-dir',type=str,default='data/train.tsv', help='data dir [default : data/train.tsv]')
parser.add_argument('-output-dir',type=str,default='outputs/fakeNews', help='output dir [default : outputs/fakeNews]')
parser.add_argument('-reports-dir',type=str,default='reports/fakeNews_evaluation_report', help='reports dir [default : reports/fakeNews_evaluation_report]')
parser.add_argument('-cache-dir',type=str,default='cache', help='cache dir [default : cache]')
parser.add_argument('-cuda-num',type=int,default=0, help='cuda device num [default : 0]')
parser.add_argument('-epoch-num',type=int,default=10,help='training epoch [default : 10]')
parser.add_argument('-mode',type=str,default='train', help='train or test or dev mode [default : train]')
parser.add_argument('-batch-size',type=int,default=30, help='batch size [default : 30]')
parser.add_argument('-lr',type=float,default=2e-5, help='learning rate [default : 2e-5]')
# CNN
parser.add_argument('-cnn',type=bool,default=True, help='cnn on off [default : True]')
parser.add_argument('-cnn-max-len',type=int,default=300, help='maximum evidence length [default : 300]')
parser.add_argument('-embed-dim',type=int,default=300, help='embed dim for evidence [default : 300]')
parser.add_argument('-kernel-num',type=int,default =100,help='number of each kind of kernel')
parser.add_argument('-kernel-sizes',type=str,default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-cnn-dropout',type=float,default=0.1, help='the probability for dropout [default : 0.1]')
# BERT
parser.add_argument('-bert-max-len',type=int,default=128, help='maximum statement length [default : 128]')
parser.add_argument('-bert-hidden-dropout',type=float,default=0.1, help='bert hidden dropout [default : 0.1]')
parser.add_argument('-bert-att-dropout',type=float,default=0.1, help='bert attention dropout [default : 0.1]')
parser.add_argument('-bert-model',type=str,default='bert-base-cased', help='bert model name [default : bert-base-cased]')
parser.add_argument('-weights-name',type=str,default='pytorch_model.bin', help='bert weights name [default : pytorch_model.bin]')
parser.add_argument('-config-name',type=str,default='config.json', help='bert config name [default : config.json]')

args = parser.parse_args()
argsDict = vars(args)

################################ GENERAL ####################################
device = torch.device("cuda:"+str(args.cuda_num) if torch.cuda.is_available() else "cpu")

################################ CNN PART ####################################

args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

dataReader = csvReader(args.data_dir)

contextList = dataReader.getContextList()
wordEmbedding = Embedding(args.embed_dim)
allInputCNN = wordEmbedding.getEmbedTensor(contextList,args.cnn_max_len)
cnnModel = CNN_Text(args)

######################### FC PART ########################

cnnOutputDim = len(args.kernel_sizes)*args.kernel_num

# bert hidden dim : 768, class label : 2
if(args.cnn):
	fcLayer = nn.Linear(cnnOutputDim+768,2)
else:
	fcLayer = nn.Linear(768,2)

###################### BERT PART ########################
# MODEL
if(args.mode =='train'):
	configuration = BertConfig(hidden_dropout_prob= args.bert_hidden_dropout, attention_probs_dropout_prob=args.bert_att_dropout)
	modelBERT = BertModel(configuration).from_pretrained(args.bert_model, cache_dir = args.cache_dir)

	param_optimizer = list(modelBERT.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay':0.01},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
		]

	optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

	modelBERT.train()

if(args.mode =='test' or args.mode =='dev'):
	modelBERT = BertModel.from_pretrained(args.cache_dir)
	modelBERT.eval()

	cnnModelFile = os.path.join(args.cache_dir,'cnnModel')
	cnnModel.load_state_dict(torch.load(cnnModelFile))
	cnnModel.eval()

	fcModelFile = os.path.join(args.cache_dir,'fcModel')
	fcLayer.load_state_dict(torch.load(fcModelFile))
	fcLayer.eval()

modelBERT.to(device)
cnnModel.to(device)
fcLayer.to(device)

# READ DATA
processor = BinaryClassificationProcessor()

examples = processor.get_data_examples(args.data_dir)
examples_len = len(examples)

label_list = processor.get_labels()
num_labels = len(label_list)

label_map = {label: i for i, label in enumerate(label_list)}
tokenizer = BertTokenizer.from_pretrained(args.output_dir+'/vocab.txt',do_lower_case=False)
examples_for_processing = [(example, label_map, args.bert_max_len, tokenizer) for example in examples]

process_count = cpu_count() - 1
if __name__=="__main__":
	print(f'Preparing to convert {examples_len} examples..')
	print(f'Spawning {process_count} processes..')
	with Pool(process_count) as p:
		features = list(tqdm_notebook(p.imap(convert_examples_to_features.convert_example_to_feature, examples_for_processing), total=examples_len))

all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,allInputCNN)

dataSampler = SequentialSampler(data)
loader = DataLoader(data, sampler=dataSampler, batch_size=args.batch_size)

# TRAIN/TEST/DEV
if(args.mode =='train'):
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", examples_len)
	logger.info("  Batch size = %d", args.batch_size)
	epochDesc = "Epoch"

if(args.mode == 'test' or args.mode =='dev'):
	args.epoch_num = 1
	epochDesc = "Test"


# TEST/DEV directory generation
if(args.mode =='test' or args.mode =='dev'):
	if not os.path.exists(args.reports_dir):
		os.makedirs(args.reports_dir)

	if os.path.exists(args.reports_dir) :
		args.reports_dir += f'/report_{len(os.listdir(args.reports_dir))}'
		if not os.path.exists(args.reports_dir):
			os.makedirs(args.reports_dir)

preds = []
for i in trange(int(args.epoch_num), desc = epochDesc):
	totalLoss = 0
	nb_steps =0 

	for input_ids, input_mask, segment_ids, label_ids,inputCNN in tqdm_notebook(loader):
		input_ids = input_ids.to(device)
		input_mask = input_mask.to(device)
		segment_ids = segment_ids.to(device)
		label_ids = label_ids.to(device)
		inputCNN = inputCNN.to(device)

		outputBERT = modelBERT(input_ids, segment_ids, input_mask)
		lastLayerCLS = outputBERT[0][:,0,:].squeeze()
		if(args.cnn):
			cnnOutput = cnnModel(inputCNN)
			logits = fcLayer(torch.cat((lastLayerCLS,cnnOutput),1))
		else:
			logits = fcLayer(lastLayerCLS)
		
		loss_fct = CrossEntropyLoss()
		loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
		
		print("\r%f" % loss,end='')
		if(args.mode =='train'):
			loss.backward()
		
		totalLoss += loss.mean().item()
		nb_steps += 1
		if len(preds) ==0:
			preds.append(logits.detach().cpu().numpy())
		else:
			preds[0] = np.append(preds[0],logits.detach().cpu().numpy(), axis=0)
		
	totalLoss = totalLoss / nb_steps

	if(args.mode =='train'):
		optimizer.step()
		optimizer.zero_grad()
		epochDir = args.output_dir+"/"+str(i)+"epoch"
		if not os.path.exists(epochDir):
			os.makedirs(epochDir)

		flog = open(args.output_dir+'/parameters.txt','w')
		for key in argsDict.keys():
			flog.write(str(key)+'\t'+str(argsDict[key])+'\n')

		model_to_save = modelBERT.module if hasattr(modelBERT, 'module') else modelBERT

		output_model_file = os.path.join(epochDir, args.weights_name)
		output_config_file = os.path.join(epochDir, args.config_name)

		torch.save(model_to_save.state_dict(), output_model_file)
		model_to_save.config.to_json_file(output_config_file)
		tokenizer.save_vocabulary(args.output_dir)

		# SAVE OTHER MODELS
		cnnModelFile = os.path.join(epochDir, 'cnnModel')
		torch.save(cnnModel.state_dict(),cnnModelFile)

		fcModelFile = os.path.join(epochDir,'fcModel')
		torch.save(fcLayer.state_dict(),fcModelFile)

	# EVALUATION RESULT
	if(args.mode == 'test' or args.mode =='dev'):
		flog = open(args.reports_dir+'/parameters.txt','w')
		for key in argsDict.keys():
			flog.write(str(key)+'\t'+str(argsDict[key])+'\n')

		# PRINT RESULT
		preds = preds[0]
		preds = np.argmax(preds, axis = 1)
		result = compute_metrics(args.task_name,all_label_ids.numpy(),preds)
		result['totalLoss'] = totalLoss
		output_file = os.path.join(args.reports_dir, args.mode+"_results.txt")
		with open(output_file,'w') as writer:
			logger.info("**** "+args.mode+" RESULTS ****")
			for key in (result.keys()):
				logger.info(" %s=%s", key, str(result[key]))
				writer.write("%s=%s\n" % (key, str(result[key])))

