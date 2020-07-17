
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN_Text(nn.Module):
	def __init__(self,args):
		super(CNN_Text,self).__init__()

		D = args.embed_dim
		C = args.class_num
		Ci = 1
		Co = args.kernel_num
		Ks = args.kernel_sizes

		#self.embed = nn.Embedding(V,D)

		self.convs1 = nn.ModuleList([nn.Conv2d(Ci,Co,(K,D)) for K in Ks])

		self.dropout = nn.Dropout(args.dropout)
		self.fc1 = nn.Linear(len(Ks)*Co,C)
	
	def forward(self,x):
		#x = self.embed(x)

		# N * 1 * W * D
		x = x.unsqueeze(1)

		# [N * Cout * Wout]*Ks
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]

		# [N * Cout]*Ks
		x = [F.max_pool1d(i,i.size(2)).squeeze(2) for i in x]

		# N*(Cout*Ks)
		x = torch.cat(x,1)

		x = self.dropout(x)
		return x 
