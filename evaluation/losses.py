#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import functools
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F


#------------------------------------------------------------------------------
#   Cross Entropy loss
#------------------------------------------------------------------------------
def ce_loss(logits, targets, loss_weight=1.0, **kargs):
	"""
	logits: (torch.float32)  shape (N, C)
	targets: (torch.int64) shape (N,), value {0,1,...,C-1}
	"""
	loss = F.cross_entropy(logits, targets)
	return loss * loss_weight
