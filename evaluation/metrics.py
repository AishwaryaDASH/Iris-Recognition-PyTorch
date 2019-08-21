#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import torch
import torch.nn.functional as F


#------------------------------------------------------------------------------
#  Accuracy
#------------------------------------------------------------------------------
def acc(logits, targets, **kargs):
	"""
	logits: (torch.float32)  shape (N, C)
	targets: (torch.int64) shape (N), value {0,1,...,C-1}
	"""
	with torch.no_grad():
		outputs = logits.argmax(dim=1).type(targets.dtype)
		return (outputs==targets).float().mean()
