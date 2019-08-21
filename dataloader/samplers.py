#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import torch
from torch.utils.data.dataloader import default_collate


#------------------------------------------------------------------------------
#  patch_collate_fn
#------------------------------------------------------------------------------
def patch_collate_fn(batch):
	data = dict()
	for key in batch[0]:
		samples = [element.unsqueeze(0) for sample in batch for element in sample[key]]
		data[key] = torch.cat(samples, dim=0)
	return data
