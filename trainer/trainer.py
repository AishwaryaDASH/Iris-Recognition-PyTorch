#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad
from torchvision.utils import make_grid

import numpy as np
from tqdm import tqdm
from base import BaseTrainer
from collections import OrderedDict


#------------------------------------------------------------------------------
#   ClassificationTrainer
#------------------------------------------------------------------------------
class ClassificationTrainer(BaseTrainer):
	def __init__(self, model, losses, metrics, optimizer, resume, config, data_loader,
				valid_data_loader=None, lr_scheduler=None, grad_clip=None):

		super(ClassificationTrainer, self).__init__(
			model, losses, metrics, optimizer, resume,
			config, data_loader, valid_data_loader, lr_scheduler, grad_clip,
		)

	def _forward(self, data):
		self.optimizer.zero_grad()
		output = self.model(**data)
		losses = OrderedDict([
			(loss_func.func.__name__, loss_func(**output, **data))
			for loss_func in self.losses
		])
		loss = self._sum_losses(losses)
		return output, losses, loss

	def _get_progress_bar_dict(self, losses, loss, metrics):
		pbar_dict = dict()
		pbar_dict['lr'] = self.optimizer.param_groups[0]['lr']
		# for key, val in losses.items():
		# 	pbar_dict[key] = val.item()
		pbar_dict['loss'] = loss
		if self.verbosity>=3:
			for i, metric in enumerate(self.metrics):
				pbar_dict['%s'%(metric.func.__name__)] = metrics[i]
		return pbar_dict
