#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import (
	StepLR, MultiStepLR, ReduceLROnPlateau,
	CyclicLR, CosineAnnealingLR, CosineAnnealingWarmRestarts,
)


#------------------------------------------------------------------------------
#  FixedLR
#------------------------------------------------------------------------------
class FixedLR(_LRScheduler):
	def __init__(self, optimizer, last_epoch=-1):
		super(FixedLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		return [base_lr for base_lr in self.base_lrs]


#------------------------------------------------------------------------------
#  PolyLR
#------------------------------------------------------------------------------
class PolyLR(_LRScheduler):
	def __init__(self, optimizer, max_epoch, power=0.9, last_epoch=-1):
		self.max_epoch = max_epoch
		self.power = power
		super(PolyLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		return [base_lr * (1 - self.last_epoch/self.max_epoch)**self.power for base_lr in self.base_lrs]
