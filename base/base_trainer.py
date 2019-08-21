#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import numpy as np
from time import time
from tqdm import tqdm
import os, math, json, datetime

import torch
from torch.nn.utils import clip_grad

from utils.logger import Logger
from utils import WriterTensorboardX


#------------------------------------------------------------------------------
#   BaseTrainer
#------------------------------------------------------------------------------
class BaseTrainer(object):
	def __init__(self, model, losses, metrics, optimizer, resume, config, data_loader,
				valid_data_loader=None, lr_scheduler=None, grad_clip=None):

		self.config = config
		self.data_loader = data_loader
		self.valid_data_loader = valid_data_loader
		self.do_validation = self.valid_data_loader is not None
		self.lr_scheduler = lr_scheduler
		self.lr_schedule_by_epoch = config['lr_scheduler']['by_epoch']
		self.grad_clip = grad_clip

		# Setup directory for checkpoint saving
		start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
		self.checkpoint_dir = os.path.join(config['trainer']['save_dir'], config['name'], start_time)
		os.makedirs(self.checkpoint_dir, exist_ok=True)

		# Build logger
		logname = self.config['name']
		logfile = os.path.join(self.checkpoint_dir, "logging.log")
		self.logger = Logger(logname, logfile)

		# Setup GPU device if available, move model into configured device
		self.device, device_ids = self._prepare_device(config['n_gpu'])
		self.model = model.to(self.device)
		if len(device_ids) > 1:
			self.model = torch.nn.DataParallel(model, device_ids=device_ids)

		self.losses = losses
		self.metrics = metrics
		self.optimizer = optimizer

		self.epochs = config['trainer']['epochs']
		self.save_freq = config['trainer']['save_freq']
		self.verbosity = config['trainer']['verbosity']
		self.logger.info("Total epochs: {}".format(self.epochs))

		# configuration to monitor model performance and save best
		self.monitor = config['trainer']['monitor']
		self.monitor_mode = config['trainer']['monitor_mode']
		assert self.monitor_mode in ['min', 'max', 'off']
		self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf
		self.start_epoch = 1

		# setup visualization writer instance
		writer_train_dir = os.path.join(config['visualization']['log_dir'], config['name'], start_time, "train")
		writer_valid_dir = os.path.join(config['visualization']['log_dir'], config['name'], start_time, "valid")
		self.writer_train = WriterTensorboardX(writer_train_dir, self.logger, config['visualization']['tensorboardX'])
		self.writer_valid = WriterTensorboardX(writer_valid_dir, self.logger, config['visualization']['tensorboardX'])

		# Save configuration file into checkpoint directory
		config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
		with open(config_save_path, 'w') as handle:
			json.dump(config, handle, indent=4, sort_keys=False)

		# Resume
		if resume:
			self._resume_checkpoint(resume)

	def train(self):
		for epoch in range(self.start_epoch, self.epochs + 1):
			print("----------------------------------------------------------------")
			self.logger.info("[EPOCH %d/%d]" % (epoch, self.epochs))
			start_time = time()
			result = self._train_epoch(epoch)
			finish_time = time()
			self.logger.info("Finish at {}, Runtime: {:.3f} [s]".format(datetime.datetime.now(), finish_time-start_time))

			# save logged informations into log dict
			log = {}
			for key, value in result.items():
				if key == 'train_metrics':
					log.update({'train_' + mtr.func.__name__ : value[i] for i, mtr in enumerate(self.metrics)})
				elif key == 'valid_metrics':
					log.update({'valid_' + mtr.func.__name__ : value[i] for i, mtr in enumerate(self.metrics)})
				else:
					log[key] = value

			# print logged informations to the screen
			if self.logger is not None:
				self.logger.add_entry(log)
				if self.verbosity >= 1:
					for key, value in sorted(list(log.items())):
						self.logger.info('{:15s}: {}'.format(str(key), value))

			# evaluate model performance according to configured metric, save best checkpoint as model_best
			best = False
			if self.monitor_mode != 'off':
				try:
					if  (self.monitor_mode == 'min' and log[self.monitor] < self.monitor_best) or\
						(self.monitor_mode == 'max' and log[self.monitor] > self.monitor_best):
						self.logger.info("Monitor improved from %f to %f" % (self.monitor_best, log[self.monitor]))
						self.monitor_best = log[self.monitor]
						best = True
				except KeyError:
					if epoch == 1:
						msg = "Warning: Can\'t recognize metric named '{}' ".format(self.monitor)\
							+ "for performance monitoring. model_best checkpoint won\'t be updated."
						self.logger.warning(msg)

			# Save checkpoint
			self._save_checkpoint(epoch, save_best=best)

	def _train_epoch(self, epoch):
		self.logger.info("Train on epoch...")
		self.model.train()
		self.writer_train.set_step(epoch)

		# Perform training
		total_loss = 0
		total_metrics = np.zeros(len(self.metrics))
		n_iter = len(self.data_loader)
		train_pbar = tqdm(enumerate(self.data_loader), total=n_iter)
		for batch_idx, data in train_pbar:
			# Send data to device
			for key, value in data.items():
				data[key] = value.to(self.device)

			# Forward and Backward
			output, losses, loss = self._forward(data)
			self._backward(loss)

			# Learning rate scheduler by iteration
			if self.lr_scheduler is not None and not self.lr_schedule_by_epoch:
				self.lr_scheduler.step()

			# Accumulate loss and metrics
			loss_iter = loss.item()
			total_loss += loss_iter
			metrics_iter = self._eval_metrics(output, data)
			total_metrics += metrics_iter

			# Visualize results
			if (batch_idx==n_iter-2) and (self.verbosity>=2):
				self._visualize_results(output, data)

			# tqdm progress bar
			if self.verbosity>=1:
				pbar_dict = self._get_progress_bar_dict(losses, loss_iter, metrics_iter)
				train_pbar.set_postfix(**pbar_dict)

		# Learning rate scheduler by epoch
		if self.lr_scheduler is not None and self.lr_schedule_by_epoch:
			self.lr_scheduler.step()

		# Record log
		total_loss /= len(self.data_loader)
		total_metrics /= len(self.data_loader)
		log = {'train_loss': total_loss, 'train_metrics': total_metrics.tolist()}

		# Write training result to TensorboardX
		self.writer_train.add_scalar('loss', total_loss)
		for i, metric in enumerate(self.metrics):
			self.writer_train.add_scalar('metrics/%s'%(metric.func.__name__), total_metrics[i])

		if self.verbosity>=2:
			for i in range(len(self.optimizer.param_groups)):
				self.writer_train.add_scalar('lr/group%d'%(i), self.optimizer.param_groups[i]['lr'])

		# Perform validating
		if self.do_validation:
			self.logger.info("Validate on epoch...")
			val_log = self._valid_epoch(epoch)
			log = {**log, **val_log}
		return log

	def _valid_epoch(self, epoch):
		self.model.eval()
		total_val_loss = 0
		total_val_metrics = np.zeros(len(self.metrics))
		n_iter = len(self.valid_data_loader)
		self.writer_valid.set_step(epoch)
		with torch.no_grad():
			# Validate
			for batch_idx, data in tqdm(enumerate(self.valid_data_loader), total=n_iter):
				# Send data to device
				for key, value in data.items():
					data[key] = value.to(self.device)

				# Forward
				output, _, loss = self._forward(data)

				# Accumulate loss and metrics
				total_val_loss += loss.item()
				total_val_metrics += self._eval_metrics(output, data)

				# Visualize results
				if (batch_idx==n_iter-2) and (self.verbosity>=2):
					self._visualize_results(output, data)

			# Record log
			total_val_loss /= len(self.valid_data_loader)
			total_val_metrics /= len(self.valid_data_loader)
			val_log = {
				'valid_loss': total_val_loss,
				'valid_metrics': total_val_metrics.tolist(),
			}

			# Write validating result to TensorboardX
			self.writer_valid.add_scalar('loss', total_val_loss)
			for i, metric in enumerate(self.metrics):
				self.writer_valid.add_scalar('metrics/%s'%(metric.func.__name__), total_val_metrics[i])

		return val_log

	def _forward(self, data):
		self.optimizer.zero_grad()
		output = self.model(**data)
		losses = self.losses(**output, **data)
		loss = self._sum_losses(losses)
		return output, losses, loss

	def _backward(self, loss):
		loss.backward()
		if self.grad_clip is not None:
			clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), **self.grad_clip)
		self.optimizer.step()

	def _eval_metrics(self, output, data):
		acc_metrics = np.zeros(len(self.metrics))
		for i, metric in enumerate(self.metrics):
			acc_metrics[i] += metric(**output, **data)
		return acc_metrics

	def _sum_losses(self, losses):
		loss = sum(loss_val for loss_val in list(losses.values()))
		return loss

	def _visualize_results(self, output, data):
		pass

	def _get_progress_bar_dict(self, losses, loss, metrics):
		pbar_dict = dict()
		pbar_dict['lr'] = self.optimizer.param_groups[0]['lr']
		for key, val in losses.items():
			pbar_dict['loss_%s'%(key)] = val.item()
		pbar_dict['loss'] = loss
		if self.verbosity>=3:
			for i, metric in enumerate(self.metrics):
				pbar_dict['%s'%(metric.func.__name__)] = metrics[i]
		return pbar_dict

	def _prepare_device(self, n_gpu_use):
		""" 
		setup GPU device if available, move model into configured device
		""" 
		n_gpu = torch.cuda.device_count()
		if n_gpu_use > 0 and n_gpu == 0:
			self.logger.warning("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
			n_gpu_use = 0
		if n_gpu_use > n_gpu:
			msg = "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu)
			self.logger.warning(msg)
			n_gpu_use = n_gpu
		device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
		list_ids = list(range(n_gpu_use))
		return device, list_ids

	def _save_checkpoint(self, epoch, save_best=False):
		"""
		Saving checkpoints

		:param epoch: current epoch number
		:param log: logging information of the epoch
		:param save_best: if True, rename the saved checkpoint to 'model_best.pth'
		"""
		# Construct savedict
		arch = type(self.model).__name__
		state = {
			'arch': arch,
			'epoch': epoch,
			'state_dict': self.model.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'monitor_best': self.monitor_best,
			'config': self.config
		}

		# Save checkpoint for each epoch
		if self.save_freq is not None:	# Use None mode to avoid over disk space with large models
			if epoch % self.save_freq == 0:
				filename = os.path.join(self.checkpoint_dir, 'epoch{}.pth'.format(epoch))
				torch.save(state, filename)
				self.logger.info("Saving checkpoint at {}".format(filename))

		# Save the best checkpoint
		if save_best:
			best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
			torch.save(state, best_path)
			self.logger.info("Saving current best at {}".format(best_path))
		else:
			self.logger.info("Monitor is not improved from %f" % (self.monitor_best))

	def _resume_checkpoint(self, resume_path):
		"""
		Resume from saved checkpoints

		:param resume_path: Checkpoint path to be resumed
		"""
		self.logger.info("Loading checkpoint: {}".format(resume_path))
		checkpoint = torch.load(resume_path)
		self.start_epoch = checkpoint['epoch'] + 1
		self.monitor_best = checkpoint['monitor_best']

		# load architecture params from checkpoint.
		if checkpoint['config']['arch'] != self.config['arch']:
			self.logger.warning('Warning: Architecture configuration given in config file is different from that of checkpoint. ' + \
								'This may yield an exception while state_dict is being loaded.')
		self.model.load_state_dict(checkpoint['state_dict'], strict=True)

		# load optimizer state from checkpoint only when optimizer type is not changed. 
		if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
			self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. ' + \
								'Optimizer parameters not being resumed.')
		else:
			self.optimizer.load_state_dict(checkpoint['optimizer'])
	
		self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch-1))
