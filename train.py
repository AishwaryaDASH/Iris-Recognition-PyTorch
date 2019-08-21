#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import os, json, argparse, functools

import torch
torch.random.manual_seed(0)
from torch.utils.data import DataLoader

import models as module_arch
import evaluation.losses as module_loss
import evaluation.metrics as module_metric
import dataloader as module_data
import optimizers as module_optimizer
import trainer as module_trainer


#------------------------------------------------------------------------------
#   Get instance
#------------------------------------------------------------------------------
def get_instance(module, name, config, *args):
	if name in config:
		return getattr(module, config[name]['type'])(*args, **config[name]['args'])
	else:
		return None


#------------------------------------------------------------------------------
#   Main function
#------------------------------------------------------------------------------
def main(config, resume):

	# Build model
	model = get_instance(module_arch, 'arch', config)
	img_sz = config["train_loader"]['dataset']["args"]["input_size"]
	model.summary(input_shape=(3, *img_sz))
	if config["arch"]["pretrained"] is not None:
		pretrained = torch.load(config["arch"]["pretrained"], map_location='cpu')
		model.load_state_dict(pretrained['state_dict'], strict=True)
		print("All parameters are initialized from %s" % (config["arch"]["pretrained"]))

	# Build dataloader
	train_dataset = get_instance(module_data, 'dataset', config['train_loader'])
	collate_fn = getattr(module_data, config['train_loader']['collate'])
	train_loader = DataLoader(
		dataset=train_dataset,
		batch_size=config['train_loader']['batch_size'],
		shuffle=config['train_loader']['shuffle'],
		num_workers=config['train_loader']['num_workers'],
		pin_memory=config['train_loader']['pin_memory'],
		drop_last=False,
		collate_fn=collate_fn,
	)
	valid_loader = None
	if config.get('valid_loader', None) is not None:
		valid_dataset = get_instance(module_data, 'dataset', config['valid_loader'])
		collate_fn = getattr(module_data, config['valid_loader']['collate'])
		valid_loader = DataLoader(
			dataset=valid_dataset,
			batch_size=config['valid_loader']['batch_size'],
			shuffle=config['valid_loader']['shuffle'],
			num_workers=config['valid_loader']['num_workers'],
			pin_memory=config['valid_loader']['pin_memory'],
			collate_fn=collate_fn,
		)

	# Build loss and metrics
	losses = [
		functools.partial(getattr(module_loss, loss['type']), **loss['args'])
		for loss in config['losses']
	]
	metrics = [
		functools.partial(getattr(module_metric, metric['type']), **metric['args'])
		for metric in config['metrics']
	]

	# Build optimizer and learning rate scheduler
	trainable_params = filter(lambda p: p.requires_grad, model.parameters())
	optimizer = get_instance(module_optimizer, 'optimizer', config, trainable_params)
	grad_clip = config['optimizer']['grad_clip']
	lr_scheduler = get_instance(module_optimizer, 'lr_scheduler', config, optimizer)

	# Create trainer and start training
	Trainer = getattr(module_trainer, config['trainer']['type'])
	trainer = Trainer(
		model=model,
		losses=losses,
		metrics=metrics,
		optimizer=optimizer, 
		resume=resume,
		config=config,
		data_loader=train_loader,
		valid_data_loader=valid_loader,
		lr_scheduler=lr_scheduler,
		grad_clip=grad_clip,
	)
	trainer.train()


#------------------------------------------------------------------------------
#   Main execution
#------------------------------------------------------------------------------
if __name__ == '__main__':

	# Argument parsing
	parser = argparse.ArgumentParser(description='Train model')

	parser.add_argument('-c', '--config', default='config/resnet18.json', type=str,
						   help='config file path')

	parser.add_argument('-r', '--resume', default=None, type=str,
						   help='path to latest checkpoint (default: None)')

	parser.add_argument('-d', '--device', default='-1', type=str,
						   help='indices of GPUs to enable (default: all)')
 
	args = parser.parse_args()


	# Load config file
	if args.config:
		config = json.load(open(args.config))
		path = os.path.join(config['trainer']['save_dir'], config['name'])


	# Load config file from checkpoint, in case new config file is not given.
	# Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
	elif args.resume:
		config = torch.load(args.resume)['config']


	# AssertionError
	else:
		raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")
	

	# Set visible devices
	if args.device:
		os.environ["CUDA_VISIBLE_DEVICES"]=args.device


	# Run the main function
	main(config, args.resume)
