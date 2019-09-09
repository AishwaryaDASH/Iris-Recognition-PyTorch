#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import numpy as np
from glob import glob
import os, cv2, torch

from base import BaseDataset
from dataloader import transforms
from dataloader.augmentation import Augmentation


#------------------------------------------------------------------------------
#	TxtDataset
#------------------------------------------------------------------------------
class TxtDataset(BaseDataset):

	_FORMAT = ('.jpg', '.png', '.bmp')

	def __init__(self, txtfile, num_classes, use_sigmoid=True, color_channel="RGB",
		input_size=(512,512), normalize=True, one_hot=False, is_training=True, 
		rot90=0.3, flip_hor=0.5, flip_ver=0.5, brightness=0.2, contrast=0.1, shift=0.1625, scale=0.6, rotate=10,
		img_loader_mode='pillow', normalize_mode='imagenet'):

		# Init BaseDataset
		super(TxtDataset, self).__init__(img_loader_mode=img_loader_mode, normalize_mode=normalize_mode)

		# Data augmentation
		self.augmentor = Augmentation(
			rot90=rot90, flip_hor=flip_hor, flip_ver=flip_ver, brightness=brightness,
			contrast=contrast, shift=shift, scale=scale, rotate=rotate,
		)

		# Get image files
		self._get_image_files(txtfile)
		self._get_labels()

		# Parameters
		self.use_sigmoid = use_sigmoid
		self.num_classes = num_classes+1 if self.use_sigmoid else num_classes
		self.color_channel = color_channel
		self.input_size = tuple(input_size)
		self.is_training = is_training
		self.normalize = normalize
		self.one_hot = one_hot

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, idx):
		# Read image and label
		image = self.img_loader(self.image_files[idx])
		label = np.array(self.labels[idx])

		# Data augmentation
		if self.is_training:
			image = self._augment_data(image)

		# Preprocess image
		image = transforms.resize_image(image, height=self.input_size[0], width=self.input_size[1], mode=cv2.INTER_LINEAR)
		if self.normalize:
			image = self.normalize_fnc(image)
		image = np.transpose(image, axes=(2,0,1))

		# Preprocess label
		if self.one_hot:
			label = (np.arange(self.num_classes) == label[..., None])

		# Convert to tensor and return
		data = {
			"images": torch.from_numpy(image.astype(np.float32)),
			"targets": torch.from_numpy(label.astype(np.int64)),
		}
		return data

	def _get_image_files(self, txtfile):
		fp = open(txtfile, 'r')
		image_files = [line for line in fp.read().split("\n") if len(line)]
		self.image_files = self.filter_files(image_files)
		print("[%s] Number of samples:" % (self.__class__.__name__), len(self.image_files))
		self.check_filepaths(self.image_files)

	def _get_labels(self):
		raise NotImplementedError

	def _augment_data(self, image):
		image = self.augmentor(image)
		return image


#------------------------------------------------------------------------------
#  MMU2Dataset
#------------------------------------------------------------------------------
class MMU2Dataset(TxtDataset):
	def __init__(self, *args, **kargs):
		super(MMU2Dataset, self).__init__(args, kargs)

	def _get_labels(self):
		self.labels = [
			int(os.path.basename(file).split(".")[0][:-4])-1
			for file in self.image_files
		]


#------------------------------------------------------------------------------
#  CASIA1Dataset
#------------------------------------------------------------------------------
class CASIA1Dataset(TxtDataset):
	def __init__(self, *args, **kargs):
		super(CASIA1Dataset, self).__init__(args, kargs)

	def _get_labels(self):
		self.labels = [
			int(os.path.basename(file).split('_')[0])-1
			for file in self.image_files
		]
