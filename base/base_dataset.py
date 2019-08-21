#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import os, cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


#------------------------------------------------------------------------------
#   Image loader
#------------------------------------------------------------------------------
def img_loader_opencv(filepath, gray=False):
	if gray:
		return cv2.imread(filepath, 0)
	else:
		return cv2.imread(filepath)[...,::-1]

def img_loader_pillow(filepath, gray=False):
	if gray:
		return np.array(Image.open(filepath).convert('L'))
	else:
		return np.array(Image.open(filepath).convert('RGB'))


#------------------------------------------------------------------------------
#   BaseDataset
#------------------------------------------------------------------------------
class BaseDataset(Dataset):

	_FORMAT = ('.jpg', '.png', '.bmp')

	def __init__(self, img_loader_mode='pillow', normalize_mode='imagenet'):
		super(BaseDataset, self).__init__()

		# img_loader_mode
		assert img_loader_mode in ['opencv', 'pillow']

		if img_loader_mode=='opencv':
			self.img_loader = img_loader_opencv

		elif img_loader_mode=='pillow':
			self.img_loader = img_loader_pillow

		# normalize_mode
		assert normalize_mode in ['imagenet', 'tanh', 'none']

		if normalize_mode=='imagenet':
			self.mean = np.array([0.485,0.456,0.406], dtype=np.float32)[None,None,:]
			self.std = np.array([0.229,0.224,0.225], dtype=np.float32)[None,None,:]
			self.normalize_fnc = lambda x: (x/255.0-self.mean)/self.std
			self.denormalize_fnc = lambda x: (x*self.std + self.mean)*255.0

		elif normalize_mode=='tanh':
			self.normalize_fnc = lambda x: x/127.5-1.0
			self.denormalize_fnc = lambda x: (x+1.0)*127.5

		elif normalize_mode=='none':
			self.normalize_fnc = lambda x: x
			self.denormalize_fnc = lambda x: x


	def check_filepaths(self, filepaths):
		print("[%s] Checking file paths..." % (self.__class__.__name__))
		error_flg = False
		for file in filepaths:
			if not os.path.exists(file):
				print("%s does not exist!" % (file))
				error_flg = True
		if error_flg:
			raise ValueError("[%s] Some file paths are corrupted! Please re-check your file paths!" % (self.__class__.__name__))
		else:
			print("[%s] All file paths exist" % (self.__class__.__name__))


	def filter_files(self, image_files):
		return [file for file in image_files if os.path.splitext(file)[1] in self._FORMAT]
