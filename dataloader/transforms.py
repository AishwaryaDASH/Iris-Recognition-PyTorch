#------------------------------------------------------------------------------
#   Library
#------------------------------------------------------------------------------
import cv2, functools
import numpy as np
np.random.seed(0)


#------------------------------------------------------------------------------
#   Basic functions
#------------------------------------------------------------------------------
def _slice_func(x, i, j, size):
	return x[i:i+size, j:j+size, ...]

_flip_hor = functools.partial(np.flip, axis=1)
_flip_ver = functools.partial(np.flip, axis=0)


#------------------------------------------------------------------------------
#   Random crop
#------------------------------------------------------------------------------
def random_crop(*inputs, crop_range=[1.0, 1.0]):
	##### Exception #####
	if crop_range[0]==crop_range[1] and crop_range[0]==1.0:
		inputs = inputs[0] if len(inputs)==1 else inputs
		return inputs

	# Get random crop_ratio
	crop_ratio = np.random.choice(np.linspace(crop_range[0], crop_range[1], num=10), size=())
	
	# Get random coordinates
	H, W = inputs[0].shape[:2]
	size = H if H<W else W
	size = int(size*crop_ratio)
	max_i, max_j = H-size, W-size
	i = np.random.choice(np.arange(0, max_i+1), size=())
	j = np.random.choice(np.arange(0, max_j+1), size=())

	# Crop
	inputs = tuple(map(functools.partial(_slice_func, i=i, j=j, size=size), inputs))
	inputs = inputs[0] if len(inputs)==1 else inputs
	return inputs


#------------------------------------------------------------------------------
#   Horizontal flip
#------------------------------------------------------------------------------
def flip_horizon(*inputs, prob=0):
	if prob:
		if np.random.choice([False, True], size=(), p=[1-prob, prob]):
			inputs = tuple(map(_flip_hor, inputs))
	inputs = inputs[0] if len(inputs)==1 else inputs
	return inputs


#------------------------------------------------------------------------------
#   Vertical flip
#------------------------------------------------------------------------------
def flip_vertical(*inputs, prob=0):
	if prob:
		if np.random.choice([False, True], size=(), p=[1-prob, prob]):
			inputs = tuple(map(_flip_ver, inputs))
	inputs = inputs[0] if len(inputs)==1 else inputs
	return inputs


#------------------------------------------------------------------------------
#   Rotate 90
#------------------------------------------------------------------------------
def rotate_90(*inputs, prob=0):
	if prob:
		k = np.random.choice([-1, 0, 1], size=(), p=[prob/2, 1-prob, prob/2])
		if k:
			inputs = tuple(map(functools.partial(np.rot90, k=k, axes=(0,1)), inputs))
	inputs = inputs[0] if len(inputs)==1 else inputs
	return inputs


#------------------------------------------------------------------------------
#   Rotate angle
#------------------------------------------------------------------------------
def rotate_angle(*inputs, angle_max):
	if angle_max:
		# Random angle in range [-angle_max, angle_max]
		angle = np.random.choice(np.linspace(-angle_max, angle_max, num=21), size=())

		# Get parameters for affine transform
		(h, w) = inputs[0].shape[:2]
		(cX, cY) = (w // 2, h // 2)

		M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
		cos = np.abs(M[0, 0])
		sin = np.abs(M[0, 1])

		nW = int((h * sin) + (w * cos))
		nH = int((h * cos) + (w * sin))

		M[0, 2] += (nW / 2) - cX
		M[1, 2] += (nH / 2) - cY

		# Perform transform
		inputs = tuple(map(functools.partial(cv2.warpAffine, M=M, dsize=(nW, nH)), inputs))
	inputs = inputs[0] if len(inputs)==1 else inputs
	return inputs


#------------------------------------------------------------------------------
#  Gaussian noise
#------------------------------------------------------------------------------
def random_noise(image, std):
	if std:
		noise = np.random.normal(0, std, size=image.shape)
		image = image + noise
		image[image<0] = 0
		image[image>255] = 255
		image = image.astype(np.uint8)
	return image


#------------------------------------------------------------------------------
#  Resize image
#------------------------------------------------------------------------------
def resize_image(image, height, width, mode=cv2.INTER_LINEAR):
	if not (image.shape[0]==height and image.shape[1]==width):
		image = cv2.resize(image, (width, height), interpolation=mode)
	return image
