#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2, json
import sys
import numpy as np
import argparse
import models as module_arch
from train import get_instance


#------------------------------------------------------------------------------
#  Utilities
#------------------------------------------------------------------------------
def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = Variable(preprocessed_img, requires_grad = True)
	return input

def show_cam_on_image(img, mask, outfile="pics/cam.jpg"):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	cv2.imwrite(outfile, np.uint8(255 * cam))
	print("Grad-CAM result is saved at", outfile)


#------------------------------------------------------------------------------
#  FeatureExtractor
#------------------------------------------------------------------------------
class FeatureExtractor(object):
	""" Class for extracting activations and 
	registering gradients from targetted intermediate layers """
	def __init__(self, model):
		self.model = model
		self.gradients = []

	def save_gradient(self, grad):
		self.gradients.append(grad)

	def __call__(self, x):
		outputs = []
		self.gradients = []

		x = self.model.conv_stem(x)
		x = self.model.bn1(x)
		x = self.model.act_fn(x, inplace=True)
		x = self.model.blocks(x)
		x = self.model.conv_head(x)
		x = self.model.bn2(x)
		x = self.model.act_fn(x, inplace=True)
		x.register_hook(self.save_gradient)
		outputs += [x]

		x = self.model.global_pool(x)
		x = x.view(x.size(0), -1)
		return outputs, x


#------------------------------------------------------------------------------
#  ModelOutputs
#------------------------------------------------------------------------------
class ModelOutputs(object):
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model)

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output  = self.feature_extractor(x)
		output = output.view(output.size(0), -1)
		output = self.model.classifier(output)
		return target_activations, output


#------------------------------------------------------------------------------
#  GradCam
#------------------------------------------------------------------------------
class GradCam(object):
	def __init__(self, model, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()
		self.extractor = ModelOutputs(self.model)

	def forward(self, input):
		return self.model(input) 

	def __call__(self, input, index = None):
		if self.cuda:
			features, output = self.extractor(input.cuda())
		else:
			features, output = self.extractor(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.zero_grad()
		one_hot.backward(retain_graph=True)

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

		target = features[-1]
		target = target.cpu().data.numpy()[0]

		weights = grads_val[0]
		cam = np.zeros(target.shape[1:], np.float32)

		for i, w in enumerate(weights):
			cam += w * target[i]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (256,256), interpolation=cv2.INTER_LINEAR)
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam


#------------------------------------------------------------------------------
#  get_args
#------------------------------------------------------------------------------
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--image', type=str, help='Input image path')
	parser.add_argument('--config', type=str, default="config/efficientnet_b0.json", help='Config path')
	parser.add_argument('--weight', type=str, help='Trained weight path')
	parser.add_argument('--use-cuda', action='store_true', default=False, help='Use NVIDIA GPU acceleration')
	args = parser.parse_args()

	args.use_cuda = args.use_cuda and torch.cuda.is_available()
	if args.use_cuda:
		print("Using GPU for acceleration")
	else:
		print("Using CPU for computation")

	return args


#------------------------------------------------------------------------------
#  Main execution
#------------------------------------------------------------------------------
if __name__ == '__main__':
	""" python grad_cam.py <path_to_image>
	1. Loads an image with opencv.
	2. Preprocesses it for VGG19 and converts to a pytorch variable.
	3. Makes a forward pass to find the category index with the highest score,
	and computes intermediate activations.
	Makes the visualization. """

	args = get_args()
	config = json.load(open(args.config))
	model = get_instance(module_arch, 'arch', config)
	model.load_pretrained_model(args.weight)
	grad_cam = GradCam(model=model, use_cuda=args.use_cuda)

	img = cv2.imread(args.image)[...,::-1]
	img = np.float32(cv2.resize(img, (256,256), interpolation=cv2.INTER_LINEAR)) / 255
	input = preprocess_image(img)

	mask = grad_cam(input, index=None)
	show_cam_on_image(img, mask, outfile="pics/grad_cam.jpg")
