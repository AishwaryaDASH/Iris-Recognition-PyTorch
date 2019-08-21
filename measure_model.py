#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
from time import time
import torch, argparse
from models import ClassificationModel


#------------------------------------------------------------------------------
#   Argument parsing
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Arguments for the script")

parser.add_argument('--use_cuda', action='store_true', default=False,
					help='Use GPU acceleration')

parser.add_argument('--height', type=int, default=224,
					help='Height of the input')

parser.add_argument('--width', type=int, default=224,
					help='Width of the input')

parser.add_argument('--in_channels', type=int, default=3,
					help='Size of the input')

parser.add_argument('--n_measures', type=int, default=10,
					help='Number of time measurements')

args = parser.parse_args()


#------------------------------------------------------------------------------
#	Create model
#------------------------------------------------------------------------------
model = ClassificationModel('efficientnet_b0', num_classes=100, pretrained=True)
model.eval()
model.summary(input_shape=(args.in_channels, args.height, args.width), device='cpu')


# #------------------------------------------------------------------------------
# #   Measure time
# #------------------------------------------------------------------------------
# input = torch.randn([1, args.in_channels, args.height, args.width], dtype=torch.float)
# if args.use_cuda:
# 	model.cuda()
# 	input = input.cuda()

# for _ in range(10):
# 	model(input)

# start_time = time()
# for _ in range(args.n_measures):
# 	model(input)
# finish_time = time()

# if args.use_cuda:
# 	print("Inference time on cuda: %.2f [ms]" % ((finish_time-start_time)*1000/args.n_measures))
# 	print("Inference fps on cuda: %.2f [fps]" % (1 / ((finish_time-start_time)/args.n_measures)))
# else:
# 	print("Inference time on cpu: %.2f [ms]" % ((finish_time-start_time)*1000/args.n_measures))
# 	print("Inference fps on cpu: %.2f [fps]" % (1 / ((finish_time-start_time)/args.n_measures)))
