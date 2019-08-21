#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import os, argparse
from glob import glob
import tensorflow as tf
from collections import defaultdict
from matplotlib import pyplot as plt


#------------------------------------------------------------------------------
#   get_log
#------------------------------------------------------------------------------
def get_log(path, tags=['loss', 'lr', 'metrics']):
	logs = defaultdict(lambda: [])
	for event in tf.train.summary_iterator(path):
		for v in event.summary.value:
			if any(tag in v.tag for tag in tags):
				logs[v.tag].append(v.simple_value)
	return logs


#------------------------------------------------------------------------------
#   Parameters
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='ArgumentParser')
parser.add_argument('-d', '--dir', default=None, type=str, help='Root dir of saving Tensorboard data')
args = parser.parse_args()

TRAIN_DIR = os.path.join(args.dir, "train")
VALID_DIR = os.path.join(args.dir, "valid")


#------------------------------------------------------------------------------
#   Main execution
#------------------------------------------------------------------------------
# Get logs
train_file = glob(os.path.join(TRAIN_DIR, "*.*"))
valid_file = glob(os.path.join(VALID_DIR, "*.*"))
print("train_file", train_file)
print("valid_file", valid_file)
train_logs = get_log(train_file[0])
valid_logs = get_log(valid_file[0])

# Plot
for tag in train_logs.keys():
	train_vals = train_logs[tag]
	valid_vals = valid_logs[tag]

	plt.figure(1, dpi=200)
	plt.clf()
	plt.plot(list(range(1,1+len(train_vals))), train_vals, '--r')
	plt.plot(list(range(1,1+len(valid_vals))), valid_vals, '-b')
	if ('loss' in tag) or ('lr' in tag):
		plt.yscale('log')
	plt.title(tag)
	plt.grid(True)
	plt.legend(['train', 'valid'])
	plt.savefig(os.path.join(args.dir, "%s.png" % (tag.replace("/", "."))))
