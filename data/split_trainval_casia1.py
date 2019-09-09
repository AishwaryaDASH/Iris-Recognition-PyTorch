#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
from tqdm import tqdm
from glob import glob
import os, cv2, random
from collections import defaultdict
random.seed(0)


#------------------------------------------------------------------------------
#  Parameters
#------------------------------------------------------------------------------
IMAGE_DIR = "/home/member/Workspace/thuync/datasets/Iris/CASIA1"


#------------------------------------------------------------------------------
#  Main execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
	# Get files
	files = sorted(glob(os.path.join(IMAGE_DIR, "**/*.*"), recursive=True))
	print("Number of files:", len(files))

	# Aggregate data
	data_dict = defaultdict(lambda: defaultdict(lambda: list()))
	for file in files:
		basename = os.path.basename(file).split('.')[0]
		person_id, eye_id, ins_id = basename.split('_')
		data_dict[person_id][eye_id].append(file)

	for vals in data_dict.values():
		for val in vals.values():
			random.shuffle(val)

	# Split train/valid
	train_list = []
	valid_list = []

	for key, vals in data_dict.items():
		for val in vals.values():
			train_list += [val[i] for i in range(2)]
			valid_list += [val[i] for i in range(2,len(val))]

	# Write to file
	random.shuffle(train_list)
	with open("data/casia1_train.txt", 'w') as fp:
		for file in train_list:
			fp.writelines(file+'\n')

	random.shuffle(valid_list)
	with open("data/casia1_valid.txt", 'w') as fp:
		for file in valid_list:
			fp.writelines(file+'\n')
