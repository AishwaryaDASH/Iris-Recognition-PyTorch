#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import numpy as np
import json, logging


#------------------------------------------------------------------------------
#   Logger
#------------------------------------------------------------------------------
class Logger(object):
	"""
	Training process logger used by BaseTrainer to save training history
	"""
	def __init__(self, logname, logfile):
		self.entries = {}
		self._build_logger(logname, logfile)

	def add_entry(self, entry):
		self.entries[len(self.entries) + 1] = entry

	def info(self, message):
		self.logger.info(message)

	def warn(self, message):
		self.logger.warn(message)

	def warning(self, message):
		self.logger.warning(message)

	def error(self, message):
		self.logger.error(message)

	def __str__(self):
		return json.dumps(self.entries, sort_keys=True, indent=4)

	def _build_logger(self, logname, logfile):
		logging.basicConfig(
			level=logging.INFO,
			format="%(asctime)s-%(levelname)s-%(name)s-%(filename)s-%(lineno)d: %(message)s",
			handlers=[
				logging.FileHandler(logfile),
				logging.StreamHandler(),
		])
		self.logger = logging.getLogger(logname)


#------------------------------------------------------------------------------
#   ImproveChecker
#------------------------------------------------------------------------------
class ImproveChecker(object):
	def __init__(self, mode='min', init_val=np.inf):
		assert mode in ['min', 'max']
		self.mode = mode
		self.best_val = init_val

	def check(self, val):
		if self.mode=='min':
			if val < self.best_val:
				print("[%s] Monitor improved from %f to %f" %(self.__class__.__name__, self.best_val, val))
				self.best_val = val
				return True
			else:
				print("[%s] Monitor not improved from %f" %(self.__class__.__name__, self.best_val))
				return False

		else:
			if val > self.best_val:
				print("[%s] Monitor improved from %f to %f" %(self.__class__.__name__, self.best_val, val))
				self.best_val = val
				return True
			else:
				print("[%s] Monitor not improved from %f" %(self.__class__.__name__, self.best_val))
				return False
