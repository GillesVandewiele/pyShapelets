import numpy as np


def random_mask(timeseries, mask_size=5):
	"""When discretizing a continous real-valued timeseries, the problem of
	false dismissals arises. This is caused by the fact that two timeseries that
	differ only by a tiny epsilon can result in two different words. To 
	alleviate this problem, we take random subsets of each word by masking
	them. Now a trade-off between false dismissals and false positives must
	be considered. The higher the mask_size, the higher the probability of
	false positives."""
	random_idx = np.random.choice(
		range(timeseries.shape[1]),
		size=timeseries.shape[1] - mask_size,
		replace=False
	)
	return timeseries[:, random_idx]