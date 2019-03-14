import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np

reward_files = 'simple_*'

fn_prefix = 'saved_models/'
fn_suffix = '/score_record.p'

rewards_pickles = glob.glob(fn_prefix + reward_files + fn_suffix)

rewards = []

for rp in rewards_pickles:

	with open(rp, 'rb') as f:

		z = pickle.load(f)

	rewards.append(z)

# create average

lens = [len(x) for x in rewards]

average_scores = []

for i in range(max(lens)):

	current_score = 0
	count = 0

	for r in rewards:

		try:

			current_score += r[i]
			count+= 1

		except:

			pass

	current_score = 1.0 * current_score / count

	average_scores.append(current_score)

N = 30
running_ave = np.convolve(average_scores, np.ones(N)/ N, mode = 'valid')

plt.figure()

for r in rewards:

	print(len(r))
	plt.plot(r, c = 'blue', alpha = 0.2)

plt.plot(average_scores, c = 'blue')

plt.plot(running_ave, c = 'fuchsia')

plt.show()
