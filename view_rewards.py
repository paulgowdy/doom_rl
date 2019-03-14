import matplotlib.pyplot as plt
import pickle

reward_fn = 'saved_models/simple_3/score_record.txt'
save_fn = 'saved_models/simple_3/score_record.p'

with open(reward_fn, 'r') as f:

	rewards = f.readlines()

rewards = [x.split(',') for x in rewards]
rewards = [float(x[1].split(']')[0]) for x in rewards]

print(len(rewards))
plt.figure()
plt.plot(rewards)
plt.show()

print('pickling rewards...')

with open(save_fn, 'wb') as f:

	pickle.dump(rewards, f)
