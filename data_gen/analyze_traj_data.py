import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import json
from common import TrajDataManager

traj_mgr = TrajDataManager('resources/datasets/traj_dataset/', 0, 3270000)

dof_6_indices = []
dof_7_indices = []
path_len_map = defaultdict(lambda: defaultdict(list))

for idx in range(3270000):
    dof, _, _, traj = traj_mgr.retrieve_trajectory(idx)
    if dof == 6:
        dof_6_indices.append(idx)
        path_len_map[traj.shape[0]][6].append(idx)
    if dof == 7:
        dof_7_indices.append(idx)
        path_len_map[traj.shape[0]][7].append(idx)

np.save('resources/datasets/traj_dataset/analysis/dof_6_indices.npy', np.asarray(dof_6_indices, dtype=np.int32))
np.save('resources/datasets/traj_dataset/analysis/dof_7_indices.npy', np.asarray(dof_7_indices, dtype=np.int32))
with open('resources/datasets/traj_dataset/analysis/path_len_map.json', 'w') as json_file:
    json.dump(path_len_map, json_file)

path_len_hist = defaultdict(int)
for key, dof_dict in path_len_map.items():
    path_len_hist[key] += len(dof_dict[6]) + len(dof_dict[7])

path_lens = path_len_hist.keys()
histogram = np.zeros((max(path_lens)+1), dtype=np.int32)
for key, freq in path_len_hist.items():
    histogram[key] = freq
plt.bar(range(histogram.shape[0]), histogram)
plt.savefig('resources/datasets/traj_dataset/analysis/path_len_hist.png')
    