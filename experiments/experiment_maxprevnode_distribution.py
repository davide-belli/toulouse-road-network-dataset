from dataset import ToulouseRoadNetworkDataset, custom_collate_fn
from config import ensure_dir

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')

sns.set(color_codes=True)


def generate_plots():
    r"""
       Generate plots studying the distribution of max_prev_node distances in the different splits.
       This number corresponds to the fronteer of the BFS ordering
    """
    for split in ["valid", "test", "train"]:
        d = ToulouseRoadNetworkDataset(split=split, max_prev_node=10, step=0.001)
        dataloader = DataLoader(d, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)
        
        prev_nodes = []
        for datapoint in dataloader:
            _, _, y_adj, _, _, seq_len, _ = datapoint
            y_adj = pack_padded_sequence(y_adj, seq_len, batch_first=True)[0].cpu().numpy()
            for i in range(len(y_adj)):
                idxs = np.where(y_adj[i] == 1.)[0]
                if len(idxs) == 0:
                    prev_nodes.append(0)
                else:
                    prev_nodes.append(np.max(idxs) + 1)
        prev_nodes = np.array(prev_nodes)
        print(f"Split: {split}")
        print(f"mean {np.mean(prev_nodes)} | std {np.std(prev_nodes)} | max {np.max(prev_nodes)}")
        perc = np.percentile(prev_nodes, np.array([50., 75., 90., 95., 99., 99.5, 99.9]))
        print(f"Percentiles for {[50., 75., 90., 95., 99., 99.5, 99.9]}")
        print(perc, "\n")
        
        c = Counter(prev_nodes)
        sns_plot = sns.distplot(prev_nodes, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8], norm_hist=True,
                                kde_kws=dict(kernel="gau", bw=.5),
                                hist_kws=dict(rwidth=1, density=True))
        sns_plot.set_xlim(-0.2, 8)
        
        plt.axvline(4.5, 0, 1, ls='--', c='r', label="99th percentile")
        sns_plot.set_ylabel("p(distance) = M")
        sns_plot.set_xlabel("M")
        sns_plot.set_title(f"Distribution of farthest connection in the BFS ({split})")
        sns_plot.legend()
        sns_plot.figure.savefig(f"plots/max_prev_node_{split}.png")
        sns_plot.figure.clf()


if __name__ == "__main__":
    ensure_dir("plots/")
    generate_plots()
    

"""
Expected results:

Split: train
mean 0.8714662137017875 | std 0.9141921523476215 | max 7
Percentiles for [50.0, 75.0, 90.0, 95.0, 99.0, 99.5, 99.9]
[1. 1. 2. 3. 3. 4. 4.]

Split: valid
mean 0.8590506139252946 | std 0.9002315278376958 | max 6
Percentiles for [50.0, 75.0, 90.0, 95.0, 99.0, 99.5, 99.9]
[1.  1.  2.  2.4 3.  4.  4. ]

Split: test
mean 0.87302590414216 | std 0.9215516161662604 | max 6
Percentiles for [50.0, 75.0, 90.0, 95.0, 99.0, 99.5, 99.9]
[1. 1. 2. 3. 3. 4. 4.]

"""