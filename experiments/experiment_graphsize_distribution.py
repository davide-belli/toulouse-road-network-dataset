import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from config import ensure_dir
from dataset import ToulouseRoadNetworkDataset, custom_collate_fn


sns.set(color_codes=True)


def generate_plots(plot_type=""):
    r"""
    Generate plots studying the distribution of graphs in different splits with respect to the graph size (|V| and |E|)
    
    :param plot_type: type of plot in {"histograms", "marginal_E", "marginal_V", "joint"}
    """
    assert plot_type in {"histograms", "marginal_E", "marginal_V", "joint"}
    split_names = ["test", "valid", "train"]
    
    tot_n_nodes = []
    tot_n_edges = []
    for split_name in split_names:
        d = ToulouseRoadNetworkDataset(split=split_name, step=0.001, max_prev_node=8)
        dataloader = DataLoader(d, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
        
        n_nodes = []
        n_edges = []
        for datapoint in dataloader:
            this_x_adj, this_x_coord, this_y_adj, this_y_coord, this_img, this_seq_len, this_id = datapoint
            n_edges.append(int(this_y_adj.view(-1).sum().item()))
            n_nodes.append(int(this_seq_len[0]-2))
            
        tot_n_edges += n_edges
        tot_n_nodes += n_nodes
        n_nodes = np.array(n_nodes)
        n_edges = np.array(n_edges)
    
        print(f"{split_name} min/mean/max len nodes", np.min(n_nodes), np.mean(n_nodes), np.max(n_nodes))
        print(f"{split_name} min/mean/max len edges", np.min(n_edges), np.mean(n_edges), np.max(n_edges))
        
        if plot_type == "histograms":
            plt.hist(n_nodes, bins=np.max(n_nodes)-np.min(n_nodes)+1)  # arguments are passed to np.histogram
            plt.title(f"Histogram of |V| for {split_name}")
            plt.savefig(f"plots/histogram_|V|_{split_name}.png")
            plt.clf()
            plt.hist(n_edges, bins=np.max(n_edges)-np.min(n_edges)+1)  # arguments are passed to np.histogram
            plt.title(f"Histogram of |E| for {split_name}")
            plt.savefig(f"plots/histogram_|E|_{split_name}.png")
            plt.clf()
        elif plot_type == "marginal_V":
            a = sns.kdeplot(n_nodes, bw=.5, shade=True, label=split_name)
        elif plot_type == "marginal_E":
            b = sns.kdeplot(n_edges, bw=.5, shade=True, label=split_name)
        else:
            sns_plot = sns.jointplot(np.log10(n_nodes), np.log10(n_edges), marginal_kws=dict(kernel="gau", bw=.02),
                                     kind="kde", bw=.05)
            sns_plot.ax_joint.set_xlabel("log10 |V|", fontsize=15)
            sns_plot.ax_joint.set_ylabel("log10 |E|", fontsize=15)
            sns_plot.ax_marg_x.set_title(split_name, fontsize=20)
            sns_plot.ax_joint.set_xlim(0.6, 1.2)
            sns_plot.ax_joint.set_ylim(0.4, 1.2)
            sns_plot.savefig(f"plots/joint_{split_name}.png")
        
    tot_n_nodes = np.array(tot_n_nodes)
    tot_n_edges = np.array(tot_n_edges)
    print(f"min/mean/max len nodes", np.min(tot_n_nodes), np.mean(tot_n_nodes), np.max(tot_n_nodes))
    print(f"min/mean/max len edges\n", np.min(tot_n_edges), np.mean(tot_n_edges), np.max(tot_n_edges))
    
    if plot_type == "marginal_V":
        a.set_xlabel("|V|")
        a.set_ylabel("p(x)")
        a.set_title("Distributions of |V|")
        a.legend()
        a.figure.savefig(f"plots/marginal_|V|.png")
        a.figure.clf()

    if plot_type == "marginal_E":
        b.set_xlabel("|E|")
        b.set_ylabel("p(x)")
        b.set_title("Distributions of |E|")
        b.legend()
        b.figure.savefig(f"plots/marginal_|E|.png")
        b.figure.clf()
    
    print("Done!")


if __name__ == '__main__':
    ensure_dir("plots/")
    for plot_type in ["histograms", "marginal_E", "marginal_V", "joint"]:
        generate_plots(plot_type)


"""
Expected output:

Started loading the dataset (test)...
Started loading the images...
Dataset loading completed, took 7.32 seconds!
Dataset size: 18998
test min/mean/max len nodes 5 6.412464469944204 9
test min/mean/max len edges 2 5.0792715022633965 12

Started loading the dataset (valid)...
Started loading the images...
Dataset loading completed, took 4.51 seconds!
Dataset size: 11679
valid min/mean/max len nodes 5 6.277506635842109 9
valid min/mean/max len edges 2 4.965579244798356 13

Started loading the dataset (train)...
Started loading the images...
Dataset loading completed, took 30.38 seconds!
Dataset size: 80357
train min/mean/max len nodes 5 6.293291188073224 9
train min/mean/max len edges 2 4.992931542989409 14

Total
min/mean/max len nodes 5 6.3120215429508075 9
min/mean/max len edges 2 5.004827350181025 14
Done!
"""