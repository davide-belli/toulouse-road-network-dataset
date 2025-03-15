
# Toulouse Road Network dataset
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
Python code to generate the Toulouse Road Network dataset as introduced in <i>Image-Conditioned Graph Generation for Road Network Extraction</i> (https://arxiv.org/abs/1910.14388)


## Overview
This library contains a PyTorch Dataset Class to use the Toulouse Road Network dataset as presented in [[1]](#citation)(https://arxiv.org/abs/1910.14388), in addition to all the code developed to extract, preprocess, filter, augment and store the dataset.  
You can also find instructions to download a copy of this dataset in the [repo](https://github.com/davide-belli/generative-graph-transformer?tab=readme-ov-file#usage) containing the official implementation and experiments presented in our paper.  
Find out more about this project in our [blog post](https://davide-belli.github.io/toulouse-road-network). 

## Dependencies

See [`requirements.txt`](https://github.com/davide-belli/toulouse-road-network-dataset/requirements.txt)

* **matplotlib==2.2.2**
* **torch==1.1.0**
* **seaborn==0.9.0**
* **numpy==1.14.2**
* **Pillow==6.1.0**
* **pyshp==2.1.0**
* **torchvision==0.4.0**


## Structure
* [`dataset/`](https://github.com/davide-belli/toulouse-road-network-dataset/tree/master/dataset): Should contain the output Toulouse Road Network dataset. If you run `download_dataset.sh` the script will download the dataset as introduced in our paper (Toulouse Road Network dataset).
* [`raw/`](https://github.com/davide-belli/toulouse-road-network-dataset/tree/master/raw): Contains the raw files publicly available at [geofabrik.de](https://www.geofabrik.de/data/shapefiles.html) used as source to extract the Toulouse Road Network dataset.
* [`utils/`](https://github.com/davide-belli/toulouse-road-network-dataset/tree/master/utils): Contains utils for the generation of the dataset. 
* [`experiments/`](https://github.com/davide-belli/toulouse-road-network-dataset/tree/master/experiments): Contains two experiment to study the distribution of graphs in different splits and the size of the BFS fronteer, used to linearize the dataset (see paper or blog post for info).
* [`config.py`](https://github.com/davide-belli/toulouse-road-network-dataset/tree/master/config.py): Configuration used to extract our version of Toulouse Road Network dataset.
* [`generate_toulouse_road_network_dataset.py`](https://github.com/davide-belli/toulouse-road-network-dataset/tree/master/generate_toulouse_road_network_dataset.py): Main script to generate the dataset. It takes a few hours on a laptop. It calls three main subcomponents: 
  * `generate_bins.py` which extracts a grid of bins from the original map to speed up the generation of datapoints. It runs in few seconds.
  * `generate_datapoints.py` generate the main as discussed in the paper and blog post. It takes some hours to run and saves the output in the `dataset/` directory, including raw 64x64 images for the semantic segmentations and pickle files for the graph representations, canonical ordering and other metadata.
  * `generate_image_arrays.py` saves the images as compressed pickle files with numpy array, to speed up performance when loading the dataset for experiments.
* [`dataset.py`](https://github.com/davide-belli/toulouse-road-network-dataset/tree/master/dataset.py): PyTorch class extending Dataset Class. Includes options to load from source images, pickle files, apply BFS heuristics and others.

## Usage
- Run `generate_toulouse_road_network_dataset.py` or sequentially run `generate_bins.py`, `generate_datapoints.py` and optionally `generate_image_arrays.py`.
- The dataset will be saved in `dataset/`


Find out more about this project in our [blog post](https://davide-belli.github.io/toulouse-road-network). 
Please cite [[1](#citation)] in your work when using this library in your experiments.

## Feedback
For questions and comments, feel free to contact [Davide Belli](mailto:davidebelli95@gmail.com).

## Citation
```
[1] Belli, Davide and Kipf, Thomas (2019). 
Image-Conditioned Graph Generation for Road Network Extraction. 
NeurIPS 2019 workshop on Graph Representation Learning.
```

BibTeX format:
```
@article{belli2019image,
  title={Image-Conditioned Graph Generation for Road Network Extraction},
  author={Belli, Davide and Kipf, Thomas},
  journal={NeurIPS 2019 workshop on Graph Representation Learning},
  year={2019}
}

```

## Copyright

Copyright © 2019 Davide Belli.

This project is distributed under the <a href="LICENSE">MIT license</a>. This was developed as part of a master thesis supervised by [Thomas Kipf](https://tkipf.github.io/) at the University of Amsterdam, and presented as a paper at the [Graph Representation Learning workshop in NeurIPS 2019](https://grlearning.github.io/papers/), Vancouver, Canada.
