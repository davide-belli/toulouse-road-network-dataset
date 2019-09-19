"""
Divides the original map in bins, to simplify and speed up the extraction of the datapoints (map tiles) in generate_datapoints.py
Optionally, compute some statistics over the number of nodes, edges and roads in the map.
The histogram of bin counts in function of the number of edges can be plotted with plot_bins_histogram()

All costants and parameters chosen for Toulouse Road Network dataset are defined in config.py

######################################################################################################

Statistics with step = 0.001:

There are 424029 points, 361793 segments, 62236 roads
There are 61363 bins with at least one road out of 123096 possible bins
The min, max, avg sizes of the bins excluded zero are: min:2 min_without_0: max:99 avg:8.764842361306016 num:53445
"""

import shapefile as shp  # Requires the pyshp package
from collections import defaultdict
import time
import pickle

from config import *

import matplotlib.pyplot as plt
plt.switch_backend('agg')


def find_bin(p_x, p_y):
    r"""
    Find the bin in which a point lies
    
    :param p_x: x-coordinate of the point (float)
    :param p_y: y-coordinate of the point (float)
    :return: the x and y indexes of the bin (int)
    """
    bin_x = int((p_x - GLOBAL_X_MIN) / STEP)
    bin_y = int((p_y - GLOBAL_Y_MIN) / STEP)
    
    assert bin_x >= 0
    assert bin_y >= 0
    
    return bin_x, bin_y


def get_bins(x_a, y_a, x_b, y_b):
    r"""
    Get the bins for which and edge may pass.
    This preprocessing returns rectangle of bins where the edge lies, but it does not mean that the edge lies in all
    the proposed bins.
    This pre-processing is mainly to speedup the following dataset generation, which will take care of
    considering exclusively the bins where the edge lies
    
    :param x_a: x-coordinate, first point (float)
    :param y_a: y-coordinate, first point (float)
    :param x_b: x-coordinate, second point (float)
    :param y_b: y-coordinate, second point (float)
    :return: a set of bins, among which all the ones where the edge lies
    """
    # find bins in which the extreme points lie
    idx_a, idx_b = find_bin(x_a, y_a), find_bin(x_b, y_b)
    res = {idx_a, idx_b}
    
    if abs(idx_a[0] - idx_b[0]) + abs(idx_a[1] - idx_b[1]) <= 1:
        # if the two bins are contiguous or the same, return
        return res
    else:
        # otherwise, add bins where the edge MAY be passing by.
        for x in range(idx_a[0], idx_b[0] + 1):
            for y in range(idx_a[1], idx_b[1] + 1):
                res.add((x, y))
    return res


def extract_bins(sf):
    r"""
    Divides the segments of roads into squared bins, to simplify further dataset generation
    
    :param sf: shapefile object imported from file
    :return bins_dict: dictionary of bins with size up to N_TOTAL_BINS (empty bins are not stored)
    :return n_points: number of points in the roads
    :return n_roads: number of roads in the shapefile
    """
    n_points, n_roads = 0, 0
    bins_dict = defaultdict(list)
    
    for count, shape in enumerate(sf.shapeRecords()):
        points = shape.shape.points[:]
        if len(points) < 2:
            continue
        
        for i in range(len(points) - 1):
            # four coordinates defining an edge
            x_a = points[i][0]
            y_a = points[i][1]
            x_b = points[i + 1][0]
            y_b = points[i + 1][1]
            
            # get all the bins for which this edge passes
            bins = get_bins(x_a, y_a, x_b, y_b)
            
            # add this edge to all the bins in which it passes
            for bin_key in bins:
                bins_dict[bin_key].append(((x_a, y_a, x_b, y_b), count))
        
        n_points += len(points)
        n_roads += 1
    
    return bins_dict, n_points, n_roads


def get_bins_lengths(bins_dict, max_len=100):
    r"""
    Get the list of bins lengths and print some statistics
    
    :param bins_dict: dictionary of bins, extracted through extract_bins()
    :param max_len: maximum lenght allowed. Default to 100, to avoid few outlier cases when visualizing the statistics
    :return: list of bins lengths
    """
    bin_lengths = []
    for key, value in bins_dict.items():
        if 1 < len(value) < max_len:
            bin_lengths.append(len(value))
    print(f"The min, max, avg sizes of the bins excluded zero are: min:{min(bin_lengths)} min_without_0: " +
          f"max:{max(bin_lengths)} avg:{sum(bin_lengths) / len(bin_lengths)} num:{len(bin_lengths)}")
    return bin_lengths


def plot_bins_histogram(bin_lengths, nbins=99):
    r"""
    Plots the histogram of bins in ./histogram_bins.png
    
    :param bin_lengths: list of bins lengths
    :param nbins: number of bins to use in the histogram plots
    """
    plt.figure()
    plt.hist(bin_lengths, density=True, bins=nbins)
    plt.title("Distribution of number of road segment per bin")
    plt.savefig("./plots/histogram_bins.png")


def main_generate_bins():
    r"""
    Run the main script for the generation of bins
    :return:
    """
    sf = shp.Reader("./raw/gis_osm_roads_07_1.shp")  # shapefile object, containing an iterable of shape objects
    start_time = time.time()
    bins_dict, n_points, n_roads = extract_bins(sf)  # extract bins
    n_segments = n_points - n_roads
    pickle.dump((bins_dict, (N_X_BINS, N_Y_BINS)), open(f"raw/bins.pickle", "wb"))  # save bins in pickle format
    print(f"Total time: {time.time() - start_time}s\n")
    
    print(f"There are {n_points} points, {n_segments} segments, {n_roads} roads")
    print(f"There are {len(bins_dict.keys())} bins with at least one road out of {N_TOTAL_BINS} possible bins")
    
    # optionally, get some statistics and plot a histogram of bins
    bins_lengths = get_bins_lengths(bins_dict)
    plot_bins_histogram(bins_lengths)
    
    
if __name__ == '__main__':
    main_generate_bins()

