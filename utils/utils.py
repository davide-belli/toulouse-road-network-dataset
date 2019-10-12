import numpy as np
import math
import os
import pickle

import matplotlib.pyplot as plt
plt.switch_backend('agg')

ROAD_COLORS = np.random.rand(62236, 3)  # colors used for each road_id among the 62236 roads in the source dataset
ROTATION_OFFSETS = [0, 90, 180, 270]  # rotation offsets for augmentation

"""
Classes definitions for different object types
"""

class Point(object):
    r"""
    A class to represent Point objects, defined by two float coordinates
    """
    def __init__(self, x, y=None):
        r"""
        Initilalizes the Point object through a tuple (x) or two coordinates (x, y)
        :param x: a tuple of coordinates or the x-coordinate of the point
        :param y: None if x is a tuple, otherwise the y-coordinate of the point
        """
        if y is None and len(x) == 2:
            y = x[1]
            x = x[0]
        assert isinstance(x, float)
        assert isinstance(y, float)
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        r"""
        Overrides the default Equals behavior
        Checks if the Point object is the same as another Point in terms of 2 coordinates. Sensitive to the ordering
        of the two coordinates
        
        :param other: the other Point to be compared to this Point object"""
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False
    
    def tuple(self):
        r"""
        Returns a tuple representation of the point.
        
        :return: the current point, represented as a tuple (x, y)
        """
        return self.x, self.y


class Bin(object):
    r"""
    A class to represent Bin objects, defined by two indexes of the bin
    """
    def __init__(self, x, y):
        r"""
        Initializes the Bin object by its two indexes. Must be intergers
        
        :param x: index over rowx
        :param y: index over columns
        """
        assert isinstance(x, int)
        assert isinstance(y, int)
        self.x = x
        self.y = y


class Line(object):
    r"""
    A class to represent Line objects, defined by two Point objects
    """
    def __init__(self, a, b):
        r"""
        Initializes the Line object by passing two Point objects or two tuple of point coordinates
        :param a:
        :param b:
        """
        if isinstance(a, Point) and isinstance(b, Point):
            self.a = a
            self.b = b
        elif isinstance(a, tuple) and isinstance(b, tuple):
            if len(a) == 2 and len(b) == 2:
                self.a = Point(a[0], a[1])
                self.b = Point(b[0], b[1])
        else:
            raise ValueError(f"Wrong parameters to Line init: {type(a)}, {type(b)}")
    
    def __eq__(self, other):
        r"""
        Overrides the default Equals behavior.
        Checks if the Line object is the same as another Line in terms of 4 extreme points. Sensitive to the ordering
        of the two Points
        :param other: the other Line to be compared to this Line object
        """
        if isinstance(other, Line):
            return self.a.x == other.a.x and self.a.y == other.a.y and \
                   self.b.x == other.b.x and self.b.y == other.b.y
        return False
    
    def length(self):
        r"""
        The length of the current line, computed with euclidean distance
        
        :return: the length of the current line
        """
        dxdy = np.array([self.a.x - self.b.x, self.a.y - self.b.y])
        return np.linalg.norm(dxdy)


class Square:
    r"""
    A class to represent Square objects, defined by the coordinates of Up, Down, Left and Right edges
    """
    def __init__(self, up, down, left, right):
        r"""
        Instantiate the Square by setting its 4 attributes
        :param up: y-coordinate of the top edge
        :param down: y-coordinate of the bottom edge
        :param left: x-coordinate of the left edge
        :param right: x-coordinate of the right edge
        """
        assert isinstance(up, float)
        assert isinstance(down, float)
        assert isinstance(left, float)
        assert isinstance(right, float)
        self.up = up
        self.down = down
        self.left = left
        self.right = right

"""
Utils functions for data processing
"""


def coefficients(l):
    r"""
    Computes coefficients a,b,c defining the implicit form of the line equation
    :param l: Line object
    :return: the three coefficient in the line equation
    """
    p1 = l.a
    p2 = l.b
    A = (p1.y - p2.y)
    B = (p2.x - p1.x)
    C = (p1.x * p2.y - p2.x * p1.y)
    return A, B, -C


def line_intersection(l1, l2, threshold=1e-10):
    r"""
    Detects the intersection between two line, up to a maximum threshold (to overcome inaccuracies in the
    source data and numerical instabilities from Python operations).

    :param line1: line object 1
    :param line2: line object 2
    :param threshold: maximum distance between any two points in the line to be considered intersecting
    :return: The intersecting point coordinates (tuple) if the intersection verifies, None otherwise
    """
    L1 = coefficients(l1)
    L2 = coefficients(l2)
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        
        if not (min(l1.a.x, l1.b.x) - threshold <= x <= max(l1.a.x, l1.b.x) + threshold and
                min(l1.a.y, l1.b.y) - threshold <= y <= max(l1.a.y, l1.b.y) + threshold):
            return None
        if not (min(l2.a.x, l2.b.x) - threshold <= x <= max(l2.a.x, l2.b.x) + threshold and
                min(l2.a.y, l2.b.y) - threshold <= y <= max(l2.a.y, l2.b.y) + threshold):
            return None
        
        return x, y
    else:
        return None


def euclidean_distance_points(pt1, pt2):
    r"""
    Get euclidean list between two points, either in Point format or float format

    :param pt1: the Point a
    :param pt2: the Point b
    :return: the euclidean distance between the two points
    """
    dx = pt1.x - pt2.x
    dy = pt1.y - pt2.y
    return math.sqrt(dx ** 2 + dy ** 2)


def euclidean_distance(ax, ay, bx=None, by=None):
    r"""
    Get euclidean list between two points, either in Point format or float format
    
    :param ax: if bx and by are None, the Point a, otherwise the x-coordinate of point a
    :param ay: if bx and by are None, the Point b, otherwise the y-coordinate of point a
    :param bx: None if Point representation is used, otherwise the x-coordinate of point b
    :param by: None if Point representation is used, otherwise the y-coordinate of point b
    :return: the euclidean distance between the two points
    """
    if bx is None and by is None:
        return euclidean_distance_points(ax, ay)
    
    dx = ax - bx
    dy = ay - by
    return math.sqrt(dx ** 2 + dy ** 2)


def get_nodes_from_lines(lines):
    r"""
    Starting from Line representation, get a node representation as two arrays of X and Y node coordinates.
    Removes duplicate nodes if any.

    :param lines: list of Line objects
    :return: two array of floats, containing respectively X and Y coordinates of the nodes
    """
    nodes = []
    for l in lines:
        nodes.append((l.a.x, l.a.y))
        nodes.append((l.b.x, l.b.y))
    nodes = np.array(list(set(nodes)))
    return nodes[:, 0], nodes[:, 1]


def generate_datapoint(id, id_original, nodes, edges, graph, bfs_edges, bfs_nodes, dfs_edges, dfs_nodes,
                       this_split, this_point, id_augment=None, ):
    r"""
    Generates a dictionary describing the current datapoint
    
    :param id: id of the current datapoint
    :param id_original: id of the original datapoint. It only differs from id if augmentation is used
    :param nodes: list of nodes
    :param edges: list of edges
    :param graph: adjacency list of the graph
    :param bfs_edges: bfs over the edges
    :param bfs_nodes: bfs over the nodes
    :param dfs_edges: dfs over the edges
    :param dfs_nodes: dfs over the ndoes
    :param this_split: the split to which this datapoint belong (0: train, 1: valid, 2: test, 3:augmented train)
    :param this_point: the coordinates of the center of the datapoint
    :param id_augment: type of augmentation used, None if no augmentation is used
    :return:
    """
    res = dict()
    res['id'] = id
    res['id_original'] = id_original
    res['nodes'] = nodes
    res['edges'] = edges
    res['graph'] = graph
    res['split'] = this_split
    res['n_nodes'] = len(nodes)
    res['n_edges'] = len(edges)
    res['coordinates'] = (this_point.x, this_point.y)
    
    # if augmented datapoint
    if this_split == 3:
        res['is_augment'] = True
        res['flip'] = id_augment // 4
        res['rotation'] = ROTATION_OFFSETS[id_augment % 4]
    else:
        res['is_augment'] = False
    
    res['bfs_edges'] = [[e[0][0], e[0][1], e[1][0], e[1][1]] for e in bfs_edges]
    res['dfs_edges'] = [[e[0][0], e[0][1], e[1][0], e[1][1]] for e in dfs_edges]
    
    adj_rows, points = map(list, zip(*bfs_nodes))
    points = [list(p) for p in points]
    res['bfs_nodes_adj'] = adj_rows
    res['bfs_nodes_points'] = points
    
    adj_rows, points = map(list, zip(*dfs_nodes))
    points = [list(p) for p in points]
    res['dfs_nodes_adj'] = adj_rows
    res['dfs_nodes_points'] = points
    
    assert len(res['dfs_nodes_points']) == len(res['bfs_nodes_points']) == len(nodes) + 1
    assert len(res['dfs_nodes_adj']) == len(res['bfs_nodes_adj']) == len(nodes) + 1
    assert len(res['dfs_edges']) == len(res['bfs_edges'])
    
    return res


"""
Data saving and plotting utilities
"""


def ensure_dir(dir_path):
    r"""
    Check if a directory exists, if not, create it
    
    :param dir_path: path of the directory to be checked
    :return:
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def full_frame(plt, width=0.64, height=0.64):
    r"""
    Generates a particular tight layout for Pyplot plots
    
    :param plt: pyplot
    :param width: width, default is 64 pixels
    :param height: height, default is 64 pixels
    :return:
    """
    import matplotlib as mpl
    mpl.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width, height)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)


def save_image_by_lines(square, lines, path, show_square=False, id_roads=None, plot_nodes=False, step=0.001):
    r"""
    Plot and save image before preprocessing (using raw lines data)
    
    :param square: the coordinates of the current square
    :param lines: lines in the current datapoint, before preprocessing
    :param path: path for saving the data
    :param show_square: boolean, whether to plot the square defining the region and coordinates of the current datapoint
    :param id_roads: passes the id of roads to be plotted
    :param plot_nodes: boolean, plot the nodes in red or
    :param step: step used for data generation, default is step=STEP=0.001
    """
    if id_roads is None:
        colors = np.random.rand(len(lines), 3)
        # colors = ["b"] * len(lines)  # plot lines in black
    else:
        colors = [ROAD_COLORS[id] for id in id_roads]
    
    plt.xlim(square.left, square.right)
    plt.ylim(square.down, square.up)
    for i, line in enumerate(lines):
        plt.plot([line.a.x, line.b.x], [line.a.y, line.b.y], c=colors[i])
    
    if show_square:
        plt.xlim(square.left - step / 4, square.right + step / 4)
        plt.ylim(square.down - step / 4, square.up + step / 4)
        plt.plot([square.left, square.left, square.right, square.right, square.left],
                 [square.down, square.up, square.up, square.down, square.down])
    
    if plot_nodes:
        nodes_x, nodes_y = get_nodes_from_lines(lines)
        plt.plot(nodes_x, nodes_y, 'ro', alpha=0.5)
    # plt.show()
    plt.savefig(path)
    plt.clf()


def save_image_by_nodes_edges(square, nodes, edges, path, show_square=False, plot_nodes=False, step=0.001):
    r"""
    Plot and save image after preprocessing, possibly using colored edges, nodes, and others

    :param square: the coordinates of the current Square
    :param nodes: the nodes in the graph, list of (node_x, node_y)
    :param edges: the edges in the graph, list of (node_id_a, node_id_b, road_id)
    :param path: path for saving the data
    :param show_square: boolean, whether to plot the square defining the region and coordinates of the current datapoint
    :param plot_nodes: boolean, plot the nodes in red or
    :param step: step used for data generation, default is step=STEP=0.001
    """
    plt.xlim(square.left, square.right)
    plt.ylim(square.down, square.up)
    used_nodes = set()
    for i, edge in enumerate(edges):
        a, b, id = edge
        node_a, node_b = (nodes[a][0], nodes[a][1]), (nodes[b][0], nodes[b][1])
        used_nodes.add(node_a)
        used_nodes.add(node_b)
        plt.plot((nodes[a][0], nodes[b][0]), (nodes[a][1], nodes[b][1]), c=ROAD_COLORS[id])
    
    if show_square:
        plt.xlim(square.left - step / 4, square.right + step / 4)
        plt.ylim(square.down - step / 4, square.up + step / 4)
        plt.plot([square.left, square.left, square.right, square.right, square.left],
                 [square.down, square.up, square.up, square.down, square.down])
    
    if plot_nodes:
        nodes_x, nodes_y = zip(*list(used_nodes))
        plt.plot(nodes_x, nodes_y, 'ro', alpha=0.5)
    # plt.show()
    plt.savefig(path)
    plt.clf()
    

def save_image(nodes, edges, path):
    r"""
    Plot and save images after preprocessing in the format of semantic segmentation to be used in the generated dataset
    (i.e. black graphs on white background, grayscale images, no borders, no nodes plotted and 64x64 dimension)

    :param nodes: the nodes in the graph, list of (node_x, node_y)
    :param edges: the edges in the graph, list of (node_id_a, node_id_b, road_id)
    :param path: path for saving the data
    """
    square = Square(1., -1., -1., 1.)
    full_frame(plt)
    plt.xlim(square.left, square.right)
    plt.ylim(square.down, square.up)
    
    for i, edge in enumerate(edges):
        a, b, id = edge
        plt.plot((nodes[a][0], nodes[b][0]), (nodes[a][1], nodes[b][1]), c='k', linewidth=2)
    
    plt.savefig(path)
    plt.clf()
    plt.close('all')


def save_image_bfs(square, edges, path, plot_nodes=False):
    r"""
    Plot and save a sequence of images showing the bfs (or dfs) for the sequential generation of the graph

    :param square: the Square describing the current datapoint.
    :param edges: the edges in the graph, list of (node_id_a, node_id_b, road_id)
    :param path: path for saving the sequence of images
    :param plot_nodes: whether or not to plot nodes in red
    """
    plt.xlim(square.left, square.right)
    plt.ylim(square.down, square.up)
    used_nodes = set()
    for i, edge in enumerate(edges):
        (ax, ay), (bx, by) = edge
        if ax < -2:
            continue
        node_a, node_b = (ax, ay), (bx, by)
        used_nodes.add(node_a)
        used_nodes.add(node_b)
        plt.plot((ax, bx), (ay, by))
    if plot_nodes and used_nodes:
        nodes_x, nodes_y = zip(*list(used_nodes))
        plt.plot(nodes_x, nodes_y, 'ro', alpha=0.5)
    # plt.show()
    plt.savefig(path)
    plt.clf()


def save_dataset(datasets, paths):
    r"""
    Save the generated splits of the dataset in different pickle files
    
    :param datasets: dataset (dictionary containing the different splits)
    :param paths: paths to save each split
    :return:
    """
    for i in range(len(datasets)):
        f = open(paths[i], 'wb')
        pickle.dump(datasets[i], f)

