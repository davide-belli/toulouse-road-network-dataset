"""
Generates the three splits of the dataset.
Each data point is represented by its graph descriptors  (node, edges, coordinates, id), the linearization of the graph
(sequence of nodes and edges under the BFS or DFS canonical ordering), and images showing the semantic segmentation
of the map tile.

In this file, the main script for the dataset generation uses the bins.pickle previously generated through
generate_bins.py and a set of utility functions defined here and in utils.py.
The main script iteratively generate all the datapoints for the different splits, performing preprocessing, filtering
and linearization of the graph (BFS or DFS), finally saving the resulting images and pickle files.
Optionally, augmentation by rotation and flip can be performed and the augmented training set saved in a separate file.
Uncommenting some lines in the main script will allow for plotting the graphs during the dataset generation, visualizing
 he changes introduced by preprocessing, augmentation and examples of BFS/DFS orderings.

All costants and parameters chosen for Toulouse Road Network dataset are defined in config.py

######################################################################################################
"""

from collections import defaultdict, Counter
import time
import operator
import copy

from config import *
from utils import line_intersection, euclidean_distance, generate_datapoint
from utils import save_image_by_lines, save_image_by_nodes_edges, save_image_bfs, save_image, save_dataset
from utils import Point, Bin, Square, Line


def check_split(cx, cy):
    """
    Check to which split does a data point belong.
    
    :param cx: x_center coordinate (float)
    :param cy: y_center coordinate (float)
    :return:
        -1 to be discarded (overlap with test or valid regions)
        0 train
        1 valid
        2 test
    """
    if test_region["x"] - step_region["x"] < cx < test_region["x"] + step_region["x"] or \
            test_region["y"] - step_region["y"] < cy < test_region["y"] + step_region["y"]:
        return 2
    elif test_region["x"] - step_region["x"] - STEP < cx < test_region["x"] + step_region["x"] + STEP or \
            test_region["y"] - step_region["y"] - STEP < cy < test_region["y"] + step_region["y"] + STEP:
        return -1
    elif valid_region["x"] - step_region["x"] < cx < valid_region["x"] + step_region["x"] or \
            valid_region["y"] - step_region["y"] < cy < valid_region["y"] + step_region["y"]:
        return 1
    elif valid_region["x"] - step_region["x"] - STEP < cx < valid_region["x"] + step_region["x"] + STEP or \
            valid_region["y"] - step_region["y"] - STEP < cy < valid_region["y"] + step_region["y"] + STEP:
        return -1
    else:
        return 0


def to_x_coordinate(x):
    r"""
    Convert bin x-index to x-coordinate
    
    :param x: bin index (int)
    :return: x-coordinate (float)
    """
    return GLOBAL_X_MIN + STEP * x


def to_y_coordinate(y):
    r"""
    Convert bin y-index to y-coordinate

    :param y: bin index (int)
    :return: y-coordinate (float)
    """
    return GLOBAL_Y_MIN + STEP * y


def to_x_idx(x):
    r"""
    Convert x-coordinate to bin x-index
    
    :param x: x-coordinate (float)
    :return: bin x-index (int)
    """
    return int((x - GLOBAL_X_MIN) / STEP)


def to_y_idx(y):
    r"""
    Convert y-coordinate to bin y-index

    :param y: y-coordinate (float)
    :return: bin y-index (int)
    """
    return int((y - GLOBAL_Y_MIN) / STEP)


def get_possible_lines(bin):
    r"""
    Get all possible roads for this point (i.e. all roads from the 3x3 bins surrounding the current bin)
    
    :param bin: bin object representing the center bin for this datapoints
    :return res: list of Line objects
    """
    res_set = set
    for idx_x in [int(bin.x) - 1, int(bin.x), int(bin.x) + 1]:
        for idx_y in [int(bin.y) - 1, int(bin.y), int(bin.y) + 1]:
            res_set = res_set.union(set(bins_dict[(idx_x, idx_y)]))
    res = []
    for line, id in res_set:
        line = Line((line[0], line[1]), (line[2], line[3]))
        res.append((line, id))
        
    return res


def is_point_in_square(square, point):
    r"""
    Check if a point lies in a square.
    
    :param square: input square
    :param point: input point
    :return: True if points lies in square, False otherwise
    """
    return square.left < point.x < square.right and square.down < point.y < square.up


def is_line_in_square(square, line):
    r"""
    Check if a line lies in a square
    
    :param square: input square
    :param point: input line
    :return: True if line lies in square, False otherwise
    """
    return is_point_in_square(square, line.a) and \
           is_point_in_square(square, line.b)


def find_border_intersection(square, segment):
    r"""
    Find all intersection between the squares' edges and a line segment
    
    :param square: the current square
    :param segment: a line segment
    :return: a list of points (if any)
    """
    points = []
    # 4 lines for the edges of a square
    edges = [
        [square.left, square.down, square.right, square.down],
        [square.left, square.up, square.right, square.up],
        [square.left, square.down, square.left, square.up],
        [square.right, square.down, square.right, square.up]
    ]
    # find all intersection between the edges and the segment
    for edge in edges:
        # point = line_intersection(edge, [segment.a.x, segment.a.y, segment.b.x, segment.b.y])
        point = line_intersection(Line((edge[0], edge[1]), (edge[2], edge[3])), segment)
        if point is not None:
            points.append(point)
    return points


def get_valid_line(square, line):
    r"""
    Check if a line is valid and handle intersection with square borders.
    
    :param square: the square representing the current data point
    :param line: the proposal, original line
    :return: a Line object if valid, otherwise None
    """
    # case 1: both points in the square
    if is_line_in_square(square, line):
        return line
    # case 2: both points are outside the square
    elif not is_point_in_square(square, line.a) and not is_point_in_square(square, line.b):
        # find intersection with borders (generally it will be either 0 or 2, but it can be 1 if an extreme of the line
        # segment lies on the edge and the other extreme is out of the square. in this case, we discard the line)
        points = find_border_intersection(square, line)
        if len(points) == 0:
            return None
        elif len(points) == 2:
            return Line(points[0], points[1])
        elif len(points) == 1:
            return None
        else:
            raise Exception("This should never happen!")
            # raise Exception("Both points are outside the square but 1 or mor than 2 edges are crossed!")
    # case 3: one point a is not in the square, looking for intersection
    else:
        points = find_border_intersection(square, line)
        if not len(points) == 1:
            return None
        # return a Line with the point inside the square and the point on the square's edge interesecting the line
        if is_point_in_square(square, line.a):
            return Line(line.a, Point(points[0]))
        elif is_point_in_square(square, line.b):
            return Line(line.b, Point(points[0]))
        else:
            raise Exception("This should never happen!")


def get_valid_lines(lines, this_square):
    r"""
    Wrapper for get_valid_line()
    Get valid lines, i.e. only the lines lying in the current square.
    Also, handle lines intersecting the border of the square by generating a new node on the edge of the square.
    
    :param lines: all possible lines from surrounding bins
    :param this_square: the square representing the current proposal  datapoint
    :return valid_lines: only and all the lines in the current square
    :return valid_id_roads: the id of the original roads to which each line (edge) belongs to
    """
    valid_lines = []
    valid_id_roads = []
    
    for this_line, id_road in lines:
        # get line if is valid, handle intersection with square borders
        line = get_valid_line(this_square, this_line)
        if line is not None:
            valid_lines.append(line)
            valid_id_roads.append(id_road)
    
    return valid_lines, valid_id_roads


def detect_intersection(line1, line2, threshold=STEP / 100):
    r"""
    Detects the intersection between two line, up to a maximum threshold (to overcome inaccuracies in the
    source data and numerical instabilities from Python operations).
    Wrapper for the general method line_intersection()
    
    :param line1: line object 1
    :param line2: line object 2
    :param threshold: maximum distance between any two points in the line to be considered intersecting
    :return: The intersecting point coordinates (Point object) if the intersection verifies, None otherwise
    """
    p = line_intersection(line1, line2, threshold=threshold)
    return None if p is None else Point(p)


def join_intersection(line, point, threshold=STEP/50):
    r"""
    Check if the two newly generated lines from intersection are long enough. Return the one that are long enough.
    We threshold the minimum length of the segment to prevent noise or irrepresentability in the low-resolution
    semantic segmentation.
    
    :param line: input line
    :param point: intersection point
    :param threshold: minimum length for a new line to be accepted
    :return: new accepted lines after intersection
    """
    new_lines = []
    for p in [line.a, line.b]:
        dist = euclidean_distance(p, point)
        if dist > threshold:
            new_lines.append(Line(p, point))
    return new_lines


def handle_crossings(lines, id_roads):
    r"""
    Handle intersection between road segments by substituting the segments with new lines starting in the commong
    crossing points. Takes care of not including tiny segments that may result by inaccuracies in the source
    files and numerical inaccuracies in Python. Handles the incremental change in the line lists caused using a
    enqueue/dequeue approach.
    
    :param lines: lines in this square
    :param id_roads: id of the roads in this square
    :return lines: update list of lines
    :return id_roads: updated list of id_roads accounting for the newly inserted lines
    """
    for i, line1 in enumerate(lines):
        flag_continue = False
        for j, line2 in enumerate(lines):
            # find possible intersection between the two lines
            intersection_point = detect_intersection(line1, line2)
            if intersection_point is not None:
                # find the new proposed segments by splitting the original lines in the intersection point
                new_lines1 = join_intersection(line1, intersection_point)
                new_lines2 = join_intersection(line2, intersection_point)
                if len(new_lines2) == 1:
                    # if it's just a line update, update and go on
                    lines[j] = new_lines2[0]
                elif len(new_lines2) == 2:
                    # otherwise, two new lines, we need to substitute them to the previous line2
                    del lines[j]
                    id = id_roads.pop(j)
                    lines.append(new_lines2[0])
                    lines.append(new_lines2[1])
                    id_roads.extend([id, id])
                else:
                    # Both resulting lines are too short, so the image is discarded
                    return [], []
                if len(new_lines1) == 1:
                    # if it's just a line update, update and go on
                    # also, update the current line, as it will be used again to compare it with other line2
                    lines[i] = new_lines1[0]
                    line1 = new_lines1[0]
                elif len(new_lines1) == 2:
                    # otherwise, two new lines, we need to use them now instead, and stop using the old line1
                    del lines[i]
                    id = id_roads.pop(i)
                    lines.append(new_lines1[0])
                    lines.append(new_lines1[1])
                    id_roads.extend([id, id])
                    flag_continue = True
                    break
                else:
                    # Both resulting lines are too short, so the image is discarded
                    return [], []
        
        if flag_continue:
            continue
    
    return lines, id_roads


def to_nodes_edges(lines, id_roads):
    r"""
    Change data format to nodes: (x, y), and edges: (node_a_id, node_b_id, road_id).
    The node ids correspond to their idx in the node list.
    
    :param lines: list of edges in the graph
    :param id_roads: list of road id for every line in lines
    :return nodes: list of (x, y) coordinates, float
    :return edges: list of (node_a_id, node_b_id, road_id) for each edge in the graph, integers
    """
    nodes = set()
    edges = []
    for l in lines:
        nodes = nodes.union({l.a.tuple(), l.b.tuple()})
    nodes = list(nodes)
    
    for i, l in enumerate(lines):
        a = nodes.index(l.a.tuple())
        b = nodes.index(l.b.tuple())
        id_road = id_roads[i]
        edges.append([a, b, id_road])
    
    return nodes, edges


def merge_duplicate_nodes(nodes, edges, distance_threshold=DISTANCE_THRESHOLD):
    r"""
    Merge duplicate nodes and modify edges accordingly. Nodes are considered coincident if closer than a threshold
    
    :param nodes: list of nodes (x, y)
    :param edges: list of edges (node_a_id, node_b_id, road_id)
    :param distance_threshold: maximum distance for points to be considered coincident.
    :return nodes: list of nodes (x, y) after cleaning
    :return edges: list of edges (node_a_id, node_b_id, road_id) after cleaning
    :return len(to_del): how many nodes have been deleted in the merging procedure
    """
    to_del = set()  # nodes to be deleted (duplicates)
    to_del_edges = []  # edges to be deleted (duplicates)
    replace_destination = dict()  # to which node a duplicate will be substituted. used to handle recurrent dependencies
    for i, (ax, ay) in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            bx, by = nodes[j]
            if euclidean_distance(ax, ay, bx, by) < distance_threshold:
                # check if a coincides with b
                replace_destination[j] = i
                for k, edge in enumerate(edges):
                    # update all edges that have the removed duplicate node to point to the substitutive one
                    a, b, id_road = edge
                    a = a if a != j else i
                    b = b if b != j else i
                    edges[k] = [a, b, id_road]
                to_del.add(j)

    to_del = list(to_del)
    old_nodes = nodes[:]  # make a copy of nodes, for the following self-edge removal
    
    # simplify recursive dependencies in the destinations for the replacemente
    for k in replace_destination.keys():
        while replace_destination[k] in replace_destination.keys():
            replace_destination[k] = replace_destination[replace_destination[k]]
    
    # delete duplicate nodes starting from the end of the list to not break indexing in the iteration
    for i in reversed(sorted(to_del)):
        del nodes[i]
        
    # delete self-edges (we found some self-edges from point to itself. This removes this self-edges)
    for i in range(len(edges)):
        a, b, id_road = edges[i]
        a = nodes.index(old_nodes[a]) if a not in replace_destination.keys() else nodes.index(old_nodes[replace_destination[a]])
        b = nodes.index(old_nodes[b]) if b not in replace_destination.keys() else nodes.index(old_nodes[replace_destination[b]])
        if a == b:
            to_del_edges.append(i)
        edges[i] = [a, b, id_road]

    # delete duplicate edges starting from the end of the list to not break indexing in the iteration
    if len(edges) > 1:
        for i in reversed(to_del_edges):
            del edges[i]
    
    return nodes, edges, len(to_del)


def compute_slope(p1, p2):
    r"""
    Compute the slope of the line between two points
    
    :param p1: point a (tuple)
    :param p2: point b (tuple)
    :return slope: the slope of the edge between the two points, in degrees
    """
    delta = (p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-10)  # avoid division by zero
    slope = np.arctan(delta)
    slope = np.rad2deg(slope)
    return slope


def is_mergeable(candidate_edges, nodes, alpha_threshold=ALPHA_THRESHOLD):
    r"""
    Returns true if the difference in slope between the two edges is less than alpha_threshold (in degree)
    
    :param candidate_edges: Two consecutive edges
    :param nodes: list of nodes in the graph
    :param alpha_threshold: threshold under which two consecutive edges are merged into a single straight lines
    :return: True if the angle is belove threshold
    """
    angles = []
    for edge in candidate_edges:
        a = nodes[edge[0]]
        b = nodes[edge[1]]
        slope = compute_slope(a, b)
        angles.append(slope)
    
    return abs(angles[0] - angles[1]) < alpha_threshold


def merge_straight_lines(nodes, edges):
    r"""
    Finds and merge all consecutive edges in the graph that are "almost straight lines". More precisely, for all nodes
     of degree 2 compares the slope of the two consecutive, incident edges. If the difference in slope
    (or incident angle) in degrees is below a threshold, merge the edges by substituting them with a unique edge
    connecting the two points not shared by the two edges.
    
    :param nodes: list of nodes in the graph
    :param edges: list of edges in the graph
    :return nodes: list of nodes in the graph after merging straight lines
    :return edges: list of edges in the graph after merging straight lines
    :return len(to_del): number of deleted nodes after merging
    """
    to_del = []
    
    for i in range(len(nodes)):
        incoming_edges = []
        # find all edges incoming in a node
        for edge in edges:
            if i in edge[:2]:
                incoming_edges.append(edge)
        # if the incoming edges are two and mergebale according to alpha_threshold, merge them and update the edge list
        if len(incoming_edges) == 2:
            if is_mergeable(incoming_edges, nodes):
                # if there are two edges that are the same, remove one of them and continue
                if set(incoming_edges[0][:2]) == set(incoming_edges[1][:2]):
                    edges.remove(incoming_edges[0])
                    continue
                to_del.append(i)  # add the middle node to the list of nodes to be removed
                edges.remove(incoming_edges[0])
                edges.remove(incoming_edges[1])
                # the new edge connects the two nodes that are not node shared
                new_edge = list({incoming_edges[0][0], incoming_edges[0][1], incoming_edges[1][0],
                                 incoming_edges[1][1]} - {i})
                new_edge.append(incoming_edges[0][-1])  # add the id_road of one of the two merged edges
                edges.append(new_edge)
    
    # delete duplicate nodes starting from the end of the list to not break indexing in the iteration
    old_nodes = nodes[:]
    for i in reversed(to_del):
        del nodes[i]

    # update the node indexing used in the edge list after the changes in the list of nodes
    for i in range(len(edges)):
        a, b, id_road = edges[i]
        a = nodes.index(old_nodes[a])
        b = nodes.index(old_nodes[b])
        edges[i] = [a, b, id_road]
    
    return nodes, edges, len(to_del)


def normalize_coordinates(nodes, center_coordinates):
    r"""
    Shift original node coordinates to be in [-1,+1]
    
    :param nodes: all nodes in the current square (datapoint)
    :param center_coordinates: the center of the current square in the real coordinates
    :return nodes: the normalized list of nodes
    """
    for i, (x, y) in enumerate(nodes):
        nodes[i] = ((x - center_coordinates.x) * 2 / STEP, (y - center_coordinates.y) * 2 / STEP)
    return nodes


def augment(nodes):
    r"""
    Generate augmented versions of node lists.
    Augment the nodes in the graph using rotation (90 degrees) and flip combinations.
    
    :param nodes: list of nodes
    :return nodes : list of list of nodes. the external list has length 8 (4 rotations + 2 flip combinations)
    """
    nodes_list = [[] for i in range(8)]
    for n in nodes:
        x = n[0]
        y = n[1]
        nodes_list[0].append((x, y))
        nodes_list[1].append((-y, x))
        nodes_list[2].append((-x, -y))
        nodes_list[3].append((y, -x))
        
        x = - n[0]
        nodes_list[4].append((x, y))
        nodes_list[5].append((-y, x))
        nodes_list[6].append((-x, -y))
        nodes_list[7].append((y, -x))
    
    return nodes_list


def edges_to_adjacency_lists(edges):
    r"""
    Generate a representation of the edges as adjacency lists
    
    :param edges: edges represented as list of tuples (node_a, node_b, id_road)
    :return adj_list: edges represented as adj_list, namely a dict of lists (node_a -> node_b, node_c, ...)
    """
    adj_lists = defaultdict(list)
    for a, b, _ in edges:
        adj_lists[a].append(b)
        adj_lists[b].append(a)
    return adj_lists


def clockwise_angle_between(p1, p2):
    r"""
    Computes the clockwise angle between the incoming edge and the outcoming edges
    
    :param p1: (deltax, deltay) for the incoming edge
    :param p2: (deltax, deltay) for the outcoming edge
    :return: clockwise angle inbetween
    """
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def order_by_angle(current_node, parent, neighbors, nodes):
    r"""
    Return the list of neighbor nodes (or edges going to those nodes) ordered by the angle of the outcoming edge with
    respect to the angle of the incoming edge.
    
    :param current_node: node where the branching happens
    :param parent: other node belonging to the incoming edge
    :param neighbors: all the n nodes belonging to the n outcoming edge from the current_node
    :param nodes: list of node coordinates in the graph
    :return: neighbors, sorted according to their angle with respect to the incoming edge from 'parent' node
    """
    res = []
    ax1, ay1 = bx1, by1 = nodes[current_node]
    # if we are at the beginning of the connected component, we assume a horizontal incoming edge from the left of
    # the current_node (px -1, py)
    if parent == -1:
        ax2, ay2 = ax1 - 1, ay1
    else:
        ax2, ay2 = nodes[parent]
    # compute deltax, deltay for incoming edge
    ax = ax2 - ax1
    ay = ay2 - ay1
    
    # compute angle between incoming edge and every outcoming edge
    for n_id in neighbors:
        bx2, by2 = nodes[n_id]
        bx = bx2 - bx1
        by = by2 - by1
        angle = clockwise_angle_between((ax, ay), (bx, by))  # compute angle
        res.append([n_id, angle])
    res.sort(key=operator.itemgetter(-1))  # sort the outcoming edges by increasing angle
    return [x[0] for x in res]


def generate_bfs(nodes, graph):
    r"""
    Computes node-wise and edge-wise BFS ordering over the graph. Break ties by choosing top(-left) node at beginning
    or after a connected component has been visited. Ties in branching edges are broken by ordering the outcoming edges
    from a node clockwise with respect to the incoming edge (the last visited node in the BFS).
    An edge-wise BFS is defined as a list of (node_a, node_b) edges in the sequence.
    A node-wise BFS is defined as a list of (adjacency list, node_a) nodes in the sequence
    
    :param nodes: nodes in the graph
    :param graph: adjacency lists for the edges
    :return bfs_nodes: bfs node-wise over the graph
    :return bfs_edges: bfs edge-wise over the graph
    """
    nodes_with_id = [(i, node[0], node[1]) for i, node in enumerate(nodes)]  # add idx to every node
    nodes_sorted = sorted(nodes_with_id, key=lambda x: (-x[2], x[1]))  # sort top to bottom and then left to right
    visited, visited_edges, queue = set(), set(), []  # queue of nodes to visit in the bfs (node, parent_node)
    bfs_edges, bfs_nodes, bfs_nodes_ids = [], [], []
    empty_adj_row = [0] * (MAX_NUM_NODES + 1)  # +1 for the start of sequence node
    prev_adj_row = empty_adj_row[:]
    # Used respectively as first node in a connected component for each connected component in the bfs_edges and bfs_
    # nodes. We arbitrarily choose to set them to zeros as SOS token for the recurrent generation.
    nodes.append((0., 0.))  # first node in the connected component for bfs_edges
    prev_point = (0., 0.)  # first node in the graph for bfs_nodes
    bfs_nodes.append((prev_adj_row[:], prev_point))
    
    for (id, x, y) in nodes_sorted:
        if id not in visited:
            queue = [(id, -1)]  # we are adding the first node of the connected component, which has no parent (-1)
        while queue:
            current_vertex, parent = queue.pop(0)
            # add to bfs_edges if unvisited edge
            if (parent, current_vertex) not in visited_edges:
                visited_edges.add((current_vertex, parent))
                visited_edges.add((parent, current_vertex))
                bfs_edges.append(
                    (
                        (nodes[parent][0], nodes[parent][1]),
                        (nodes[current_vertex][0], nodes[current_vertex][1])
                    )
                )
            
            # add to bfs_nodes if unvisited node
            if current_vertex not in visited:
                # add to bfs_nodes, as output, the new output
                prev_adj_row = empty_adj_row[:]
                prev_point = nodes[current_vertex]
                # fill in the adjacency vector of the current node by looking back in the bfs
                for j, j_id in enumerate(reversed(bfs_nodes_ids)):
                    if j_id in graph[current_vertex]:
                        prev_adj_row[j] = 1
                bfs_nodes.append((prev_adj_row[:], prev_point))
                # list of nodes added in the bfs
                bfs_nodes_ids.append(current_vertex)
                
                visited.add(current_vertex)  # we don't want to re add the previous edge
                # finally add the outcoming edges (if any), ordered by angle wrt incoming edge
                neighbors = list(set(graph[current_vertex]) - {parent})
                if neighbors:
                    neighbors = order_by_angle(current_vertex, parent, neighbors, nodes)
                    # add neighbors and specify the parent (current vertex)
                    for neighbor in neighbors:
                        queue.append((neighbor, current_vertex))
    
    return bfs_nodes, bfs_edges


def generate_dfs(nodes, graph):
    r"""
    Computes node-wise and edge-wise DFS ordering over the graph. Break ties by choosing top(-left) node at beginning
    or after a connected component has been visited. Ties in branching edges are broken by ordering the outcoming edges
    from a node cloclwise with respect to the incoming edge (the last visited node in the DFS).
    An edge-wise DFS is defined as a list of (node_a, node_b) edges in the sequence.
    A node-wise DFS is defined as a list of (adjacency list, node_a) nodes in the sequence

    :param nodes: nodes in the graph
    :param graph: adjacency lists for the edges
    :return dfs_nodes: dfs node-wise over the graph
    :return dfs_edges: dfs edge-wise over the graph
    """
    nodes_with_id = [(i, node[0], node[1]) for i, node in enumerate(nodes)]  # add idx to every node
    nodes_sorted = sorted(nodes_with_id, key=lambda x: (-x[2], x[1]))  # sort top to bottom and then left to right
    visited, visited_edges, queue = set(), set(), []
    dfs_edges, dfs_nodes, dfs_nodes_ids = [], [], []
    empty_adj_row = [0] * (MAX_NUM_NODES + 1)
    prev_adj_row = empty_adj_row[:]
    # Used respectively as first node in a connected component for each connected component in the dfs_edges and dfs_
    # nodes. We arbitrarily choose to set them to zeros as SOS token for the recurrent generation.
    nodes.append((0., 0.))  # first node in the connected component for dfs_edges
    prev_point = (0., 0.)  # first node in the graph for dfs_nodes
    dfs_nodes.append((prev_adj_row[:], prev_point))
    
    for (id, x, y) in nodes_sorted:
        if id not in visited:
            queue = [(id, -1)]  # add a node as new connected component, so with no parent node
        while queue:
            current_vertex, parent = queue.pop()
            # add to dfs_edges
            if (parent, current_vertex) not in visited_edges:
                visited_edges.add((current_vertex, parent))
                visited_edges.add((parent, current_vertex))
                dfs_edges.append(
                    (
                        (nodes[parent][0], nodes[parent][1]),
                        (nodes[current_vertex][0], nodes[current_vertex][1])
                    )
                )
            if current_vertex not in visited:
                # add to dfs_nodes, as output, the new output
                prev_adj_row = empty_adj_row[:]
                prev_point = nodes[current_vertex]
                for j, j_id in enumerate(reversed(dfs_nodes_ids)):
                    if j_id in graph[current_vertex]:
                        prev_adj_row[j] = 1
                dfs_nodes.append((prev_adj_row[:], prev_point))
                # list of nodes added in the bfs
                dfs_nodes_ids.append(current_vertex)
                
                visited.add(current_vertex)
                # we don't want to re add the previous edge
                neighbors = list(set(graph[current_vertex]) - {parent})
                if neighbors:
                    neighbors = order_by_angle(current_vertex, parent, neighbors, nodes)
                    # add neighbors and specify the parent (current vertex)
                    for neighbor in reversed(neighbors):
                        queue.append((neighbor, current_vertex))
    
    return dfs_nodes, dfs_edges


def main_generate_datapoints():
    r"""
    Run the main script to generate the dataset. The output of generate_bins.py should be existing in ./raw/bins.pickle
    :return:
    """
    start_time = time.time()
    datasets = [defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)]
    bins_dict, (N_X_BINS, N_Y_BINS) = pickle.load(open(f"raw/bins.pickle", "rb"))  # load pre-generated bins of road segment
    tot_count = 0  # number of datapoints proposal explored
    count = 0  # number of accepted datapoints
    count_split = [0, 0, 0]  # number of accepted datapoints per split
    count_augment = 0  # number of accepted datapoints when using rotate and flip augmentation
    this_square = Square(0., 0., 0., 0.)  # square object representing the current datapoint's coordinates
    
    # to generate the default dataset we use translation augmentation of factor x4 for both x and y dimensions
    for cx in np.arange(GLOBAL_MIN.x + STEP, GLOBAL_MAX.x - STEP, STEP / 4):
        
        if tot_count % 100 == 0:
            # log progress every now and then
            print(f"{int(100 * tot_count / N_POSSIBLE_DATAPOINTS)}%,"
                  f" {int(time.time() - start_time)}s, x={cx}/{GLOBAL_X_MAX} \t accepted:{count} / proposed:{tot_count}")
        
        # approximate to 5 decimals to avoid numerical instabilities,
        # convert coordinate to bin index and update Square representation
        cx = round(cx, 5)
        cx_idx = to_x_idx(cx)
        this_square.left = cx - (STEP / 2)
        this_square.right = cx + (STEP / 2)
        
        for cy in np.arange(GLOBAL_Y_MIN + STEP, GLOBAL_Y_MAX - STEP, STEP / 4):
            # get split for this map tile
            this_split = check_split(cx, cy)
            
            # check if square is in the conflict area between two splits, then skip datapoint
            if this_split == -1:
                continue
            
            # approximate to 5 decimals to avoid numerical instabilities,
            # convert coordinate to bin index and update Square representation
            tot_count += 1
            cy = round(cy, 5)
            cy_idx = to_y_idx(cy)
            this_square.up = cy + (STEP / 2)
            this_square.down = cy - (STEP / 2)
            
            this_point = Point(cx, cy)
            this_bin = Bin(cx_idx, cy_idx)
            
            # get all lines that could possibly intersect the current square from the bin where this data point lies
            lines = get_possible_lines(this_bin)
            
            # get only lines in this square, and handle lines intersecting the borders
            valid_lines, valid_id_roads = get_valid_lines(lines, this_square)
            
            # handle intersections by substituting intersecting lines with new lines terminating in the intersection
            # point
            valid_lines, valid_id_roads = handle_crossings(valid_lines, valid_id_roads)
            
            # change data format to nodes(x, y), and edges(node_a, node_b)
            nodes, edges = to_nodes_edges(valid_lines, valid_id_roads)
            
            # merge duplicate nodes
            nodes, edges, n_del = merge_duplicate_nodes(nodes, edges)
            
            # merge consecutive (almost) straight lines
            nodes, edges, n_del = merge_straight_lines(nodes, edges)
            
            # normalize coordinates to [-1, +1]
            nodes = normalize_coordinates(nodes, this_point)
            
            # filter for graph size
            if MIN_NUM_EDGES <= len(edges) <= MAX_NUM_EDGES and MIN_NUM_NODES <= len(nodes) <= MAX_NUM_NODES:
                longest_road = max(Counter(valid_id_roads).values())
                # optionally, filter out very long roads
                if longest_road < 10:
                    # get graph representation with adjacency lists per every node_id
                    adj_lists = edges_to_adjacency_lists(edges)
                    
                    # compute BFS, DFS in different formats
                    bfs_nodes, bfs_edges = generate_bfs(copy.deepcopy(nodes), adj_lists)
                    dfs_nodes, dfs_edges = generate_dfs(copy.deepcopy(nodes), adj_lists)
                    
                    # plot DFS/BFS
                    # for k in range(1, len(dfs_edges) + 1):
                    #     path_image = PATH_IMAGES[this_split] + "{:0>7d}_{}.png".format(count, k)
                    #     save_image_bfs(square_origin, dfs_edges[:k], path_image, plot_nodes=True)
                    
                    # plot network for current datapoint and save it
                    path_image = PATH_IMAGES[this_split] + "{:0>7d}.png".format(count)
                    save_image(nodes, edges, path_image)
                    
                    """
                    Optionally, plot the graphs before pre-processing or by coloring differently each road, and
                    higlighting nodes:
                    """
                    # # a) save before preprocessing
                    # path_image2 = PATH_IMAGES[this_split] + "extra_plots/" + "{:0>7d}b.png".format(count)
                    # save_image_by_lines(this_square, valid_lines, path_image2, id_roads=valid_id_roads, plot_nodes=True)
                    # # b) save with colored edges and nodes
                    # path_image3 = PATH_IMAGES[this_split] + "extra_plots/" + "{:0>7d}c.png".format(count)
                    # save_image_by_nodes_edges(UNIT_SQUARE, nodes, edges, path_image3, plot_nodes=True)
                    
                    # generate the representation of this datapoint, and store it in its dataset split dictionary
                    current_dict = generate_datapoint(count, count, nodes, edges, adj_lists, bfs_edges, bfs_nodes,
                                                      dfs_edges, dfs_nodes, this_split, this_point)
                    datasets[this_split][count] = current_dict
                    
                    # If this datapoint belongs to the training set, possibly augment with flip and rotation.
                    # Then, generate the datapoint and save the image with semantic segmentation.
                    # The type of augmentation used is stored as an attribute in data points in the augment split.
                    # if this_split == 0:
                    #     nodes_list = augment(nodes)
                    #     for id_augment, nodes in enumerate(nodes_list):
                    #         # plot augmented datapoint
                    #         path_image = PATH_IMAGES[3] + "{:0>7d}.png".format(count_augment)
                    #         save_image(nodes, edges, path_image)
                    #
                    #         # generate the representation of this datapoint,
                    #         # and store it in its dataset split dictionary
                    #         current_dict = generate_datapoint(count_augment, count, nodes, edges, adj_lists, bfs_edges,
                    #                                           bfs_nodes, dfs_edges, dfs_nodes, 3, this_point,
                    #                                           id_augment=id_augment)
                    #         datasets[3][count_augment] = current_dict
                    #         count_augment += 1
                    
                    count += 1
                    count_split[this_split] += 1
    
    # finally save all the splits in the generated dataset and plot the size of each
    save_dataset(datasets, PATH_FILES)
    print(
        f"Final count: {count} | augmented: {count_augment} | count per split: {count_split} | squares explored: {tot_count}")


if __name__ == '__main__':
    main_generate_datapoints()

"""
Expected outputs

With distance threshold = 0.0001:
Final count: 125905 | augmented: 0 | count per split: [94396, 13920, 17589] | squares explored: 1913472

With distance threshold = 0.0002:
Final count: 110891 | count_min_filtering: 126035| augmented: 0 | split: [80254, 11670, 18967] | squares explored: 1913472
"""