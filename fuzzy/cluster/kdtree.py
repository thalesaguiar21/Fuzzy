from dataclasses import dataclass
from operator import itemgetter
from collections import namedtuple

import numpy as np


class Node(namedtuple('Node', 'loc lbl left right')):

    def __str__(self):
        return f'({self.left}, {self.loc}, {self.right})'


def to_data(nodes):
    return np.array([node.loc + node.lbl for node in nodes])


def build(points):
    return _build(points, 0)


def _build(points, depth):
    if len(points) == 0:
        return None
    dim = len(points[0]) - 1 # The last axis is the label
    axis = depth % dim
    points.sort(key=itemgetter(axis))
    med = len(points) // 2
    return Node(loc=points[med][:-1], lbl=points[med][-1:],
                left=_build(points[:med], depth+1),
                right=_build(points[med + 1:], depth + 1))


def find_neighbours(root, point, n_neigh, p):
    """ Searches the given tree for the closest neighbours

    Args:
        root: the subtree (kdtree).
        point: the data to search for neighbours.
        n_neigh: the quantity of neighbours.
        p: the similarity, (1) for Manhattan, (2) for Euclidean, and (3) for
            Minkowski.

    Raises:
        ValueError, if the quantity of neighbours is negative.
        ValueError, if the dimension of 'point' is different from the dimension
            of data.
        ValueError, if the metric is not in [1, 2, 3]

    Returns:
        the n_neigh closest data to 'point'.
    """
    if n_neigh < 0:
        raise ValueError('The quantity of neighbours must be positive')
    elif len(point) != len(root.loc):
        raise ValueError(f'The point must have dimenson == {len(root.loc)}')
    elif n_neigh == 0:
        return np.array([])
    else:
        return _find(root, point, n_neigh, p)


def _find(root, point, n_neigh, p):
    nodes = [root]
    neighbours = []
    dists = []
    depth = 0
    dim = len(root.loc)
    while len(nodes) > 0:
        axis = depth % dim
        current = nodes.pop()
        currdist = calc_dist(current.loc, point, p)
        if len(neighbours) < n_neigh:
            dists.append(currdist)
            neighbours.append(current)
            nodes, depth = _update_nodes(nodes, current, depth)
        elif _has_intersection(current.loc, point, axis, max(dists)):
            for dist in dists:
                if dist > currdist:
                    dists.append(currdist)
                    idx = dists.index(max(dists))
                    dists.pop(idx)
                    neighbours[idx] = current
            nodes, depth = _update_nodes(nodes, current, depth)
    return to_data(neighbours)


def _update_nodes(nodes, curr, depth):
    if curr.right is not None:
        nodes.append(curr.right)
    if curr.left is not None:
        nodes.append(curr.left)
    return nodes, depth + 1


def _has_intersection(r, s, axis, radius):
    return abs(r[axis] - s[axis]) < radius


def calc_dist(r, s, p):
    ''' Sum of power differences between two points

    Args:
        r: first point
        s: second point
        p: exponent
    '''
    if p <= 0 or p > 3:
        raise ValueError('Invalid power, to be a metric it must be 1, 2 or 3')
    dist = 0
    for i in range(len(r)):
        dist += abs(r[i] - s[i])**p
    return dist

