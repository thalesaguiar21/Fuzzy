from dataclasses import dataclass
from operator import itemgetter
from collections import namedtuple

import numpy as np

class Node(namedtuple('Node', 'loc lbl left right')):

    def __str__(self):
        return f'({self.left}, {self.loc}, {self.right})'


def build(points, depth=0):
    if len(points) == 0:
        return None
    dim = len(points[0]) - 1 # The last axis is the label
    axis = depth % dim
    points.sort(key=itemgetter(axis))
    med = len(points) // 2
    return Node(loc=points[med][:-1], lbl=points[med][-1:],
                left=kdtree(points[:med], depth+1),
                right=kdtree(points[med + 1:], depth + 1))


def find_neighbours(root, point, n_neigh, p):
    nodes = [root]
    neighbours = []
    dists = []
    depth = 0
    dim = len(root.loc)
    while len(nodes) > 0:
        axis = depth % dim
        current = nodes.pop()
        currdist = make_dist(current.loc, point, p)
        if len(neighbours) < n_neigh:
            dists.append(currdist)
            neighbours.append(current.loc + current.lbl)
            nodes, depth = _update_nodes(nodes, current, depth)
        elif _has_intersection(current.loc, point, axis, max(dists)):
            for dist in dists:
                if dist > currdist:
                    dists.append(currdist)
                    idx = dists.index(max(dists))
                    dists.pop(idx)
                    neighbours[idx] = current.loc + current.lbl
            nodes, depth = _update_nodes(nodes, current, depth)
    return neighbours


def _update_nodes(nodes, curr, depth):
    if curr.right is not None:
        nodes.append(curr.right)
    if curr.left is not None:
        nodes.append(curr.left)
    return nodes, depth + 1


def _has_intersection(r, s, axis, radius):
    return abs(r[axis] - s[axis]) < radius


def make_dist(r, s, p):
    ''' Sum of power differences between two points

    Args:
        r: first point
        s: second point
        p: exponent
    '''
    dist = 0
    for i in range(len(r)):
        dist += abs(r[i] - s[i])**p
    return dist


