from dataclasses import dataclass
from operator import itemgetter
from collections import namedtuple

import numpy as np

class Node(namedtuple('Node', 'loc left right')):

    def __str__(self):
        return f'({self.left}, {self.loc}, {self.right})'


def kdtree(points, depth=0):
    if len(points) == 0:
        return None
    dim = len(points[0])
    axis = depth % dim
    order = np.argsort(points[:, axis])
    points = points[order]
    med = len(points) // 2
    return Node(loc=points[med], left=kdtree(points[:med], depth+1),
                right=kdtree(points[med + 1:], depth + 1))


def find(root, point, depth=0):
    if root.loc is None:
        return -1
    elif (root.loc == point).all():
        return depth
    elif point[depth] < root.loc[depth]:
        return find(root.left, point, depth + 1)
    elif point[depth] > root.loc[depth]:
        return find(root.right, point, depth + 1)


def find_neighbours(root, point, k, dists=[], neigh=[], depth=0):
    if root is None:
        return None
    sqrdist = ((root.loc - point) ** 2).sum()
    if len(dists) < k:
        dists.append(sqrdist)
        neigh.append(root.loc)
    else:
        plane_dist = root.loc[depth] - point[depth]
        if plane_dist > max(dists):
            return None
        for dist in dists:
            if dist > sqrdist:
                dists.append(sqrdist)
                max_ = max(dists)
                idx = dists.index(max_)
                dists.pop(idx)
                neigh[idx] = root.loc
    _find_neighbours(root.left, point, k, dists, neigh, depth+1)
    depth -= 1
    _find_neighbours(root.right, point, k, dists, neigh, depth+1)
    return neigh


point_list = np.array([[7,2], [5,4], [9,6], [4,7], [8,1], [2,3]])
tree = kdtree(point_list)
neighbours = _find_neighbours(tree, [6, 3], 3)
print(neighbours)

