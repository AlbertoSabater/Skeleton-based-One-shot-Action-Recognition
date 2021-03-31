#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:51:15 2020

@author: asabater
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import distance



def get_distance(dist_func, emb1, emb2, verbose=False):
    if dist_func in ['euc', 'euclidean']:
        dist = distance.euclidean(emb1, emb2)
    elif dist_func in ['cos', 'cosine']:
        dist = distance.cosine(emb1, emb2)
    elif dist_func in ['js', 'jensenshannon']:
        dist = distance.jensenshannon(emb1, emb2)
    else:
        raise ValueError('Distance "{}" not handled'.format(dist_func))
    if verbose: print(dist_func, dist)
    return dist


def eval_triplet_embeddings(triplet_embeddings, dist_func, verbose=False):
    res = []
    for A,P,N in triplet_embeddings:
        res.append((get_distance(dist_func, A, P), get_distance(dist_func, A, N)))
        # if dist_func == 'euclidean':
        #     res.append((distance.euclidean(A, P), distance.euclidean(A, N)))
        # elif dist_func == 'cosine':
        #     res.append((distance.cosine(A, P), distance.cosine(A, N)))
        # elif dist_func == 'jensenshannon':
        #     res.append((distance.jensenshannon(A, P), distance.jensenshannon(A, N)))
        # else:
        #     raise ValueError('Distance "{}" not handled')
    res = np.array(res)
    
    num_positives = (res[:,0] < res[:,1]).sum()
    # print(' - num great_distances: {} / {} | {:.2f} %'.format(num_positives, len(res), num_positives*100/len(res)))
    # print(' - mean P {:.3f} | mean N {:.3f}'.format(np.mean(res[:,0]), np.mean(res[:,1])))
    # print(' - difference mean: {:.3f}'.format(np.mean(res[:,1] - res[:,0]))) 
    
    stats = {
            'num_positives': num_positives,'total_samples': len(res), 'acc': num_positives*100/len(res),
            'dist_diff': np.mean(res[:,1] - res[:,0]), 'dist_p': np.mean(res[:,0]), 'dist_n': np.mean(res[:,1])
            }
    
    if verbose:
        print('|| {:<14}||'.format(dist_func) +\
              'Accuracy: {}/{} | {:.2f}%  ||  '.format(num_positives, len(res), num_positives*100/len(res)) +\
              'Dist. Difference: {:.3f}  ||  '.format(np.mean(res[:,1] - res[:,0]))  +\
              'Distances Mean: P_{:.3f} | N_{:.3f}'.format(np.mean(res[:,0]), np.mean(res[:,1])))
    
    return stats




def plot_skel(ax, skel, lim_min=-0.5, lim_max=0.5, title='', show_joint_txt=True, show_axis=True):
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')      
    
    if lim_min is not None:
        ax.set_xlim3d([-lim_min, -lim_max])
        ax.set_zlim3d([lim_min, lim_max])
        ax.set_ylim3d([lim_min, lim_max])
    
    print_3d_skeleton(ax, skel)
    if show_joint_txt:
        for i in range(len(skel)): ax.text(skel[i][0], skel[i][1], skel[i][2], str(i))
    
    ax.scatter(skel[12,0], skel[12,1], skel[12,2], alpha=1, s=20, c='r')
    ax.scatter(skel[16,0], skel[16,1], skel[16,2], alpha=1, s=20, c='r')
    ax.scatter(skel[20,0], skel[20,1], skel[20,2], alpha=1, s=20, c='r')
    
    if show_axis:
        axis_len = 2
        ax.plot([-axis_len,axis_len], [0,0], [0,0], linestyle='--')
        ax.plot([0,0], [-axis_len,axis_len], [0,0], linestyle='--')
        ax.plot([0,0], [0,0], [-axis_len,axis_len], linestyle='--')
    
    ax.set_title(title) 
    
def print_3d_skeleton(ax, skel, color=None, color_scatter=None,
                      linestyle='-', scatter_size = 10, scatter_alpha=0.3,
                      connecting_joint = [1, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11]):
    
    if color_scatter is not None: color_scatter = [color_scatter] * len(skel)
    ax.scatter(skel[:,0], skel[:,1], skel[:,2], alpha=scatter_alpha, s=scatter_size, c=color_scatter)        # , c=color
    
    
    for i in range(len(connecting_joint)):
    # for i in range(25):
        p1, p2 = i, connecting_joint[i]
        ax.plot((skel[p1][0],skel[p2][0]), 
                (skel[p1][1],skel[p2][1]), 
                (skel[p1][2],skel[p2][2]), c=color, linewidth=6, linestyle=linestyle) 



# https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def get_fixed_lims(skels, max_p = 90):
        X,Y,Z = skels[:,:,0], skels[:,:,1], skels[:,:,2]
        # max_p, min_p = 80, 20
        # max_p, min_p = 85,15
        # max_p, min_p = 90,10
        min_p = 100 - max_p
        max_range = np.array([np.percentile(X, max_p)-np.percentile(X, min_p), 
                              np.percentile(Y, max_p)-np.percentile(Y, min_p), 
                              np.percentile(Z, max_p)-np.percentile(Z, min_p)]).max() / 2.0
        mid_x, mid_y, mid_z = (np.percentile(X, max_p)+np.percentile(X, min_p)) * 0.5, \
                                (np.percentile(Y, max_p)+np.percentile(Y, min_p)) * 0.5, \
                                (np.percentile(Z, max_p)+np.percentile(Z, min_p)) * 0.5
        x_lim = (mid_x - max_range, mid_x + max_range)
        y_lim = (mid_y - max_range, mid_y + max_range)
        z_lim = (mid_z - max_range, mid_z + max_range)
        return x_lim, y_lim, z_lim

