#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:24:14 2020

@author: asabater
"""

import os
import numpy as np
import sys
import pickle
from tensorflow.keras.utils import to_categorical 


flip_correspondences_left =  [4,5,6,7,   12,13,14,15, 21,22]
flip_correspondences_right = [8,9,10,11, 16,17,18,19, 23,24]
spine = [0, 1, 2, 3, 20]

connecting_joint = [1, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11]


# %%

import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt 
from scipy.spatial.distance import cdist
from tqdm import tqdm
from scipy.special import comb
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Calculate JCD feature
def norm_scale(x):
    return (x-np.mean(x))/np.mean(x)

# Crop movement to max_seq_len frames
def zoom_to_max_len(p, max_seq_len, joints_num, joints_dim, force=False):  
    # Resize movement
    num_frames = p.shape[0]
    if force or num_frames > max_seq_len:
        # Zoom -> crop movement
        p_new = np.zeros([max_seq_len, joints_num, joints_dim], dtype="float32") 
        for m in range(joints_num):
            for n in range(joints_dim):
                # smooth coordinates
                # Zoom coordinates to fit the max_seq_len_shape
                p_new[:,m,n] = inter.zoom(p[:,m,n], max_seq_len/num_frames)[:max_seq_len]   # , mode='nearest'
    else:
        p_new = p
    return p_new

def get_jcd_features(p, joints_num, max_seq_len):
    # Get joint distances
    jcd = []
    iu = np.triu_indices(joints_num, 1, joints_num)
    for f in range(max_seq_len): 
        d_m = cdist(p[f],p[f],'euclidean')       
        d_m = d_m[iu] 
        jcd.append(d_m)
    jcd = np.stack(jcd) 
    
    return jcd

def get_bone_spherical_angles(v):
    elevation = np.arctan2(v[:,2], np.sqrt(v[:,0]**2 + v[:,1]**2))
    azimuth = np.arctan2(v[:,1], v[:,0])
    return np.column_stack([elevation, azimuth])
def get_body_spherical_angles(body):
    angles = np.column_stack([ get_bone_spherical_angles(body[:, i+1] - body[:, i]) for i in range(len(connecting_joint)-1) ])
    return angles


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def matrix_unit_vector(matrix):
    div = np.linalg.norm(matrix, axis=1)[:, None]
    return np.divide(matrix, div, out=np.zeros_like(matrix), where=div!=0)
def get_transformation_matrix_global(skel):
    o = (skel[:, 16, :] + skel[:, 12, :]) / 2
    x = matrix_unit_vector(skel[:, 12] - o)
    z = matrix_unit_vector(skel[:, 20] - o)
    y = np.cross(x,z)

    x[(x == 0).any(axis=1)] = [1, 0, 0]
    y[(y == 0).any(axis=1)] = [0, 1, 0]
    z[(z == 0).any(axis=1)] = [0, 0, 1]

    r = [ np.linalg.inv(np.column_stack([ [*x[i], 0], [*y[i], 0], [*z[i], 0], [*o[i], 1] ])) for i in range(len(skel)) ]
    return np.stack(r)

def transform_skel_global(skel, r):
    skel = np.concatenate([skel, np.ones((skel.shape[0], 25, 1))], axis=-1)
    skel = np.matmul(skel, r.transpose([0,2,1]))
    skel = skel[..., :3]
    return skel  

# Exanche coordinates between simetric joints and flip the X axis
# Flip X axis to the not simetric joints
# The body remains looking at the same side but with fliped movements respect to X axis
def flip_skeleton(skel, flip_axis=0):
    # skel[..., 0] = -skel[..., 0]
    aux = skel[..., flip_correspondences_left, :]
    skel[..., flip_correspondences_left, :] = skel[..., flip_correspondences_right, :]
    skel[..., flip_correspondences_right, :] = aux
    
    skel[..., flip_correspondences_left, flip_axis] = -skel[..., flip_correspondences_left, flip_axis]
    skel[..., flip_correspondences_right, flip_axis] = -skel[..., flip_correspondences_right, flip_axis]
    skel[..., spine, flip_axis] = -skel[..., spine, flip_axis]
    return skel

def scale_skel_by_torso(skel):
    torso_dists = np.linalg.norm(skel[:,20] - skel[:,1], axis=1) +\
                    np.linalg.norm(skel[:,1] - skel[:,0], axis=1)
    for i in range(skel.shape[0]): 
        rel = 0.4 / torso_dists[i] if torso_dists[i] != 0 else 1
        skel[i] = skel[i] * rel
    return skel




def average_wrong_frame_skels(skels):
    good_frames = np.all(~np.all(skels==0, axis=2), axis=1)
    for num_frame, gf in enumerate(good_frames):
        if gf: continue
        if num_frame == 0: skels[num_frame] = skels[num_frame+1]
        elif num_frame == len(skels)-1: skels[num_frame] = skels[num_frame-1]
        else: skels[num_frame] = (skels[num_frame+1] + skels[num_frame-1])/2
    return skels


# skip_frames -> list with the number of frames-1 to skip, to be choosen randomly
def get_pose_data_v2(body, max_seq_len, joints_num, joints_dim, center_skels, 
                     h_flip, scale_by_torso, temporal_scale, scaler, 
                     validation,
                     use_jcd_features, use_speeds,
                     use_coords_raw, use_coords, use_jcd_diff, 
                     use_bone_angles,
                     use_bone_angles_cent,
                     
                     skip_frames = [],
                     **kwargs):
    
    # Remove frames without predictions
    body = body[np.all(~np.all(body==0, axis=2), axis=1)]
    # body = body[body.sum(axis=1).sum(axis=1)!=0]
    
    
    # Crop or extend the movement by interpolation
    # If extension is longer than max_seq_len, crop to max_seq_len
    if not validation and temporal_scale is not False:
        orig_new_frames = len(body)
        temporal_scale = list(temporal_scale)
        temporal_scale[0] = int(temporal_scale[0]*orig_new_frames)
        temporal_scale[1] = int(temporal_scale[1]*orig_new_frames)
        new_num_frames = np.random.randint(*temporal_scale)
        new_num_frames = max(new_num_frames, 2)
        
        zoom_factor = new_num_frames/orig_new_frames
        body = inter.zoom(body, (zoom_factor,1,1), mode='nearest') 



    # Reduce frame rate
    if len(skip_frames) > 0:
        # print('aaaa', len(body))
        sk = np.random.choice(skip_frames)
        if validation: sk_init = 0
        else: sk_init = np.random.randint(sk)
        body = body[sk_init::sk]
        # print('bbbb', len(body))
    
    

        
    if max_seq_len > 0:
        # If movement is longer than max_seq_lenght -> crop to max_seq_length
        body = zoom_to_max_len(body, max_seq_len, joints_num, joints_dim)
        
    elif max_seq_len < 0:
        if not validation:
            # Crop randomly the movement to -max_seq_length
            start = np.random.randint(max(len(body)-abs(max_seq_len)+1, 1))
            end = start + abs(max_seq_len)
            body = body[start:end]
        else:
            # Crop to the last part of the movement
            start = max(0, (len(body) - abs(max_seq_len)) // 2)
            end = start + abs(max_seq_len)
            body = body[start:end]
        
    
    

        
    if scale_by_torso:
        body = scale_skel_by_torso(body)
    
    num_frames = len(body)
    # jcd_features, speed_features = [], []

    if not validation and h_flip and np.random.rand() > 0.5:
        body = flip_skeleton(body)
        
    body_before_center = body.copy()
    
    if center_skels:
        # Get transformation matrix
        
        r = get_transformation_matrix_global(body)
        
        skels = transform_skel_global(body, r)
        if use_speeds: skels_next = transform_skel_global(body[1:], r[:-1])
        
    else:
        skels = body
        if use_speeds: skels_next = body[1::]
        

    pose_features = []
    
    if use_bone_angles:     # 24*4
        # Elevation and azimuth for each bone (vector of consecutive joints)
        pose_features.append(get_body_spherical_angles(body))
    if use_bone_angles_cent:     # 24*4
        # Elevation and azimuth for each bone (vector of consecutive joints)
        pose_features.append(get_body_spherical_angles(skels))
    if use_coords_raw:  # 75 = 25*3
        # Raw coordinates
        pose_features.append(np.reshape(body_before_center, (num_frames,joints_num * joints_dim)))
    if use_coords:  # 75 = 25*3
        # Raw coordinates
        pose_features.append(np.reshape(skels, (num_frames,joints_num * joints_dim)))
        

    
    if use_jcd_diff or use_jcd_features:
        jcd_features = get_jcd_features(skels, joints_num, num_frames)
        
        if use_jcd_diff:  # 300 = comb(25,2)
            # Distance difference between frames per each pair of joints
            jcd_diff = jcd_features[1:] - jcd_features[:-1]
            jcd_diff = np.reshape(jcd_diff, (num_frames-1, jcd_features.shape[-1]))
            jcd_diff = np.concatenate([np.expand_dims(jcd_diff[0], axis=0), jcd_diff], axis=0)
            # print('Adding: use_jcd_diff')
            pose_features.append(jcd_diff)

        if use_jcd_features:  # 300 = comb(25,2)
            # Per-frame Joint distances
            pose_features.append(jcd_features)

        
    if use_speeds:  # 75 = 25*3
        # Frame-to-frame speeds
        speed_features = skels_next - skels[:-1]
        speed_features = np.reshape(speed_features, (num_frames-1, joints_num*joints_dim))
    
        speed_features = np.concatenate([np.expand_dims(speed_features[0], axis=0), speed_features], axis=0)
        pose_features.append(speed_features)

        
    # pose_features = np.concatenate([jcd_features, speed_features], axis=1)
    pose_features = np.concatenate(pose_features, axis=1).astype('float32')

    if scaler is not None:
        pose_features = scaler.transform(pose_features)
        
    return pose_features


def get_scaler_filename(joints_num, joints_dim, 
                        center_skels, scale_by_torso, 
                        
                        use_jcd_features, use_speeds,
                        use_coords_raw, use_coords, use_jcd_diff,
                        use_bone_angles,
                        use_bone_angles_cent,
                        num_feats,
                        **kwargs):
    return '/home/asabater/datasets/NTU-120/data_scalers/' +\
            'std_msl{}_jn{}_jd{}_cskl{}_strs{}'.format(
                -1, joints_num, joints_dim,
                'T' if center_skels else 'F',
                'T' if scale_by_torso else 'F') +\
                '_jcd{}_spds{}_coordsraw{}_coords{}_jcddiff{}_angs{}_angscent{}_numfeats{}.pckl'.format(
                
                'T' if use_jcd_features else 'F',
                'T' if use_speeds else 'F',
                'T' if use_coords_raw else 'F',
                'T' if use_coords else 'F',
                'T' if use_jcd_diff else 'F',
                'T' if use_bone_angles else 'F',
                'T' if use_bone_angles_cent else 'F',
                num_feats
                )

def load_scaler(joints_num, joints_dim, 
                        center_skels, scale_by_torso, 
                        
                        use_jcd_features, use_speeds,
                        use_coords_raw, use_coords, use_jcd_diff,
                        use_bone_angles,
                        use_bone_angles_cent,
                        num_feats,
                        **kwargs):
    filename = get_scaler_filename(joints_num, joints_dim, 
                        center_skels, scale_by_torso, 
                        
                        use_jcd_features, use_speeds,
                        use_coords_raw, use_coords, use_jcd_diff,
                        use_bone_angles,
                        use_bone_angles_cent,
                        num_feats)
    scaler = pickle.load(open(filename, 'rb'))
    return scaler

def get_num_feats(joints_num, joints_dim, 
                  use_jcd_features, use_speeds, use_coords_raw, use_coords, use_jcd_diff, 
                  use_bone_angles, use_bone_angles_cent, **kwargs):
    
    num_feats = 0
    if use_bone_angles:
        num_feats += (len(connecting_joint)-1)*2
    if use_bone_angles_cent:
        num_feats += (len(connecting_joint)-1)*2
    if use_jcd_features:
        num_feats += int(comb(joints_num,2))
    if use_speeds:
        num_feats += joints_num * joints_dim
    if use_coords_raw:
        num_feats += joints_num * joints_dim
    if use_coords:
        num_feats += joints_num * joints_dim
    if use_jcd_diff:
        num_feats += int(comb(joints_num,2))
    
    return num_feats
        
        


def get_body_skel(pose_raw, validation, mode='var'):
    n_bodys = list(set(pose_raw['nbodys']))
    if len(n_bodys) == 0:
        p = pose_raw['skel_body0']
    else:
        body_lens = np.array([ len(pose[np.all(~np.all(pose==0, axis=2), axis=1)]) for pose in \
                          [ pose_raw['skel_body{}'.format(i)] for i in range(max(n_bodys)) ] ])  
        body_lens = np.where(body_lens == max(body_lens))[0]
        if validation:
            if mode == 'normal':
                p = pose_raw['skel_body{}'.format(body_lens[0])]
            elif mode == 'var':
                stds = [ pose_raw['skel_body{}'.format(i)].std() for i in range(len(np.where(body_lens == max(body_lens))[0])) ]
                p = pose_raw['skel_body{}'.format(body_lens[stds.index(max(stds))])]
                # print(stds)
            else: raise ValueError('')
        else:
            p_ind = np.random.choice(body_lens)
            p = pose_raw['skel_body{}'.format(p_ind)]
    return p


# Triplet data generator
# Each batch is composed by K=4 samples of P=B/K different classes
# if max_seq_len == 0 -> samples inside a batch are zero-padded to fit their inner max length. 
#                           Longer sequences are zoomed out to fit max_seq_len
# if max_seq_len > 0 -> samples inside a batch are zoomed-out to fit max_seq_len
# if max_seq_len < 0 -> samples bigger than max_seq_len are randomly cropped to fit -max_seq_len
def triplet_data_generator(pose_annotations_file, 
                           batch_size, 
                           max_seq_len, joints_num, joints_dim, num_jcd_feats, 
                           scale_data, in_memory_generator, 
                           decoder, reverse_decoder,
                           center_skels, h_flip, scale_by_torso, 
                           temporal_scale, validation, 
                           triplet, 
                           classification, num_classes,
                           
                           use_jcd_features, use_speeds,
                           use_coords_raw, use_coords, use_jcd_diff,
                           use_bone_angles,
                           use_bone_angles_cent,
                           num_feats,
                           
                           skip_frames = [],
                           average_wrong_skels = True,
                           is_tcn=False,
                           K=4,
                           **kwargs):
    
    # Reads the annotations and stores them into a dict. Annotations are shuffled
    def read_annotations():
        pose_files = {}
        with open(pose_annotations_file, 'r') as f: 
            for line in f:
                filename, label = line.split()
                label = int(label)
                if label in pose_files: pose_files[label].append(filename)
                else: pose_files[label] = [filename]
        for k in pose_files.keys(): np.random.shuffle(pose_files[k])
        return pose_files

    # Return a random sample with the given label or a random one if there is no 
    # more samples with that label
    def get_random_sample(label):
        if label in pose_files and len(pose_files[label]) > 0:
            return pose_files[label].pop(), label
        else:
            if label in pose_files: del pose_files[label]
            new_label = np.random.choice(list(pose_files.keys()))
            return get_random_sample(new_label)

    if in_memory_generator: 
        print(' ** Data Generator | data will be cached | Validation: {} **'.format(validation))
        cached_data = {}
    if scale_data: 
        print(' ** Loading data scaler | Validation: {} **'.format(validation))
        scaler = load_scaler(joints_num, joints_dim, 
                        center_skels, scale_by_torso, 
                        
                        use_jcd_features, use_speeds,
                        use_coords_raw, use_coords, use_jcd_diff,
                        use_bone_angles,
                        use_bone_angles_cent,
                        num_feats)
    else:
        scaler = None
    
    print(' *** is_tcn', is_tcn)
        
        
    
    if not triplet: K = 1
    
    assert batch_size % K == 0
    P = batch_size // K
    pose_files = read_annotations()
    print('*************', K, P, batch_size)
    
    if classification:
        total_labels = sorted(list(pose_files.keys()))
        labels_dict = { l:i for i,l in enumerate(total_labels) }

    
    while True:
        if sum([ len(v) for v in pose_files.values() ]) < batch_size:
            # print('Update annotations')
            pose_files = read_annotations()
            
        batch_labels = []
        batch_samples = []
        if classification: y_clf = []
        for _ in range(P):
            label_iter = np.random.choice(list(pose_files.keys()))
            for i in range(K):
                filename, label = get_random_sample(label_iter)
                if classification:
                    label_cat = to_categorical(labels_dict[int(label)], num_classes=num_classes)

                if in_memory_generator and filename in cached_data.keys():
                    # print('Recovering data', filename)
                    sample = cached_data[filename]
                else:
                    # print('******', filename, '********')
                    pose_raw = np.load(filename, allow_pickle=True).item()
                    
                    p = get_body_skel(pose_raw, validation)

                    if average_wrong_skels: average_wrong_frame_skels(p)
                    sample = get_pose_data_v2(p, max_seq_len, joints_num, joints_dim, 
                                              center_skels, h_flip, scale_by_torso, 
                                              temporal_scale, scaler, validation,
                                              use_jcd_features, use_speeds,
                                              use_coords_raw, use_coords, use_jcd_diff,
                                              use_bone_angles, use_bone_angles_cent,
                                              skip_frames = skip_frames,
                                              )
                    
                    # print(validation, in_memory_generator)
                    if in_memory_generator: 
                        # print('Storing:', filename)
                        cached_data[filename] = sample
                batch_samples.append(sample); batch_labels.append(label)
                if classification: y_clf.append(label_cat)
                        
        
        if triplet: batch_labels = np.stack(batch_labels)       # for triplets
        if classification: y_clf = np.stack(y_clf).astype('int')              # for classification
        
        X, Y, sample_weights = [], [], {}
        
        X = pad_sequences(batch_samples, padding='pre', dtype='float32')
        
        if triplet: 
            Y.append(batch_labels)
        if classification: 
            # Y.append(y_clf)
            Y = y_clf
        if decoder:
            decoder_data = [ bs[::-1] for bs in batch_samples ] if reverse_decoder else batch_samples
            padding = 'pre' if is_tcn else 'post'
            # decoder_data = pad_sequences(decoder_data, padding='post', dtype='float32')
            decoder_data = pad_sequences(decoder_data, padding=padding, dtype='float32')
            Y.append(decoder_data)
            sample_weights['output_{}'.format(len(Y))] =  (decoder_data[:, :, 0] != 0).astype('float32')
        
        # print(Y)
        # print(X.shape, len(Y))
        yield X, Y, sample_weights
        

