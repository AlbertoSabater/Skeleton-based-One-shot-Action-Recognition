#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 19:45:28 2020

@author: asabater
"""

import os
import sys
sys.path.append('../..')
sys.path.append('../../eval_scripts/')
sys.path.append(os.getcwd() + '/eval_scripts/')

import prediction_utils
import eval_utils

from train_callbacks import load_pose
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pickle
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from data_generator import load_scaler
import tensorflow as tf



def get_ntu_one_shot_triplets(model, model_params, one_shot_eval_anchors, one_shot_eval_set, triplets,
                              in_memory_callback, cached_anchors, cached_targets):

    # Get anchor predictions
    anchor_preds = {}
    # for f in tqdm(one_shot_eval_anchors):
    for f in one_shot_eval_anchors:
        filename, label = f.split()

        # if in_memory_callback and filename in cached_anchors:
        if in_memory_callback and filename in cached_anchors:
            pose_data = cached_anchors[filename]
        else:
            pose_len, label, pose_data = load_pose(f, model_params)
            pose_data = np.expand_dims(pose_data, axis=0)
            if in_memory_callback: cached_anchors[filename] = pose_data
        
        anchor_preds[label] = np.array(model.get_embedding(pose_data)[0])
    classes = list(anchor_preds.keys())

    
    if not in_memory_callback or len(cached_targets) == 0:
        # Load skels
        total_pose_data = Parallel(n_jobs=1)(delayed(load_pose)(f, model_params) \
                                              # for f in tqdm(one_shot_eval_set))
                                              for f in one_shot_eval_set)
        # Group skels by length
        pose_data_by_len = {}
        filenames_by_length = {}
        y_true_by_len = {}
        for (pose_len, label, pose_data), ann in zip(total_pose_data, one_shot_eval_set):
            if pose_len in pose_data_by_len: 
                pose_data_by_len[pose_len].append(pose_data)
                filenames_by_length[pose_len].append(ann.split()[0])
                y_true_by_len[pose_len].append(label)
            else: 
                pose_data_by_len[pose_len] = [pose_data]
                filenames_by_length[pose_len] = [ann.split()[0]]
                y_true_by_len[pose_len] = [label]   
        del total_pose_data
        
        if in_memory_callback: 
            cached_targets.append(pose_data_by_len)
            cached_targets.append(filenames_by_length)
            cached_targets.append(y_true_by_len)
    else: 
        # print(' * Loading cached targets')
        pose_data_by_len, filenames_by_length, y_true_by_len = cached_targets
    
    # Predict embeddings on batch
    batch_size = 256
    file_embs = {}
    file_labels = {}
    # for k in tqdm(pose_data_by_len.keys()):
    for k in pose_data_by_len.keys():
        for i in np.arange(0, len(pose_data_by_len[k]), batch_size):
            samples = np.stack(pose_data_by_len[k][i:i+batch_size])
            filenames = filenames_by_length[k][i:i+batch_size]
            y_true = y_true_by_len[k][i:i+batch_size]
    
            preds_length = np.array(model.get_embedding(samples))
            
            for pred, filename, label in zip(preds_length, filenames, y_true):
                file_embs[filename] = pred
                file_labels[filename] = label
    
    
    # def get_one_shot_label(file, emb, anchor_preds, file_labels, classes):
    #     dists_euc = [ distance.euclidean(anchor, emb) for label, anchor in anchor_preds.items() ]
    #     dists_cos = [ distance.cosine(anchor, emb) for label, anchor in anchor_preds.items() ]
        
    #     y_true = file_labels[file]
    #     y_pred_euc = classes[dists_euc.index(min(dists_euc))]
    #     y_pred_cos = classes[dists_cos.index(min(dists_cos))]

    #     return y_true, y_pred_euc, y_pred_cos

    # # Evaluate one-shot
    # y = Parallel(n_jobs=7)(delayed(get_one_shot_label)(file, emb, anchor_preds, file_labels, classes) \
    #                                           for file, emb in file_embs.items())
    # y = np.array(y)
    # one_shot_acc_euc = accuracy_score(y[:,0], y[:,1])
    # one_shot_acc_cos  = accuracy_score(y[:,0], y[:,2])
    

    # Evaluate one-shot
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn.fit(list(anchor_preds.values()), list(anchor_preds.keys()))
    one_shot_acc_euc = knn.score(list(file_embs.values()), list(file_labels.values()))

    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    knn.fit(list(anchor_preds.values()), list(anchor_preds.keys()))
    one_shot_acc_cos = knn.score(list(file_embs.values()), list(file_labels.values()))

    knn = KNeighborsClassifier(n_neighbors=1, metric=distance.jensenshannon)
    knn.fit(list(anchor_preds.values()), list(anchor_preds.keys()))
    one_shot_acc_js = knn.score(list(file_embs.values()), list(file_labels.values()))
    
    print(' *** ', one_shot_acc_euc, one_shot_acc_cos, one_shot_acc_js)
          
    # # Evaluate one-shot
    # total_y_true, total_y_pred = [], []
    # for file, emb in file_embs.items():
    #     dists = []
    #     for label, anchor in anchor_preds.items(): dists.append(distance.euclidean(anchor, emb))
    #     # for label, anchor in anchor_preds.items(): dists.append(distance.cosine(anchor, emb))
    #     total_y_pred.append(classes[dists.index(min(dists))]) 
    #     total_y_true.append(file_labels[file])
    # acc = accuracy_score(total_y_true, total_y_pred)
    
    
    # Create triplet embeddings
    triplet_embs = []
    for A,P,N in triplets: triplet_embs.append((file_embs[A], file_embs[P], file_embs[N]))
    
    return one_shot_acc_euc, one_shot_acc_cos, triplet_embs


def eval_ntu_one_shot_triplets_callback(model, model_params, writer, return_one_shot_as_metric=False):
    
    with open(model_params['eval_ntu_one_shot_eval_anchors_file'], 'r') as f: one_shot_eval_anchors = f.read().splitlines()
    with open(model_params['eval_ntu_one_shot_eval_set_file'], 'r') as f: one_shot_eval_set = f.read().splitlines()
    triplets_file = '/'.join(model_params['eval_ntu_one_shot_eval_set_file'].split('/')[:-1] + ['triplets/'+model_params['eval_ntu_one_shot_eval_set_file'].split('/')[-1]])
    triplets = pickle.load(open(triplets_file, 'rb'))

    
    in_memory_callback = model_params['in_memory_callback']
    if in_memory_callback: print(' ** NTU Callback | data will be cached **')
    cached_anchors, cached_targets = {}, []

    scale_data = model_params['scale_data']
    if scale_data: model_params['scaler'] = load_scaler(**model_params)
    else: model_params['scaler'] = None
	

    def eval_ntu(epoch, logs):
        one_shot_acc_euc, one_shot_acc_cos, triplet_embs = get_ntu_one_shot_triplets(model, model_params,
                              one_shot_eval_anchors, one_shot_eval_set, triplets,
                              in_memory_callback, cached_anchors, cached_targets)
        
        print(' ** One-shot NTU acc | euclidean {:.3f} | cosine {:.3f}'.format(one_shot_acc_euc, one_shot_acc_cos))
        
        total_stats = []
        sys.stdout.flush()
        # print(' * {} triplets'.format('NTU'))
        stats = eval_utils.eval_triplet_embeddings(triplet_embs, 'euclidean')
        total_stats.append({ 'ntu_trip_euc_{}'.format(k):v for k,v in stats.items() })
        stats = eval_utils.eval_triplet_embeddings(triplet_embs, 'cosine')
        total_stats.append({ 'ntu_trip_cos_{}'.format(k):v for k,v in stats.items() })
        stats = eval_utils.eval_triplet_embeddings(triplet_embs, 'jensenshannon')
        total_stats.append({ 'ntu_trip_js_{}'.format(k):v for k,v in stats.items() })

        print(' ** Triplets NTU-120        ** Euc.: {:.2f}% | diff. {:.2f} ||| Cos.: {:.2f}% | diff. {:.2f} ||| JS.: {:.2f}% | diff. {:.2f}'.format(\
                total_stats[0]['ntu_trip_euc_{}'.format('acc')], total_stats[0]['ntu_trip_euc_{}'.format('dist_diff')],
                total_stats[1]['ntu_trip_cos_{}'.format('acc')], total_stats[1]['ntu_trip_cos_{}'.format('dist_diff')] ,
                total_stats[2]['ntu_trip_js_{}'.format('acc')], total_stats[2]['ntu_trip_js_{}'.format('dist_diff')] ))
         
        # print(total_stats)
        sys.stdout.flush()
        
    
        if writer is not None:
	        print(' * Writing callback logs')
	        with writer.as_default():
				
				
	            logs.update({'ntu_one_shot_acc_euc': one_shot_acc_euc,
	                         'ntu_one_shot_acc_cos': one_shot_acc_cos,
	                         # 'ntu_one_shot_acc_js': one_shot_acc_js
	                         })
	            tf.summary.scalar('ntu_one_shot_acc_euc', data=one_shot_acc_euc, step=epoch)
	            tf.summary.scalar('ntu_one_shot_acc_cos', data=one_shot_acc_cos, step=epoch)
	            # tf.summary.scalar('ntu_one_shot_acc_js', data=one_shot_acc_js, step=epoch)
	            for stats in total_stats:
	                logs.update(**stats)
	                for k,v in stats.items(): tf.summary.scalar(k, data=v, step=epoch)
	            writer.flush()    

    return eval_ntu



