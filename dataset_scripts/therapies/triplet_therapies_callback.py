#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 19:31:42 2020

@author: asabater
"""

import os
import json
import tensorflow as tf
import numpy as np
import time
from joblib import Parallel, delayed

import pickle
from tqdm import tqdm
from scipy.spatial import distance

import sys
sys.path.append('../..')

import prediction_utils
import eval_utils
from data_generator import load_scaler, get_pose_data_v2, average_wrong_frame_skels




# =============================================================================
# Predictions in the full video
# =============================================================================
def get_therapies_triplet_distances(model, model_params, triplets, triplets_bgnd, video_skels,
                                    in_memory_callback, cached_videos):

    video_preds = {}
    for vid,(tempos,skels) in video_skels.items():
    # for vid,(tempos,skels) in tqdm(video_skels.items()):
        
        if not in_memory_callback or vid not in cached_videos:
            if model_params['average_wrong_skels']: skels = average_wrong_frame_skels(skels)
            skels = get_pose_data_v2(skels, validation=True, **model_params)
            skels = np.expand_dims(skels, axis=0)
            if in_memory_callback: cached_videos[vid] = skels
        else: skels = cached_videos[vid]

        preds = np.array(model.get_embedding(skels))[0]
        video_preds[vid] = preds
        # break

    triplet_embs = []
    for A,P,N in triplets: triplet_embs.append((video_preds[A[0]][A[2]], video_preds[P[0]][P[2]], video_preds[N[0]][N[2]]))
    
    triplet_bgnd_embs = []
    for A,P,N in triplets_bgnd: triplet_bgnd_embs.append((video_preds[A[0]][A[2]], video_preds[P[0]][P[2]], video_preds[N[0]][N[2]]))
    
    return triplet_embs, triplet_bgnd_embs

    
# =============================================================================
# Predictions on each action
# =============================================================================
def get_therapies_triplet_distances_by_samples(model, model_params, triplets, triplets_bgnd, video_skels, 
                                               in_memory_callback, cache):

    triplets_flat = [ '|'.join(map(str, tup)) for t in triplets + triplets_bgnd for tup in t ]
    triplets_flat = list(set(triplets_flat))
    triplets_flat = [ s.split('|') for s in triplets_flat ]
    
    def load_skels(skels, model_params):
        if model_params['average_wrong_skels']: skels = average_wrong_frame_skels(skels)
        skels = get_pose_data_v2(skels, validation=True, **model_params)
        return skels
    
    
    if not in_memory_callback or len(cache) == 0:
        total_skels = Parallel(n_jobs=7)(delayed(load_skels)\
                                         (video_skels[vid][1][int(init):int(end)], model_params) \
                                                  # for vid, init, end in tqdm(triplets_flat))
                                                  for vid, init, end in triplets_flat)
        
        skels_by_len = {}
        sample_by_length = {}
        for skels, sample in zip(total_skels, triplets_flat):
            pose_len = skels.shape[0]
            if pose_len in skels_by_len: 
                skels_by_len[pose_len].append(skels)
                sample_by_length[pose_len].append('|'.join(sample))
            else: 
                skels_by_len[pose_len] = [skels]
                sample_by_length[pose_len] = ['|'.join(sample)]
        del total_skels
        if in_memory_callback: 
            cache.append(skels_by_len)
            cache.append(sample_by_length)
    else: 
        # print(' * Loading cached targets')
        skels_by_len, sample_by_length = cache
        
        
    # Predict embeddings on batch
    batch_size = 128
    samples_embs = {}
    # for k in tqdm(skels_by_len.keys()):
    for k in skels_by_len.keys():
        for i in np.arange(0, len(skels_by_len[k]), batch_size):
            samples = np.stack(skels_by_len[k][i:i+batch_size])
            samples_name = sample_by_length[k][i:i+batch_size]
    
            preds_length = np.array(model.get_embedding(samples))
            
            for pred, sample_name in zip(preds_length, samples_name):
                samples_embs[sample_name] = pred
                
    
    triplet_embs = []
    for A,P,N in triplets: triplet_embs.append((samples_embs['|'.join(map(str, A))], samples_embs['|'.join(map(str, P))], samples_embs['|'.join(map(str, N))]))            

    triplet_bgnd_embs = []
    for A,P,N in triplets_bgnd: triplet_bgnd_embs.append((samples_embs['|'.join(map(str, A))], samples_embs['|'.join(map(str, P))], samples_embs['|'.join(map(str, N))]))            

    return triplet_embs, triplet_bgnd_embs


def eval_therapies_triplet_callback(model, model_params, writer, mode):
    # triplets = pickle.load(open(raw_data_path + 'triplets_dataset.pckl', 'rb'))
    # triplets_bgnd = pickle.load(open(raw_data_path + 'triplets_ther_pat_bgnd_dataset.pckl', 'rb'))
    # video_skels = pickle.load(open(os.path.join(raw_data_path, 'video_skels.pckl'), 'rb'))
    
    triplets = pickle.load(open(model_params['eval_therapies_triplets_dataset'], 'rb'))
    triplets_bgnd = pickle.load(open(model_params['eval_therapies_triplets_bgnd_dataset'], 'rb'))
    video_skels = pickle.load(open(model_params['eval_therapies_video_skels'], 'rb'))
    
    in_memory_callback = model_params['in_memory_callback']
    if in_memory_callback: print(' ** OneShot-Therapies Callback | {} | data will be cached **'.format(mode))
    
    if mode == 'full': cache = {}
    elif mode == 'sample': cache = []
    else: raise ValueError('Callback mode not recognized')
        
    scale_data = model_params['scale_data']
    if scale_data: model_params['scaler'] = load_scaler(**model_params)
    else: model_params['scaler'] = None

    model_params['max_seq_len'] = 0   
    model_params['skip_frames'] = []
    
    def eval_therapies(epoch, logs):
        if mode == 'full':
            model.set_encoder_return_sequences(True)
            triplet_embs, triplet_bgnd_embs = get_therapies_triplet_distances(
                                        model, model_params, 
                                        triplets, triplets_bgnd, video_skels,
                                        in_memory_callback, cache)
            model.set_encoder_return_sequences(False)
        elif mode == 'sample':
            triplet_embs, triplet_bgnd_embs = get_therapies_triplet_distances_by_samples(
                                        model, model_params, 
                                        triplets, triplets_bgnd, video_skels, 
                                        in_memory_callback, cache)
        else: raise ValueError('Callback mode not recognized')
       
        total_stats = []
        sys.stdout.flush()
        # print(' * {} triplets'.format(mode))
        stats = eval_utils.eval_triplet_embeddings(triplet_embs, 'euclidean')
        total_stats.append({ 'ther_trip_{}_euc_{}'.format(mode,k):v for k,v in stats.items() })
        stats = eval_utils.eval_triplet_embeddings(triplet_embs, 'cosine')
        total_stats.append({ 'ther_trip_{}_cos_{}'.format(mode,k):v for k,v in stats.items() })
        stats = eval_utils.eval_triplet_embeddings(triplet_embs, 'jensenshannon')
        total_stats.append({ 'ther_trip_{}_js_{}'.format(mode,k):v for k,v in stats.items() })
        txt = ' ** Triplet therapies {} ** '.format(mode)
        txt += ' ||| Euc.: {:.2f}% | diff. {:.2f} ||| Cos.: {:.2f}% | diff. {:.2f} ||| JS.: {:.2f}% | diff. {:.2f} ||| '.format(\
                total_stats[0]['ther_trip_{}_euc_{}'.format(mode,'acc')], total_stats[0]['ther_trip_{}_euc_{}'.format(mode,'dist_diff')],
                total_stats[1]['ther_trip_{}_cos_{}'.format(mode,'acc')], total_stats[1]['ther_trip_{}_cos_{}'.format(mode,'dist_diff')],
                total_stats[2]['ther_trip_{}_js_{}'.format(mode,'acc')], total_stats[2]['ther_trip_{}_js_{}'.format(mode,'dist_diff')] )
        
        # print(' * {} triplets_bgnd'.format(mode))
        stats = eval_utils.eval_triplet_embeddings(triplet_bgnd_embs, 'euclidean')
        total_stats.append({ 'ther_trip_bgnd_{}_euc_{}'.format(mode,k):v for k,v in stats.items() })
        stats = eval_utils.eval_triplet_embeddings(triplet_bgnd_embs, 'cosine')
        total_stats.append({ 'ther_trip_bgnd_{}_cos_{}'.format(mode,k):v for k,v in stats.items() })
        stats = eval_utils.eval_triplet_embeddings(triplet_bgnd_embs, 'jensenshannon')
        total_stats.append({ 'ther_trip_bgnd_{}_js_{}'.format(mode,k):v for k,v in stats.items() })
        txt += ' ** BGND ** Euc.: {:.2f}% | diff. {:.2f} ||| Cos.: {:.2f}% | diff. {:.2f} ||| JS.: {:.2f}% | diff. {:.2f}'.format(\
                total_stats[3]['ther_trip_bgnd_{}_euc_{}'.format(mode,'acc')], total_stats[3]['ther_trip_bgnd_{}_euc_{}'.format(mode,'dist_diff')],
                total_stats[4]['ther_trip_bgnd_{}_cos_{}'.format(mode,'acc')], total_stats[4]['ther_trip_bgnd_{}_cos_{}'.format(mode,'dist_diff')],
                total_stats[5]['ther_trip_bgnd_{}_js_{}'.format(mode,'acc')], total_stats[5]['ther_trip_bgnd_{}_js_{}'.format(mode,'dist_diff')] )
            
        print(txt)

        sys.stdout.flush()
    
        with writer.as_default():
            for stats in total_stats:
                logs.update(**stats)
                for k,v in stats.items(): tf.summary.scalar(k, data=v, step=epoch)
            writer.flush()    
            
    return eval_therapies






