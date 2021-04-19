#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 19:11:16 2020

@author: asabater
"""

import pickle
import os
import json
import matplotlib.pyplot as plt

import prediction_utils


from train_callbacks import load_pose
from tqdm import tqdm
import numpy as np
import time

from data_generator import get_body_skel, average_wrong_frame_skels, get_pose_data_v2, get_num_feats

np.random.seed(0)


# =============================================================================
# Batch evaluation
# =============================================================================


def ntu_batch_iterator(ntu_annotations, model_params):
    for ann in tqdm(ntu_annotations):
        # Load skel
        filename, label = ann.split()
        pose_raw = np.load(filename, allow_pickle=True).item()  #['skel_body0']        
        skels = get_body_skel(pose_raw, validation=True)
        if model_params['average_wrong_skels']: skels = average_wrong_frame_skels(skels)
        
        # Process skel data
        t = time.time()
        motion_data = get_pose_data_v2(skels, validation=True, **model_params)
        motion_len = len(motion_data)
        motion_data = np.expand_dims(motion_data, axis=0)
        t = time.time() - t
        yield t, motion_len, motion_data
        
        
def ntu_per_frame_iterator(ntu_annotations, model_params, max_seq_len):
    # Speed features are not handled by the per-frame iterator
    assert not model_params['use_speeds']
    num_feats = get_num_feats(**model_params)
    for ann in tqdm(ntu_annotations):
        # Load skel
        filename, label = ann.split()
        pose_raw = np.load(filename, allow_pickle=True).item()  #['skel_body0']        
        skels = get_body_skel(pose_raw, validation=True)
        if model_params['average_wrong_skels']: skels = average_wrong_frame_skels(skels)
        clip_data = np.zeros((1,0,num_feats))
        for num_frame in range(skels.shape[0]): 
            # Process skel data
            skel = np.expand_dims(skels[num_frame], axis=0)
            skel = skel[np.all(~np.all(skel==0, axis=2), axis=1)]
            if len(skel) == 0: continue
            t = time.time()
            skel_data = get_pose_data_v2(skel, validation=True, **model_params)
            skel_data = np.expand_dims(skel_data, axis=0)
            clip_data = np.concatenate([clip_data, skel_data], axis=1)
            motion_data = clip_data[:, min(0, -abs(max_seq_len)):, :]
            # motion_len = motion_data.shape[1]
            t = time.time() - t
            yield t, 1, motion_data
    

def ther_batch_iterator(video_skels, model_params):
    print(' *** ', ther_batch_iterator)
    for vid, skels in tqdm(video_skels.items()):
        # Load skel
        if model_params['average_wrong_skels']: skels = average_wrong_frame_skels(skels)

        # Process skel data
        t = time.time()
        motion_data = get_pose_data_v2(skels, validation=True, **model_params)
        motion_len = len(motion_data)
        motion_data = np.expand_dims(motion_data, axis=0)
        t = time.time() - t
        yield t, motion_len, motion_data
      
def ther_per_frame_iterator(video_skels, model_params, max_seq_len):
    # Speed features are not handled by the per-frame iterator
    assert not model_params['use_speeds']
    num_feats = get_num_feats(**model_params)
    for vid, skels in tqdm(video_skels.items()):
        # Load skel
        if model_params['average_wrong_skels']: skels = average_wrong_frame_skels(skels)
        clip_data = np.zeros((1,0,num_feats))
        for num_frame in range(skels.shape[0]): 
            # Process skel data
            skel = np.expand_dims(skels[num_frame], axis=0)
            skel = skel[np.all(~np.all(skel==0, axis=2), axis=1)]
            if len(skel) == 0: continue
            t = time.time()
            skel_data = get_pose_data_v2(skel, validation=True, **model_params)
            skel_data = np.expand_dims(skel_data, axis=0)
            clip_data = np.concatenate([clip_data, skel_data], axis=1)
            motion_data = clip_data[:, min(0, -abs(max_seq_len)):, :]
            # motion_len = motion_data.shape[1]
            t = time.time() - t
            yield t, 1, motion_data          

def main(model, online_evaluation, iterator, max_seq_len, plot_results):


    data_fps, pred_fps, total_fps = [], [], []
    motion_lens = []
    for data_time, motion_len, pose_data in iterator:
        t = time.time()
        _ = np.array(model.get_embedding(pose_data)[0])
        pred_time = time.time() - t
        total_time = data_time + pred_time
    
        data_fps.append(motion_len/data_time)
        pred_fps.append(motion_len/pred_time)
        total_fps.append(motion_len/total_time)
        motion_lens.append(motion_len)
        
    print(' * Stats: mean_data_fps   {:.2f} | : mean_pred_fps   {:.2f} | : mean_total_fps   {:.2f}'.format(
        np.mean(data_fps), np.mean(pred_fps), np.mean(total_fps)))
    print(' * Stats: median_data_fps {:.2f} | : median_pred_fps {:.2f} | : median_total_fps {:.2f}'.format(
        np.median(data_fps), np.median(pred_fps), np.median(total_fps)))


    if plot_results:
        plt.boxplot([data_fps, pred_fps, total_fps], labels=['pre-processing', 'predictions', 'total']);


if __name__ == '__main__':
    
    # For NTU
    # python demo_speed_ntu.py --use_ntu --use_gpu --test_online --test_offline --max_clips 1000 --path_model './pretrained_models/ntu_benchmark_model/' --path_ntu_anns './ntu_annotations/one_shot_aux_set_full.txt' 
    # python demo_speed_ntu.py --use_ntu --test_online --test_offline --max_clips 1000 --path_model './pretrained_models/ntu_benchmark_model/' --path_ntu_anns './ntu_annotations/one_shot_aux_set_full.txt' 

    # For THERAPIES
    # python demo_speed_ntu.py --use_therapies --use_gpu --test_online --test_offline --path_model './pretrained_models/therapies_model_7/'
    # python demo_speed_ntu.py --use_therapies --test_online --test_offline --path_model './pretrained_models/therapies_model_7/
    
    import argparse
    parser = argparse.ArgumentParser(description = 'Performs a speed test over NTU-120 skeleton clips')
    parser.add_argument('--use_gpu', action='store_true', help='enable GPU usage')
    parser.add_argument('--test_online', action='store_true', help='test online prediction speed')
    parser.add_argument('--test_offline', action='store_true', help='test offline speed')
    parser.add_argument('--path_model', type=str, required=True, help='path to the prediction model')
    parser.add_argument('--path_ntu_anns', type=str, help='path to the NTU annotations')
    parser.add_argument('--max_clips', type=int, default=1000, help='maximum skeleton clips to process')
    parser.add_argument('--plot_results', action='store_true', help='show a graph with the stats')
    parser.add_argument('--use_ntu', action='store_true', help='use NTU data for testing')
    parser.add_argument('--use_therapies', action='store_true', help='use NTU data for testing')
    args = parser.parse_args()
    
    print(args.use_ntu, args.use_therapies)
    
    assert any([ args.use_ntu, args.use_therapies ]), 'Select any dataset for testing'
    assert not all([ args.use_ntu, args.use_therapies ]), 'Select only one dataset for testing'
    if args.use_ntu and args.path_ntu_anns is None: raise ValueError('NTU annotations path not specified')
    
    
    
    if not args.use_gpu: os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        pass
            
    
    
    model, model_params = prediction_utils.load_model(args.path_model, False)
    model_params['skip_frames'] = []
    max_seq_len = model_params['max_seq_len']
    model_params['max_seq_len'] = 0  
    
    if args.use_ntu:
        with open(args.path_ntu_anns, 'r') as f: ntu_annotations = f.read().splitlines()
        ntu_annotations = list(np.random.choice(ntu_annotations, size=args.max_clips, replace=False))
        
        if args.test_offline:
            print(' ** Performing NTU OFFLINE predictions ** ')
            model.set_encoder_return_sequences(True)
            iterator = ntu_batch_iterator(ntu_annotations, model_params)
            main(model, False, iterator, max_seq_len, args.plot_results)
        if args.test_online:
            print(' ** Performing NTU ONLINE predictions ** ')
            model.set_encoder_return_sequences(False)
            iterator = ntu_per_frame_iterator(ntu_annotations, model_params, max_seq_len)
            main(model, True, iterator, max_seq_len, args.plot_results)
            
    elif args.use_therapies:
        raw_data_path = './datasets/therapies_dataset/'
        video_skels = pickle.load(open(os.path.join(raw_data_path, 'video_skels_v2.pckl'), 'rb'))
        video_skels = { k:v[2] for k,v in video_skels.items() }
        
        if args.test_offline:
            print(' ** Performing THERAPIES OFFLINE predictions ** ')
            model.set_encoder_return_sequences(True)
            iterator = ther_batch_iterator(video_skels, model_params)
            main(model, False, iterator, max_seq_len, args.plot_results)
        if args.test_online:
            print(' ** Performing THERAPIES ONLINE predictions ** ')
            model.set_encoder_return_sequences(False)
            iterator = ther_per_frame_iterator(video_skels, model_params, max_seq_len)
            main(model, True, iterator, max_seq_len, args.plot_results)

