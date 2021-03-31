#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:09:04 2020

@author: asabater
"""

import sys
sys.path.append('..')

import os
import pickle
import datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skvideo.io import FFmpegWriter
from tqdm import tqdm

import prediction_utils
from data_generator import load_scaler, get_pose_data_v2, average_wrong_frame_skels
from eval_utils import print_3d_skeleton, get_distance

from tensorboard.backend.event_processing import event_accumulator
import tensorflow as tf



# Timelines params
VLINE_W = 1.5
VLINE_REF = 2
REF_FRAMES_FREQ = 25
ANCHOR_COLOR = '#6600cc'
TARGET_COLOR = '#0066cc'
DETS_COLORS = {'med': '#F77800', 'good': '#ffcc66', 'excel': '#00cc66'}




# =============================================================================
# Get metric thresholds
# =============================================================================

def get_metric_thr(path_model, metrics=['cos', 'js'], mode='_bgnd'):
    tb_file = path_model + '/metrics/' + os.listdir(path_model + '/metrics')[-1]
    ea = event_accumulator.EventAccumulator(tb_file).Reload()
    metric_thr = {}
    for metric in metrics:
        k_acc = 'ther_trip{}_full_{}_acc'.format(mode, metric)
        k_p = 'ther_trip{}_full_{}_dist_p'.format(mode, metric)
        k_n = 'ther_trip{}_full_{}_dist_n'.format(mode, metric)
        
        if k_acc not in ea.tensors.Keys(): continue
        
        step = max(ea.Tensors(k_acc), key=lambda x: tf.make_ndarray(x.tensor_proto)).step
        acc = [ float(tf.make_ndarray(t.tensor_proto)) for t in ea.Tensors(k_acc) if t.step == step][0]
        dist_p = [ float(tf.make_ndarray(t.tensor_proto)) for t in ea.Tensors(k_p) if t.step == step][0]
        dist_n = [ float(tf.make_ndarray(t.tensor_proto)) for t in ea.Tensors(k_n) if t.step == step][0]
        
        thr_med = (dist_p + dist_n) / 2
        thr_excel = dist_p
        thr_good = (thr_med + thr_excel) / 2
        metric_thr[metric] = {'med': thr_med, 'good': thr_good, 'excel': thr_excel}
    return metric_thr



def get_video_distances(pf, actions_data, video_skels, model, model_params, batch=None, in_memory_callback=False, cache={}):

    anchors = actions_data[(actions_data.preds_file == pf) & (actions_data.is_therapist == 'y')]
    anchors_preds_filename = anchors.preds_filename.drop_duplicates().tolist()[0]
    
    if not in_memory_callback or not anchors_preds_filename in cache:
        anchors_tempos, anchors_num_frames, anchors_skels_raw = video_skels[anchors_preds_filename]
        if model_params['average_wrong_skels']: anchors_skels_raw = average_wrong_frame_skels(anchors_skels_raw)
        anchors_skels = get_pose_data_v2(anchors_skels_raw, validation=True, **model_params)
        # anchors_skels = np.expand_dims(anchors_skels, axis=0)    
        # anchor_preds = np.array(model.get_embedding(anchors_skels))[0]
        anchors_y_true = [ any((anchors.init_frame < num_frame) & (anchors.end_frame > num_frame)) \
                          for num_frame in anchors_num_frames ]
            
            
        # anchors_ids = [ any((anchors.init_frame < num_frame) & (anchors.end_frame > num_frame)) \
                          # for num_frame in anchors_num_frames ]
    
        # print(len(anchors_tempos), len(anchors_num_frames), len(list(anchors_skels_raw)), 
        #       len(list(anchors_skels)), len(anchors_y_true))
        data_anchor = pd.DataFrame({'ts': anchors_tempos, 'num_frame': anchors_num_frames, 
                                    'skels_raw': list(anchors_skels_raw),
                                    'skels_feats': list(anchors_skels),
                                    # 'emb': list(anchor_preds),
                                    'emb': None,
                                    'y_true': anchors_y_true,
                                    'action_id': None})
        for _,row in anchors.iterrows(): data_anchor.loc[(data_anchor.num_frame > row.init_frame) & (data_anchor.num_frame < row.end_frame), 'action_id'] = row.comments
        if in_memory_callback: cache[anchors_preds_filename] = data_anchor
    else:
        data_anchor = cache[anchors_preds_filename]
    

    targets = actions_data[(actions_data.preds_file == pf) & (actions_data.is_therapist == 'n')]
    targets_preds_filename = anchors_preds_filename.replace('keypoints', 'keypointschild')
    if not targets_preds_filename in cache:
        targets_tempos, targets_num_frames, targets_skels_raw = video_skels[targets_preds_filename]
        if model_params['average_wrong_skels']: targets_skels_raw = average_wrong_frame_skels(targets_skels_raw)
        targets_skels = get_pose_data_v2(targets_skels_raw, validation=True, **model_params)
        # targets_skels = np.expand_dims(targets_skels, axis=0)    
        # targets_preds = np.array(model.get_embedding(targets_skels))[0]
        targets_is_target = [ any((targets.init_frame < num_frame) & (targets.end_frame > num_frame)) \
                          for num_frame in targets_num_frames ]
        targets_is_anchor = [ any((anchors.init_frame < num_frame) & (anchors.end_frame > num_frame)) \
                          for num_frame in targets_num_frames ]
            
    
        data_target = pd.DataFrame({'ts': targets_tempos, 'num_frame': targets_num_frames, 
                                    'skels_raw': list(targets_skels_raw),
                                    'skels_feats': list(targets_skels),
                                    # 'emb': list(targets_preds),
                                    'emb': None,
                                    'tl_frame': None,
                                    'is_anchor': targets_is_anchor,
                                    'is_target': targets_is_target,
                                    'action_id': None,
                                    # 'tl_frame': list(range(len(targets_preds))),
                                    'target_id': None, 'anchor_id': None
                                    })
        
        # Add targets ids
        for _,row in targets.iterrows(): data_target.loc[(data_target.num_frame > row.init_frame) & (data_target.num_frame < row.end_frame), 'target_id'] = row.comments
        for _,row in anchors.iterrows(): data_target.loc[(data_target.num_frame > row.init_frame) & (data_target.num_frame < row.end_frame), 'anchor_id'] = row.comments
        cache[targets_preds_filename] = data_target
    else:
        data_target = cache[targets_preds_filename]



    # anchor_preds = np.array(model.get_embedding(anchors_skels))[0]
    # targets_preds = np.array(model.get_embedding(targets_skels))[0]
    
    data_anchor.loc[:,'emb'] = pd.Series(list(np.array(model.get_embedding(np.expand_dims(np.array(data_anchor.skels_feats.tolist()), axis=0), batch=batch))[0]))
    data_target.loc[:,'emb'] = pd.Series(list(np.array(model.get_embedding(np.expand_dims(np.array(data_target.skels_feats.tolist()), axis=0), batch=batch))[0]))

    return data_anchor, data_target, anchors, targets



def get_anchor_embs_by_strategy(data_anchor, anchors_info, anchor_strategy):
    if anchor_strategy == 'last': # Last anchor frame
        anchor_embs = { row.end_frame:[data_anchor[data_anchor.action_id == row.comments].emb.iloc[-1]] for _,row in anchors_info.iterrows() }
    elif anchor_strategy.startswith('perc'):    # Pick embeddings by position percentage
        percs = list(map(float, anchor_strategy[5:].split('_')))
        anchor_embs = {}
        for _,row in anchors_info.iterrows():
            frame_embs = data_anchor[data_anchor.action_id == row.comments].emb
            anchor_embs[row.end_frame] = [ frame_embs.iloc[int((len(frame_embs)-1)*perc)] for perc in percs ]
    elif anchor_strategy.startswith('pos'):    # Pick embeddings by position percentage
        positions = list(map(int, anchor_strategy[4:].split('_')))
        anchor_embs = {}
        for _,row in anchors_info.iterrows():
            frame_embs = data_anchor[data_anchor.action_id == row.comments].emb
            anchor_embs[row.end_frame] = [ frame_embs.iloc[pos] for pos in positions ]
    else: raise ValueError('anchor_strategy "{}" not implemented'.format(anchor_strategy))
    
    return anchor_embs

def calculate_distances(data_anchor, data_target, anchors_info, metric_thr, 
                        anchor_strategy='last',        # how to pick the anchors
                        dist_to_anchor_func='mean',    # Mean or median od the distances to anchor
                        last_anchor=True,            # Use last anchor or all of them
                        top=False,                      # Use only best distances [perc0.2]
                        ):      
    # anchors to calculate dists -> { end_frame : list of embs } 
    anchor_embs = get_anchor_embs_by_strategy(data_anchor, anchors_info, anchor_strategy)
        
        
    min_frame = min(anchor_embs.keys())
    detections = { metric:[] for metric,_ in metric_thr.items() }
    for _,row in data_target.iterrows():
        
        
        anchors_to_compare = { anchor_frame:embs for anchor_frame,embs in anchor_embs.items() if anchor_frame < row.num_frame }
        if len(anchors_to_compare) == 0: # Skip comparison if there are no anchors
            for metric,_ in metric_thr.items(): detections[metric].append(None)
            continue
        
        if last_anchor:     # Use only the last anchor
            anchors_to_compare = dict([max(anchor_embs.items(), key=lambda x: x[0])])
        
        # Flatten the anchor embeddings
        anchors_to_compare = sum(anchors_to_compare.values(), [])
        
        for metric, thrs in metric_thr.items():
            metric_dists = sorted([ get_distance(metric, atc, row.emb) for atc in anchors_to_compare ])
            
            if top != False:
                if top.startswith('perc'):
                    # num_goods = max(1, int(len(metric_dists)*float(top[4:])))
                    top_perc = float(top[4:])
                    num_goods = int(len(metric_dists)*top_perc)
                    if num_goods == 0: num_goods = 1 if top_perc > 0 else -1
                    metric_dists = metric_dists[:num_goods] if num_goods>0 else metric_dists[num_goods:]
                    
                else: raise ValueError('top distance param not handled:', top)
                
            if dist_to_anchor_func == 'mean':
                dist = np.mean(metric_dists)
            elif dist_to_anchor_func == 'median':
                dist = np.median(metric_dists)
            elif dist_to_anchor_func == 'min':
                dist = np.min(metric_dists)
            else:
                raise ValueError('dist_to_anchor_func not handled')
                
            detections[metric].append(dist)
                
        

    for metric,thr in metric_thr.items(): 
        data_target.loc[:,metric+'_dist'] = detections[metric]
        data_target.loc[:,metric+'_det'] = data_target[metric+'_dist']<thr['med']
        
        data_target.loc[:,metric+'_med'] = data_target[metric+'_dist'].apply(lambda x: x<thr['med'] and x>thr['good'])
        data_target.loc[:,metric+'_good'] = data_target[metric+'_dist'].apply(lambda x: x<=thr['good'] and x>thr['excel'])
        data_target.loc[:,metric+'_excel'] = data_target[metric+'_dist'].apply(lambda x: x<=thr['excel'])

        data_target.loc[:,metric+'_det_level'] = None
        if len(data_target.loc[data_target[metric+'_det'],metric+'_det_level']) > 0:
            data_target.loc[data_target[metric+'_det'],metric+'_det_level'] =  data_target[data_target[metric+'_det']][[metric+'_med', metric+'_good', metric+'_excel']].apply(lambda x: np.array(['med', 'good', 'excel'])[x.values][0], axis=1).tolist()

    return data_target



def get_current_timeline(curr_frame, data_target, metric, label_left=None, label_right=None, 
                         width=100, plot=False):

    fig = plt.figure(figsize=(6,0.55), dpi=150)
    plt.axis('equal')
    plt.axis('off')

    total_frames = data_target.tl_frame.max()
    total_frames -= 1
    if curr_frame is None: curr_frame = total_frames
    curr_frame = curr_frame * width / total_frames
    
    # Total frames
    plt.hlines(y=0, xmin=0, xmax=width, color='grey', linestyle='-', linewidth=1,zorder=10)
    plt.vlines(0, -VLINE_REF, VLINE_REF, colors='k', linestyle='-', linewidth=1,zorder=10)
    plt.vlines(width, -VLINE_REF, VLINE_REF, colors='grey', linestyle='-', linewidth=1,zorder=10)
    # Current frames
    plt.hlines(y=0, xmin=0, xmax=curr_frame, color='k', linestyle='-', linewidth=1,zorder=10)
    plt.vlines(curr_frame, -VLINE_REF, VLINE_REF, colors='k', linestyle='-', linewidth=1,zorder=10)
    # Reference frames
    for i in range(REF_FRAMES_FREQ,total_frames,REF_FRAMES_FREQ):
        plt.vlines(i*width/total_frames, -VLINE_REF, VLINE_REF, colors='grey', 
                    linestyle=(0, (1, 1)), linewidth=1.3,zorder=0, alpha=0.6)
    
    
    # Anchors
    anchors = [ (g.tl_frame.min(),g.tl_frame.max()) for i,g in data_target.groupby('anchor_id') ]
    anchors = [ (init * width / total_frames, end * width / total_frames) for init,end in anchors ]
    for init, end in anchors:
        if init > curr_frame: continue
        end = min(end, curr_frame)
        
        plt.vlines(init, -VLINE_W, VLINE_W, colors=ANCHOR_COLOR, linestyle='-', alpha=0.6)
        plt.vlines(end, -VLINE_W, VLINE_W, colors=ANCHOR_COLOR, linestyle='-', alpha=0.6)
        plt.hlines(y=0, xmin=init, xmax=end, color=ANCHOR_COLOR, linestyle='-', linewidth=2, alpha=0.6,zorder=15)
    
    # Detections
    detections = data_target[data_target[metric+'_det']]
    if len(detections) == 0: detections = []
    else:
        detections = list(zip(detections.tl_frame.tolist(), detections[metric+'_det_level']))
                  # detections[[metric+'_med', metric+'_good', metric+'_excel']].apply(lambda x: np.array(['med', 'good', 'excel'])[x.values][0], axis=1).tolist()))
    # detections = [ (row.num_frame, color) for _,row in data_target[data_target[metric+'_det'].iterrows() ]
    detections = [ (det_frame * width / total_frames, color) for det_frame, color in detections ]
    for det_farame, color in detections:
        if det_farame > curr_frame: continue
        # plt.vlines(det, -VLINE_W, VLINE_W, colors=DETS_COLOR, linestyle='-', alpha=0.6)
        plt.vlines(det_farame, -VLINE_W, VLINE_W, colors=DETS_COLORS[color], linestyle='-', alpha=0.6)
    
    # # Targets
    targets = [ (g.tl_frame.min(),g.tl_frame.max()) for i,g in data_target.groupby('target_id') ]
    targets = [ (init * width / total_frames, end * width / total_frames) for init,end in targets ]
    for init, end in targets:
        if init > curr_frame: continue
        end = min(end, curr_frame)
        plt.vlines(init, -VLINE_W, VLINE_W, colors=TARGET_COLOR, linestyle='-', alpha=0.6)
        plt.vlines(end, -VLINE_W, VLINE_W, colors=TARGET_COLOR, linestyle='-', alpha=0.6)
        plt.hlines(y=0, xmin=init, xmax=end, color=TARGET_COLOR, linestyle='-', linewidth=2, alpha=0.6,zorder=15)
    
    if label_left is not None:
        plt.text(0,2.5, label_left)    
    if label_right is not None:
        plt.text(100,3, label_right, ha='right', fontsize=7)
    
    plt.text(0, -4.5, data_target.num_frame.min().astype(int), fontsize=7)
    plt.text(100,-4.5, data_target.num_frame.max().astype(int), fontsize=7, ha='right')
    
    canvas = FigureCanvas(fig)
    canvas.draw()       # draw the canvas, cache the renderer
    w, h = canvas.get_width_height()
    frame = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)
    
    if not plot:
        plt.close(fig)
        
    return frame

# timeline = get_current_timeline(None, data_target, metric, 
#                                 label_left=metric, label_right=label_right, width=100, plot=True)



def get_evaluation_stats(data_target, targets_info, metric_thr, DIST_TO_GT_TP, DIST_TO_GT_FP):
    # Supress lonely dets
    # TP if det is close to gt
    # FP if fet is far from gt

    # Num positive samples -> actions to detect
    # Num positive samples -> actions to detect
    P_num = data_target.target_id.dropna().drop_duplicates().size
    TP_dets = { metric:{ t:False for t in data_target.target_id.dropna().drop_duplicates() } for metric in metric_thr.keys() }
    FP_dets = { metric:[] for metric in metric_thr.keys() }
    for i,row in data_target.iterrows():
        for metric, thr in metric_thr.items():
            
            if not row[metric+'_det']: continue
            
            det_frame = row.num_frame
            targets_detected = targets_info[(targets_info.init_frame <= det_frame) & (targets_info.end_frame+DIST_TO_GT_TP > det_frame)]
            if len(targets_detected) != 0:
                # TP
                for _,t_row in targets_detected.iterrows(): TP_dets[metric][t_row.comments] = True
            else:
                # FP
                if all([ det_frame-v>DIST_TO_GT_FP for v in FP_dets[metric] ]):
                    FP_dets[metric].append(det_frame)
                
     
    final_stats = {}
    for metric in metric_thr.keys():
        
        TP_pf = sum(TP_dets[metric].values())
        FP_pf = len(FP_dets[metric])
        FN_pf = len(TP_dets[metric]) - TP_pf
        
        # if TP_pf == FP_pf == FN_pf == 0:
        #     precision_pf = 1; recall_pf = 1; f1_pf = 1
        # elif TP_pf == 0 and (FP_pf!=0 or FN_pf!=0): 
        #     precision_pf = 0; recall_pf = 0; f1_pf = 0
        # else:
        #     precision_pf = TP_pf / (TP_pf + FP_pf)
        #     recall_pf = TP_pf / (TP_pf + FN_pf)
        #     f1_pf = (2*precision_pf*recall_pf) / (precision_pf + recall_pf)


        if TP_pf == FP_pf == 0: precision_pf = 1
        elif TP_pf == 0 and FP_pf != 0: precision_pf = 0
        elif TP_pf != 0: precision_pf = TP_pf / (TP_pf + FP_pf)
        else: raise ValueError('Precision exception. TP_pf {}, FP_pf {}'.format(TP_pf, FP_pf))

        if TP_pf == FN_pf == 0: recall_pf = 1
        elif TP_pf == 0 and FN_pf != 0: recall_pf = 0
        elif TP_pf != 0: recall_pf = TP_pf / (TP_pf + FN_pf)
        else: raise ValueError('Recall exception. TP_pf {}, FN_pf {}'.format(TP_pf, FN_pf))
        
        if precision_pf == recall_pf == 0: f1_pf = 0
        else: f1_pf = (2*precision_pf*recall_pf) / (precision_pf + recall_pf)

            

        stats_str = 'TP {} | FP {} | FN {} || Precision: {:.2f} | Recall: {:.2f} | F1: {:.2f}'.format(
                TP_pf, FP_pf, FN_pf, precision_pf, recall_pf, f1_pf)

        final_stats[metric] = {
                'TP': TP_pf, 'FP': FP_pf, 'FN': FN_pf, 
                'precision': precision_pf, 'recall': recall_pf,
                'f1': f1_pf,
                'stats_str': stats_str
            }
    
    return final_stats



def draw_skel_bbox(skel, ax, c, label):
    xmin, xmax, zmin, zmax = [skel[:,0].min(), skel[:,0].max(), skel[:,2].min(), skel[:,2].max()]
    y_mean = skel[:,1].mean()
    ax.plot([xmin, xmax], [y_mean,y_mean], [zmin, zmin], c=c)
    ax.plot([xmin, xmax], [y_mean,y_mean], [zmax, zmax], c=c)
    ax.plot([xmin, xmin], [y_mean,y_mean], [zmin, zmax], c=c)
    ax.plot([xmax, xmax], [y_mean,y_mean], [zmin, zmax], c=c)      
    ax.text(xmin+0.01, y_mean, zmin-0.08, label)

def render_video(data_anchor, data_target, anchors_info, final_stats, 
                 output_video_filename, metric_thr, 
                 max_width, max_height, font, output_fps=12):
    
    
    data_target['anchor_skel_raw'] = [ min(data_anchor.iterrows(), key=lambda r: abs(r[1].num_frame - row.num_frame))[1].skels_raw for _,row in data_target.iterrows() ]
    
    
    total_coords = np.array(data_target.anchor_skel_raw.tolist() + data_target.skels_raw.tolist())
    X,Y,Z = total_coords[:,:,0], total_coords[:,:,2], total_coords[:,:,1]
    # max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    max_p, min_p = 99, 1
    max_range = np.array([np.percentile(X, max_p)-np.percentile(X, min_p), 
                          np.percentile(Y, max_p)-np.percentile(Y, min_p), 
                          np.percentile(Z, max_p)-np.percentile(Z, min_p)]).max() / 2.0
    # print([np.percentile(X, max_p), np.percentile(X, min_p), 
    #         np.percentile(Y, max_p), np.percentile(Y, min_p), 
    #         np.percentile(Z, max_p), np.percentile(Z, min_p)])
    # mid_x, mid_y, mid_z = (X.max()+X.min()) * 0.5, (Y.max()+Y.min()) * 0.5, (Z.max()+Z.min()) * 0.5
    mid_x, mid_y, mid_z = (np.percentile(X, max_p)+np.percentile(X, min_p)) * 0.5, \
                            (np.percentile(Y, max_p)+np.percentile(Y, min_p)) * 0.5, \
                            (np.percentile(Z, max_p)+np.percentile(Z, min_p)) * 0.5
    x_lim = (mid_x - max_range, mid_x + max_range)
    y_lim = (mid_y - max_range, mid_y + max_range)
    z_lim = (mid_z - max_range, mid_z + max_range)
    
    x_floor, y_floor = np.arange(*x_lim, 0.05), np.arange(*y_lim, 0.05)
    x_floor, y_floor = np.meshgrid(x_floor,y_floor)  
    z_floor = np.full(x_floor.shape, np.percentile(Z, 5))
    del total_coords; del X; del Y; del Z
    
    

    sess = anchors_info.iloc[0]['patient'] + '/' + anchors_info.iloc[0]['session'] + ' | ' + anchors_info.iloc[0]['preds_filename']
    max_tl_frame, max_num_frame = data_target.tl_frame.max(), int(data_target.num_frame.max())
    num_ex, label_ex = anchors_info.iloc[0].ex_num, anchors_info.iloc[0].action

    
    writer = FFmpegWriter(output_video_filename,
 								   inputdict={'-r': str(output_fps)},
 								   outputdict={'-r': str(output_fps)})   
    
    # for _,row in tqdm(data_target[60:90].iterrows(), total=len(data_target)):
    for _,row in tqdm(data_target.iterrows(), total=len(data_target)):

        fig = plt.Figure(figsize=(18,9), dpi=150)
        ax = fig.add_subplot(1,1,1, projection='3d')        
        ax.view_init(10, -60)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_zlim(*z_lim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z') 
        
        
        # Plot skeleton and floor
        print_3d_skeleton(ax, row.anchor_skel_raw[:, [0,2,1]], color=ANCHOR_COLOR)
        print_3d_skeleton(ax, row.skels_raw[:, [0,2,1]], color=TARGET_COLOR)
        ax.plot_wireframe(x_floor, y_floor, z_floor, alpha=0.2, rcount=80)

        
        # Print gt bounding boxes
        if row.is_anchor:
            draw_skel_bbox(row.anchor_skel_raw[:, [0,2,1]], ax, c=ANCHOR_COLOR, label='Recording Anchor')
        if row.is_target:
            draw_skel_bbox(row.skels_raw[:, [0,2,1]], ax, c=TARGET_COLOR, label='Recording Anchor')

        
        # Print metric distances
        
        xmin, xmax, zmin, zmax = [row.skels_raw[:,0].min(), row.skels_raw[:,0].max(), row.skels_raw[:,1].min(), row.skels_raw[:,1].max()]
        y_mean = row.skels_raw[:,2].mean()
        for i,(metric,thr) in enumerate(metric_thr.items()):
            if not row[metric+'_det']: 
                c = 'k'; weight = None
            else: 
                c = DETS_COLORS[row[metric+'_det_level']]; weight = 'bold'

            ax.text(xmin+0.01, y_mean, zmin-((i+2)*0.08), 
                    ' - {}: {:.2f}'.format(metric, row[metric+'_dist']), 
                    weight=weight, color=c)        
        
        

        # Get plot as Image
        canvas = FigureCanvas(fig)
        canvas.draw()       # draw the canvas, cache the renderer
        w, h = canvas.get_width_height()
        frame = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)
        frame = frame[:, ~np.all(np.all(frame==255, axis=0), axis=1), :]
        frame = frame[~np.all(np.all(frame==255, axis=1), axis=1), :, :]
        
        h_orig, w_orig = frame.shape[:2]
        w_rel, h_rel = w_orig / max_width, h_orig / max_height
            
        frame = Image.fromarray(frame)
        if w_rel < h_rel: new_size = (int(w_orig/h_rel), int(h_orig/h_rel))
        else: new_size = (int(w_orig/w_rel), int(h_orig/w_rel))        
        frame = frame.resize(new_size, Image.ANTIALIAS)
        frame_new = Image.new("RGB", (max_width, max_height), (255, 255, 255))
        frame_new.paste(frame, (max_width-new_size[0], max_height-new_size[1]))
        frame = frame_new; del frame_new
        
        
        timelines = []
        for metric in metric_thr.keys():
            timeline = get_current_timeline(row.tl_frame, data_target, metric, 
                                        label_left=metric, 
                                        # label_right='aaaa',
                                        label_right=final_stats[metric]['stats_str'], 
                                        width=100, plot=False)
            timelines.append(timeline)        
        timelines = np.concatenate(timelines, axis=0)
        timelines = timelines[:, ~np.all(np.all(timelines==255, axis=0), axis=1), :]
        # timelines = timelines[~np.all(np.all(timelines==255, axis=1), axis=1), :, :]
        timelines = Image.fromarray(timelines)        
        tl_width = int(max_width*0.5)
        tl_height = int(timelines.size[1] * tl_width / timelines.size[0])
        timelines = timelines.resize((tl_width, tl_height), Image.ANTIALIAS)
        frame.paste(timelines, (max_width-tl_width-18, 0))        
        

        draw = ImageDraw.Draw(frame)
        # Session
        draw.text((20,0), sess, fill=(0,0,0), font=font)
        # num_frame
        # draw.text((20,25), "#{}".format(num_frame+min_frame), fill=(0, 0, 0), font=font)
        draw.text((20,30), "{:<3} / {} || {:<3} / {}".format(row.tl_frame, max_tl_frame, int(row.num_frame), max_num_frame), fill=(0, 0, 0), font=font)
        # timestamp
        draw.text((20,60), str(datetime.datetime.fromtimestamp(row.ts)), fill=(0, 0, 0), font=font)
        # Exercise
        draw.text((20,90), "#{} | {}".format(num_ex, label_ex), fill=(0, 0, 0), font=font)
 
            
        frame = np.array(frame)
        writer.writeFrame(frame)  
        # break
        del fig; del ax; del frame; del draw; del timelines; del canvas
    
    writer.close()
    del writer


def get_therapies_metrics(model, actions_data, video_skels, metric_thr, 
                          model_params, skip_empty_targets, metrics=['cos', 'js'],
                          batch=None,
                          dist_params = {}, thr_strategy = None,
                          in_memory_callback=False, cache={}, 
                          FRAMES_BEFOR_ANCHOR=36, DIST_TO_GT_TP=64, DIST_TO_GT_FP=64):

    total_stats, total_labels = [], []
    pred_files = actions_data.preds_file.drop_duplicates().tolist()
    for num_file, pf in enumerate(pred_files):
        data_anchor, data_target, anchors_info, targets_info = get_video_distances(pf, 
                                   actions_data, video_skels, model, model_params, 
                                   batch = batch, 
                                   in_memory_callback=in_memory_callback, cache=cache)
        label = anchors_info.action.iloc[0]
        
        if skip_empty_targets and len(targets_info) == 0: continue
    
        if thr_strategy is not None and 'bfrcrop'in thr_strategy:
            metric_thr = get_dynamic_thr(data_anchor, data_target, anchors_info, metrics, 
                                         thr_strategy=thr_strategy)


        init_frame = anchors_info.init_frame.min() - FRAMES_BEFOR_ANCHOR
        data_anchor = data_anchor[data_anchor.num_frame >= init_frame]
        data_target = data_target[data_target.num_frame >= init_frame]
        data_target = data_target.assign(tl_frame=list(range(len(data_target))))
 
        if thr_strategy is not None and 'aftcrop'in thr_strategy:
            metric_thr = get_dynamic_thr(data_anchor, data_target, anchors_info, metrics, 
                                         thr_strategy=thr_strategy, dist_params=dist_params)
            
        data_target = calculate_distances(data_anchor, data_target, anchors_info, metric_thr, **dist_params)
        
        final_stats = get_evaluation_stats(data_target, targets_info, metric_thr, DIST_TO_GT_TP, DIST_TO_GT_FP)
        total_stats.append(final_stats)
        total_labels.append(label)
    
    stats = {}
    for metric in metric_thr.keys():
        stats[metric] = {'precision': np.mean([ s[metric]['precision'] for s in total_stats ]), 
                         'recall': np.mean([ s[metric]['recall'] for s in total_stats ]), 
                         'f1': np.mean([ s[metric]['f1'] for s in total_stats ]),
                         'per_class_stats': {}}
        for label in set(total_labels):
            stats[metric]['per_class_stats'][label] = {
                    'precision': np.mean([ s[metric]['precision'] for s,l in zip(total_stats,total_labels) if l==label ]), 
                    'recall': np.mean([ s[metric]['recall'] for s,l in zip(total_stats,total_labels) if l==label ]), 
                    'f1': np.mean([ s[metric]['f1'] for s,l in zip(total_stats,total_labels) if l==label ])
                    }
    
    return stats


def get_dynamic_thr(data_anchor, data_target, anchors_info, metrics, thr_strategy, dist_params):
    # Get first anchor ending
    first_anchor_end = anchors_info.end_frame.min()
    # first_anchor_emb = data_anchor[data_anchor.num_frame <= first_anchor_end].iloc[-1].emb
    
    first_anchor_embs = get_anchor_embs_by_strategy(data_anchor, anchors_info, dist_params['anchor_strategy'])
    first_anchor_embs = { num_frame:emb for num_frame,emb in first_anchor_embs.items() if num_frame <= first_anchor_end}
    first_anchor_embs = sum(list(first_anchor_embs.values()), [])
    
    # Get init target distances
    init_target_embs = data_target[data_target.num_frame < first_anchor_end].emb.tolist()
    metric_distances = { metric:[ get_distance(metric, a_emb, t_emb) \
                                 for t_emb in init_target_embs \
                                 # for num_frame, first_anchor_emb in first_anchor_embs.items()
                                 for a_emb in first_anchor_embs
                                  ] \
                                 for metric in metrics }

    # print(metric_distances)
    # print(first_anchor_embs[0].shape, init_target_embs[0].shape)
    # for metric in metrics: print(metric, min(metric_distances[metric]), max(metric_distances[metric]))
    
    if 'perc' in thr_strategy:
        perc = int([ s for s in thr_strategy.split('_') if 'perc' in s ][0][4:])
        if 'fac' in thr_strategy: fac = float([ s for s in thr_strategy.split('_') if 'fac' in s ][0][4:])
        else: fac = 1
        thr = { metric:np.percentile(metric_distances[metric],perc)*fac for metric in metrics }
        
        if 'max' in thr_strategy:
            max_value = float([ s for s in thr_strategy.split('_') if 'max' in s ][0][3:])
            thr = { metric:min(thr[metric], max_value) for metric in metrics }
            
        metric_thr = { metric:{'med':thr[metric], 'good':0, 'excel':0 } \
                      for metric in metrics }
    else: raise ValueError('thr_strategy not handled:', thr_strategy)
    
    # print(metric_thr)
    return metric_thr
        
        

        

# %%



if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
      except RuntimeError as e:
        print(e)
        
        

    # =============================================================================
    # Load model
    # =============================================================================
    
    # path_model = '/mnt/hdd/ml_results/core/multiloss_ae_tcn_norm_skip_data_feats/0610_0329_model_3/'
    # path_model = '/mnt/hdd/ml_results/core/multiloss_ae_tcn_norm_skip/0607_1833_model_40/'
    # path_model = '/mnt/hdd/ml_results/core/multiloss_ae_tcn_norm_skip_data_feats/0609_1859_model_0/'
    # path_model = '/mnt/hdd/ml_results/core/multiloss_ae_tcn_norm_skip_data_feats/0612_1023_model_4/'

    # path_model = '/mnt/hdd/ml_results/core/multiloss_ae_tcn_norm_skip_data_feats/0614_2110_model_21/'
    # path_model = '/mnt/hdd/ml_results/core/multiloss_ae_tcn_norm_skip_data_feats_decoder_v2/0628_1748_model_2/'
    path_model = '/mnt/hdd/ml_results/core/multiloss_ae_tcn_norm_skip_data_feats_decoder_v3/0701_0156_model_7/'


    model, model_params = prediction_utils.load_model(path_model, False)
    # model.set_encoder_return_sequences(True)
    model_params['max_seq_len'] = 0   
    model_params['skip_frames'] = []
    
    
    
    # =============================================================================
    # Load dataset
    # =============================================================================
    
    raw_data_path = '/mnt/hdd/datasets_hdd/data_laura/one_shot_labels/'
    video_skels = pickle.load(open(os.path.join(raw_data_path, 'video_skels_v2.pckl'), 'rb'))
    actions_data = pickle.load(open(os.path.join(raw_data_path, 'actions_data_v2.pckl'), 'rb'))
    actions_data = actions_data[~actions_data.action.isin(['no', 'si'])]
    actions_data = actions_data.sort_values(by=['patient', 'session', 'video', 'ex_num'])
    
    
    
    
    # =============================================================================
    # Evaluate Precision / Recall
    # =============================================================================
    
    # TODO: FN only if more than one det points to the same false action
    
    # DIST_TO_GT_TP = 64
    # DIST_TO_GT_FP = 64
    DIST_TO_GT_TP = 32
    DIST_TO_GT_FP = 32
    FRAMES_BEFOR_ANCHOR = 32
    
    store_timelines = True
    plot_timelines = True
    store_video = True
    batch = None
    skip_empty_targets = True
    
    
    # dist_params = {'anchor_strategy': 'last', 'last_anchor': True, 'dist_to_anchor_func': 'mean' }
    
    
    metrics=['cos', 'js']
    # metrics=['cos']
    # metric_thr = get_metric_thr(path_model, metrics=metrics)
    # # metric_thr = {'cos': {'med': 0.522, 'good': 0.41, 'excel': 0.29}, 
    # #               'js': {'med': 0.527, 'good': 0.47, 'excel': 0.42}}
    # # metric_thr = {'cos': {'med': 0.27, 'good': 0.24, 'excel': 0.19}, 
    # #               'js': {'med': 0.34, 'good': 0.32, 'excel': 0.30}}
    # print(metric_thr)
    # thr_strategy = None
    # # thr_strategy = 'perc0_fact1_max0.53_aftcrop'


    dist_params = {'anchor_strategy': 'pos_-1_-2_-3', 'last_anchor': False, 'dist_to_anchor_func': 'min' }
    # thr_strategy = 'perc10_fact1_max0.480_aftcrop'
    # metric_thr = None
    # dist_params = {}
    thr_strategy = None
    metric_thr = {'cos': {'med': 0.40, 'good': 0.3, 'excel': 0.2}, 
                   'js': {'med': 0.48, 'good': 0.35, 'excel': 0.22}}
    
    
    
    max_width, max_height = 1280, 720
    font = ImageFont.truetype("NotoSerif-Regular.ttf", int(max_height*0.033))
    
    
    
    results_folder = os.path.join('/mnt/hdd/ml_results/core_renders/', *path_model.split('/')[-3:])
    if not os.path.isdir(results_folder): os.makedirs(results_folder)
    
    timelines_folder = results_folder+'timelines_{}-{}/'.format(batch, thr_strategy)
    if not os.path.isdir(timelines_folder): os.makedirs(timelines_folder)
    
    video_folder = results_folder+'videos_{}-{}/'.format(batch, thr_strategy)
    if not os.path.isdir(video_folder): os.makedirs(video_folder)
        
    



    model.set_encoder_return_sequences(batch is None)
    
    
    
    
    precision = { metric:[] for metric in metrics }
    recall = { metric:[] for metric in metrics }
    f1 = { metric:[] for metric in metrics }    
    

    in_memory_callback, cache = False, {}
    pred_files = actions_data.preds_file.drop_duplicates().tolist()
    # for num_file, pf in enumerate(pred_files[1:2]):
    # for num_file, pf in enumerate(tqdm(pred_files)):
    for num_file, pf in enumerate(pred_files):
    # for num_file, pf in enumerate(pred_files[1:2]):
    # for num_file, pf in enumerate(pred_files[2:3]):
    # for num_file, pf in enumerate(pred_files[16:17]):
    # for num_file, pf in enumerate(['152600', '153211', '153303', '150540']):
       
        # if num_file != 15: continue
        # min_file = 48
        # if num_file < min_file or num_file > min_file+10: continue
        print(num_file, '|', pf)
    
        data_anchor, data_target, anchors_info, targets_info = get_video_distances(pf, 
                                   actions_data, video_skels, model, model_params, 
                                   batch=batch,
                                   in_memory_callback=in_memory_callback, cache=cache)
        
        # TODO: .
        if skip_empty_targets and len(targets_info) == 0: continue
    
        
        # Crop video by anchor
        init_frame = anchors_info.init_frame.min() - FRAMES_BEFOR_ANCHOR
        data_anchor = data_anchor[data_anchor.num_frame >= init_frame]
        data_target = data_target[data_target.num_frame >= init_frame]
        data_target = data_target.assign(tl_frame=list(range(len(data_target))))
    
   
        if thr_strategy is not None:
            print('thr_strategy is not None')
            metric_thr = get_dynamic_thr(data_anchor, data_target, anchors_info, metrics, 
                                         dist_params=dist_params,
                                         thr_strategy=thr_strategy)

    
        data_target = calculate_distances(data_anchor, data_target, anchors_info, 
                                          metric_thr, **dist_params)
        
    
        timelines = []
        final_stats = get_evaluation_stats(data_target, targets_info, metric_thr, 
                                           DIST_TO_GT_TP=DIST_TO_GT_TP, DIST_TO_GT_FP=DIST_TO_GT_FP)
        for metric, thr in metric_thr.items():
            precision[metric].append(final_stats[metric]['precision'])
            recall[metric].append(final_stats[metric]['recall'])
            f1[metric].append(final_stats[metric]['f1'])
            
            
            if store_timelines or plot_timelines:
                timeline = get_current_timeline(None, data_target, metric, 
                                            label_left=metric, label_right=final_stats[metric]['stats_str'], 
                                            width=100, plot=False)
                timelines.append(timeline)
            print('{:<2} / {} || {:<4} || {}'.format(num_file+1, len(pred_files), metric, final_stats[metric]['stats_str']))
            
        if store_timelines or plot_timelines:
            timelines_filename = '{}/{}_{}_{}_{}_{}.png'.format(
                                        timelines_folder,
                                        num_file, anchors_info.iloc[0].action, 
                                        anchors_info.iloc[0].patient, anchors_info.iloc[0].session, anchors_info.iloc[0].preds_file)
            timelines_img = np.vstack(timelines)
            if store_timelines:
                timelines_img = Image.fromarray(timelines_img)
                timelines_img.save(timelines_filename)     
            if plot_timelines:
                plt.figure(dpi=150)
                plt.imshow(timelines_img)
                plt.axis('off')
                plt.show()
    
    
        if store_video:
            output_video_filename = '{}{}_{}_{}_{}_{}.mp4'.format(video_folder,
                                    num_file, anchors_info.iloc[0].action, 
                                    anchors_info.iloc[0].patient, anchors_info.iloc[0].session, anchors_info.iloc[0].preds_file)
            render_video(data_anchor, data_target, anchors_info, final_stats,
                         output_video_filename, metric_thr, max_width, max_height, 
                         font, output_fps=12)         
        
        # break
    
    
    print('='*80)
    for metric in metrics:
        print('{:<3} || Av. Precision: {:.3f} | Av. Recall: {:.3f} | Av. F1: {:.3f}'.format(metric,
                       np.mean(precision[metric]), 
                       np.mean(recall[metric]),
                       np.mean(f1[metric])))





