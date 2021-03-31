#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:02:47 2020

@author: asabater
"""

import pickle
import os
import json

import prediction_utils



os.environ['CUDA_VISIBLE_DEVICES'] = ''


raw_data_path = './therapies_annotations/'
video_skels_v2 = pickle.load(open(os.path.join(raw_data_path, 'video_skels_v2.pckl'), 'rb'))
actions_data_v2 = pickle.load(open(os.path.join(raw_data_path, 'actions_data_v2.pckl'), 'rb'))
# Remove worst actions
actions_data_v2 = actions_data_v2[~actions_data_v2.action.isin(['no', 'si'])]
actions_data_v2 = actions_data_v2.sort_values(by=['patient', 'session', 'video', 'ex_num'])



# # path_model = '/mnt/hdd/ml_results/core/multiloss_ae_tcn_norm_skip_data_feats_decoder_v3/0701_0156_model_7/'
# # path_model = '/mnt/hdd/ml_results/core/multiloss_ae_tcn_norm_skip_data_feats_decoder_v3/0708_1011_model_3/'
# # path_model = '/mnt/hdd/ml_results/core/multiloss_ae_tcn_norm_skip_data_feats_decoder_v3/0708_1437_model_4/'
# # path_model = '/mnt/hdd/ml_results/core/multiloss_ae_tcn_norm_skip_data_feats_decoder_v3/0709_2120_model_9/'
# path_model = '/mnt/hdd/ml_results/core/multiloss_ae_tcn_norm_skip_data_feats_decoder_v3/0323_1331_model_4/'
path_model = './pretrained_models/therapies_model_7/'
    


model, model_params = prediction_utils.load_model(path_model, False)
model_params['max_seq_len'] = 0   
model_params['skip_frames'] = []
model_params['average_wrong_skels'] = True
# metrics = ['euc', 'cos', 'js']
metrics = ['cos', 'js']

skip_empty_targets = True
# DIST_TO_GT_TP, DIST_TO_GT_FP = 64,64
DIST_TO_GT_TP, DIST_TO_GT_FP = 32,32
# DIST_TO_GT_TP, DIST_TO_GT_FP = 36,36
batch = None
FRAMES_BEFOR_ANCHOR = 32
dist_to_anchor_func = 'min'


model.set_encoder_return_sequences(batch is None)





import sys
sys.path.append('./eval_scripts/')
from eval_scripts import evaluation_metrics as em


# =============================================================================
# Get N-shot results
# =============================================================================


stats_filename_final_base = path_model + 'TP{}_FP{}_FBA{}_B{}_best-{}.json'.format(DIST_TO_GT_TP, DIST_TO_GT_FP, FRAMES_BEFOR_ANCHOR, batch, '{}')

# %%
one_shot_params = json.load(open(stats_filename_final_base.format('oneshot'), 'r'))
one_shot_res = {}
for metric in metrics:
    print(' ** Computing one-shot action detections for  {} metric'.format(metric))
    one_shot_res[metric] = em.get_therapies_metrics(model, actions_data_v2, video_skels_v2, one_shot_params[metric]['metric_thr'], 
	                                      model_params, skip_empty_targets, metrics=metrics, 
	                                      batch=batch,
	                                      dist_params=one_shot_params[metric]['dist_params'], 
                                          thr_strategy=one_shot_params[metric]['thr_strategy'],
	                                      in_memory_callback=False, cache={}, 
	                                      FRAMES_BEFOR_ANCHOR=FRAMES_BEFOR_ANCHOR, 
	                                      DIST_TO_GT_TP=DIST_TO_GT_TP, DIST_TO_GT_FP=DIST_TO_GT_FP)
    
    

# %%
few_shot_params = json.load(open(stats_filename_final_base.format('fewshot'), 'r'))
few_shot_res = {}
for metric in metrics:
    print(' ** Computing few-shot action detections for  {} metric'.format(metric))
    # few_shot_params[metric]['dist_params']['last_anchor'] = True
    # few_shot_params[metric]['dist_params']['anchor_strategy'] = 'last'
    few_shot_res[metric] = em.get_therapies_metrics(model, actions_data_v2, video_skels_v2, 
                                          few_shot_params[metric]['metric_thr'], 
	                                      model_params, skip_empty_targets, metrics=metrics, 
	                                      batch=batch,
	                                      dist_params=few_shot_params[metric]['dist_params'], 
                                          thr_strategy=few_shot_params[metric]['thr_strategy'],
	                                      in_memory_callback=False, cache={}, 
	                                      FRAMES_BEFOR_ANCHOR=FRAMES_BEFOR_ANCHOR, 
	                                      DIST_TO_GT_TP=DIST_TO_GT_TP, DIST_TO_GT_FP=DIST_TO_GT_FP)

print(few_shot_params[metric]['dist_params']['last_anchor'])
print(few_shot_params[metric]['dist_params']['anchor_strategy'])
print('{} |||  '.format('Few-shot        ') + ' || '.join([ '{} ( {:.2f}): P {:.3f} | R {:.3f} | F1 {:.3f} '.format(metric, few_shot_params[metric]['metric_thr'][metric]['med'], few_shot_res[metric][metric]['precision'], few_shot_res[metric][metric]['recall'], few_shot_res[metric][metric]['f1']) for metric in metrics ]))

 # %%   
    
few_shot_params = json.load(open(stats_filename_final_base.format('fewshot'), 'r'))
few_shot_dyn_params = json.load(open(stats_filename_final_base.format('fewshotdyn'), 'r'))
few_shot_dyn_res = {}
for metric in metrics:
    print(' ** Computing dynamic few-shot action detections for  {} metric'.format(metric))
    # if metric in few_shot_dyn_params: few_shot_dyn_params[metric]['dist_params']['last_anchor'] = True
    # if metric in few_shot_dyn_params: few_shot_dyn_params[metric]['dist_params']['anchor_strategy'] = 'last'
    few_shot_dyn_res[metric] = em.get_therapies_metrics(model, actions_data_v2, video_skels_v2, 
                                          few_shot_dyn_params[metric]['metric_thr'], 
	                                      model_params, skip_empty_targets, metrics=metrics, 
	                                      batch=batch,
	                                      dist_params=few_shot_dyn_params[metric]['dist_params'], 
                                          thr_strategy=few_shot_dyn_params[metric]['thr_strategy'],
	                                      in_memory_callback=False, cache={}, 
	                                      FRAMES_BEFOR_ANCHOR=FRAMES_BEFOR_ANCHOR, 
	                                      DIST_TO_GT_TP=DIST_TO_GT_TP, DIST_TO_GT_FP=DIST_TO_GT_FP)
print(few_shot_dyn_params[metric]['dist_params']['last_anchor'])
print(few_shot_dyn_params[metric]['dist_params']['anchor_strategy'])
print('{} |||  '.format('Dynamic few-shot') + ' || '.join([ '{} (<{:.2f}): P {:.3f} | R {:.3f} | F1 {:.3f} '.format(metric, few_shot_params[metric]['metric_thr'][metric]['med'], few_shot_dyn_res[metric][metric]['precision'], few_shot_dyn_res[metric][metric]['recall'], few_shot_dyn_res[metric][metric]['f1']) for metric in metrics ]))



# %%


print('{} |||  '.format('One-shot        ') + ' || '.join([ '{} ( {:.2f}): P {:.3f} | R {:.3f} | F1 {:.3f} '.format(metric, one_shot_params[metric]['metric_thr'][metric]['med'], one_shot_res[metric][metric]['precision'], one_shot_res[metric][metric]['recall'], one_shot_res[metric][metric]['f1']) for metric in metrics ]))
print('{} |||  '.format('Few-shot        ') + ' || '.join([ '{} ( {:.2f}): P {:.3f} | R {:.3f} | F1 {:.3f} '.format(metric, few_shot_params[metric]['metric_thr'][metric]['med'], few_shot_res[metric][metric]['precision'], few_shot_res[metric][metric]['recall'], few_shot_res[metric][metric]['f1']) for metric in metrics ]))
print('{} |||  '.format('Dynamic few-shot') + ' || '.join([ '{} (<{:.2f}): P {:.3f} | R {:.3f} | F1 {:.3f} '.format(metric, few_shot_params[metric]['metric_thr'][metric]['med'], few_shot_dyn_res[metric][metric]['precision'], few_shot_dyn_res[metric][metric]['recall'], few_shot_dyn_res[metric][metric]['f1']) for metric in metrics ]))
for metric in metrics: print(few_shot_dyn_params[metric]['dist_params'], few_shot_dyn_params[metric]['thr_strategy'])
print(path_model)


# %%

import pandas as pd

# =============================================================================
# Show per-class results
# =============================================================================

class_data_metric = {}
for metric in metrics:
	class_data = []
	class_data.append({  k:v['f1'] for k,v in one_shot_res[metric][metric]['per_class_stats'].items() })
	class_data.append({  k:v['f1'] for k,v in few_shot_res[metric][metric]['per_class_stats'].items() })
	class_data.append({  k:v['f1'] for k,v in few_shot_dyn_res[metric][metric]['per_class_stats'].items() })
	df = pd.DataFrame(class_data, index=['one_shot', 'few_shot', 'dyn'])
	
	df = df.rename(columns={'grande': 'big', 'alto': 'high', 'felize': 'happy', 'ciao': 'waving', 'dare': 'giving', 
					'piccolo': 'small', 'vieni': 'coming', 'aspetta': 'waiting', 'dove': 'where', 'io': 'me', 
					'pointing': 'pointing', 'fame': 'hungry', 'basso': 'down', 'arrabbiato': 'angry'})

	df = df.transpose().sort_values('one_shot', ascending=False)

	
	df['few_shot'] = df['few_shot'].apply(lambda x: '{:.2f}'.format(x)) \
					 + (df['few_shot'] - df['one_shot']).apply(lambda x: ' ({}{:.2f})'.format('+' if x >= 0 else '', x))
	df['dyn'] = df['dyn'].apply(lambda x: '{:.2f}'.format(x)) \
					 + (df['dyn'] - df['one_shot']).apply(lambda x: ' ({}{:.2f})'.format('+' if x >= 0 else '', x))
	df['one_shot'] = df['one_shot'].apply(lambda x: '{:.2f}'.format(x))
	
	class_data_metric[metric] = df
    
	print('*'*40 + '\n *** Metric: {} ***\n'.format(metric) + '*'*40)
	print(df)










