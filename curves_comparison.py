#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:10:57 2020

@author: asabater
"""

import sys
sys.path.append('..')


from eval_scripts import evaluation_metrics as em
# from eval_scripts import global_model_evaluation as gme
import prediction_utils
import json
import pickle
import os
import numpy as np




# =============================================================================
# Get P/C curves and best F1/P/R
#   Full Evaluation and on sample 32/64
# Repeat adding multiple anchors
# Repeat adding dynamic threshold
# =============================================================================



# =============================================================================
# Load model
# =============================================================================
import argparse
parser = argparse.ArgumentParser(description = 'Performs a speed test over NTU-120 skeleton clips')
parser.add_argument('--path_model', type=str, default='./pretrained_models/therapies_model_7/', required=True, help='path to the prediction model')
parser.add_argument('--force_all', action='store_true', help='force the calculation of new parameters')
args = parser.parse_args()




# =============================================================================
# Load data
# =============================================================================
raw_data_path = './datasets/therapies_dataset/'
video_skels_v2 = pickle.load(open(os.path.join(raw_data_path, 'video_skels_v2.pckl'), 'rb'))
actions_data_v2 = pickle.load(open(os.path.join(raw_data_path, 'actions_data_v2.pckl'), 'rb'))
# Remove not useful actions
actions_data_v2 = actions_data_v2[~actions_data_v2.action.isin(['no', 'si'])]

actions_data_v2 = actions_data_v2.sort_values(by=['patient', 'session', 'video', 'ex_num'])
in_memory_callback = False
cache = {}





os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        

model, model_params = prediction_utils.load_model(args.path_model, False)
model_params['max_seq_len'] = 0   
model_params['skip_frames'] = []
model_params['average_wrong_skels'] = True
# metrics = ['euc', 'cos', 'js']
metrics = ['cos', 'js']

skip_empty_targets = True
DIST_TO_GT_TP, DIST_TO_GT_FP = 32,32
batch = None
FRAMES_BEFOR_ANCHOR = 32
dist_to_anchor_func = 'min'


model.set_encoder_return_sequences(batch is None)


total_data_target = {}
total_targets_info = {}
force_default, force_few, force_dyn = False, False, False



# =============================================================================
# Precision / Recall curve
# =============================================================================

from tqdm import tqdm



def get_pr_curve_data(skip_empty_targets, total_data_target, total_targets_info, thresholds, 
                      dist_params, metrics, batch,
                      FRAMES_BEFOR_ANCHOR, DIST_TO_GT_TP, DIST_TO_GT_FP):

    metric_thr = { metric: {'med':0.0, 'good':0.0, 'excel':0.0} for metric in metrics }

    pred_files = actions_data_v2.preds_file.drop_duplicates().tolist()
    for num_file, pf in enumerate(pred_files):
    
        data_anchor, data_target, anchors_info, targets_info = em.get_video_distances(pf, 
                                   actions_data_v2, video_skels_v2, 
                                   model, model_params, 
                                   batch=batch,
                                   in_memory_callback=in_memory_callback, cache=cache)
        
    
        init_frame = anchors_info.init_frame.min() - FRAMES_BEFOR_ANCHOR
        data_anchor = data_anchor[data_anchor.num_frame >= init_frame]
        data_target = data_target[data_target.num_frame >= init_frame]
        data_target = data_target.assign(tl_frame=list(range(len(data_target))))
        
        data_target = em.calculate_distances(data_anchor, data_target, anchors_info, metric_thr, **dist_params)
        
        total_data_target[pf] = data_target[['num_frame', 'target_id'] + [ m+'_dist' for m in metrics ]]
        total_targets_info[pf] = targets_info
    
    
    
    curve_data = {}
    curve_results = { metric:{} for metric in metrics }
    for thr in tqdm(thresholds):
        curve_data[thr] = []
        for num_file, pf in enumerate(pred_files[1:]):
            
            data_target = total_data_target[pf]
            targets_info = total_targets_info[pf]
            
            if skip_empty_targets and len(targets_info) == 0: continue
            
            for metric in metrics: data_target[metric+'_det'] = data_target[metric+'_dist'] < thr
            
            final_stats = em.get_evaluation_stats(
                data_target, targets_info, metric_thr, 
                DIST_TO_GT_TP, DIST_TO_GT_FP)
    
            curve_data[thr].append(final_stats)
            
        for metric in metrics:
            curve_results[metric][thr] = {
                    'precision': np.mean([ d[metric]['precision'] for d in curve_data[thr] ]),
                    'recall': np.mean([ d[metric]['recall'] for d in curve_data[thr] ]),
                    'f1': np.mean([ d[metric]['f1'] for d in curve_data[thr] ])
                }
            
    return curve_results


# %%


best_f1_metric = { m:{'f1': 0.0} for m in metrics }


base_filename = args.path_model + 'TP{}_FP{}_FBA{}_B{}'.format(DIST_TO_GT_TP, DIST_TO_GT_FP, FRAMES_BEFOR_ANCHOR, batch)



# =============================================================================
# ONE-SHOT
# =============================================================================


thresholds = sorted(list(set(np.arange(0.1,0.4,0.05).tolist() + np.arange(0.42,0.7,0.02).tolist() + np.arange(0.7,0.9,0.05).tolist())))

filename_default = base_filename + '_default.pckl'
if args.force_all or force_default or not os.path.isfile(filename_default):
    curve_results_default = get_pr_curve_data(skip_empty_targets, total_data_target, total_targets_info, thresholds,
                          dist_params={}, metrics = metrics, batch = batch,
                          FRAMES_BEFOR_ANCHOR=FRAMES_BEFOR_ANCHOR, DIST_TO_GT_TP=DIST_TO_GT_TP, DIST_TO_GT_FP=DIST_TO_GT_FP)
    pickle.dump(curve_results_default, open(filename_default, 'wb'))
else:
    curve_results_default = pickle.load(open(filename_default, 'rb'))
print('Batch:', batch)
for metric in metrics:
    best = max(curve_results_default[metric].items(), key=lambda d: d[1]['f1'])
    if best[1]['f1'] > best_f1_metric[metric]['f1']:
            best_f1_metric[metric] = {
					'f1': best[1]['f1'],
					'res': best[1],
					'dist_params': {}, 'thr_strategy': None
				}
    # if best[1]['f1'] > best_f1_metric[metric][0]: best_f1_metric[metric] = (best[1]['f1'], 'def '+str(best[0]), (best[1]))
    print('{:<3} || {:.2f} || P {:.3f} | R {:.3f} | F1 {:.3f}'.format(metric, best[0], best[1]['precision'], best[1]['recall'], best[1]['f1']))
print('='*80)


# =============================================================================
# FEW-SHOT
# =============================================================================

if dist_to_anchor_func == 'min':
    dist_params = { 'anchor_strategy': 'pos_-1_-2_-3', 'last_anchor': False, 'dist_to_anchor_func': 'min'}
if dist_to_anchor_func == 'mean':
    dist_params = { 'anchor_strategy': 'pos_-1_-2_-3', 'last_anchor': False, 'dist_to_anchor_func': 'mean'}
if dist_to_anchor_func == 'median':
    dist_params = { 'anchor_strategy': 'pos_-1_-2_-3', 'last_anchor': False, 'dist_to_anchor_func': 'median'}
print(dist_params)
filename_dp = base_filename + '_dp[{}|{}|{}].pckl'.format(*list(dist_params.values()))
if args.force_all or force_few or not os.path.isfile(filename_dp):
    curve_results_dp = get_pr_curve_data(skip_empty_targets, total_data_target, total_targets_info, thresholds,
                          dist_params=dist_params, metrics = metrics, batch = batch,
                          FRAMES_BEFOR_ANCHOR=FRAMES_BEFOR_ANCHOR, DIST_TO_GT_TP=DIST_TO_GT_TP, DIST_TO_GT_FP=DIST_TO_GT_FP)
    pickle.dump(curve_results_dp, open(filename_dp, 'wb'))
else:
    curve_results_dp = pickle.load(open(filename_dp, 'rb'))
print('Batch:', batch)
for metric in metrics:
    best = max(curve_results_dp[metric].items(), key=lambda d: d[1]['f1'])
    if best[1]['f1'] > best_f1_metric[metric]['f1']:
            best_f1_metric[metric] = {
					'f1': best[1]['f1'],
					'res': best[1],
					'dist_params': dist_params, 'thr_strategy': None
				}
    print('{:<3} || {:.2f} || P {:.3f} | R {:.3f} | F1 {:.3f}'.format(metric, best[0], best[1]['precision'], best[1]['recall'], best[1]['f1']))
print('='*80)
print('='*80)
print('='*80)


del total_data_target; del total_targets_info



# =============================================================================
# DYNAMIC
# =============================================================================

print(dist_params)


thr_strategies = []
loc = '_aftcrop'


# for fact in [1, 0.95]:
for fact in [1]:
    # for perc in [0, 10, 20]:
    for perc in [10]:
        for max_value in ['auto', '_max0.500']:
            if max_value == 'auto':
                for metric in metrics:
                    max_value = '_max{:.3f}'.format(max(curve_results_dp[metric].items(), key=lambda d: d[1]['f1'])[0])
                    thr_strategies.append('perc{}_fact{}{}{}'.format(perc, fact, max_value, loc))
            else: 
                thr_strategies.append('perc{}_fact{}{}{}'.format(perc, fact, max_value, loc))
thr_strategies = list(set(thr_strategies))




for num_strat, thr_strategy in enumerate(thr_strategies):

    print('{:<3} / {} || {}'.format(num_strat+1, len(thr_strategies), thr_strategy))
    filename_dyn = base_filename + '_dp[{}|{}|{}]_{}.pckl'.format(*list(dist_params.values()), thr_strategy)
    if args.force_all or force_dyn or not os.path.isfile(filename_dyn):
        metric_thr = { metric: {'med':None, 'good':None, 'excel':None} for metric in metrics }
        stats_perc = em.get_therapies_metrics(model, actions_data_v2, video_skels_v2, metric_thr, 
                                          model_params, skip_empty_targets, metrics=metrics, 
                                          batch=batch,
                                          dist_params=dist_params, thr_strategy=thr_strategy,
                                          in_memory_callback=in_memory_callback, cache=cache, 
                                          FRAMES_BEFOR_ANCHOR=FRAMES_BEFOR_ANCHOR, 
                                          DIST_TO_GT_TP=DIST_TO_GT_TP, DIST_TO_GT_FP=DIST_TO_GT_FP)
        pickle.dump(stats_perc, open(filename_dyn, 'wb'))
    else:
        stats_perc = pickle.load(open(filename_dyn, 'rb'))
    s = ''
    for metric in metrics: 
        s += '{:<3} || P {:.3f} | R {:.3f} | F1 {:.3f}  |||   '.format(metric, 
        stats_perc[metric]['precision'], stats_perc[metric]['recall'], stats_perc[metric]['f1'])
        
        if stats_perc[metric]['f1'] > best_f1_metric[metric]['f1']: 
            best_f1_metric[metric] = {
					'f1': stats_perc[metric]['f1'],
					'res': {'precision': stats_perc[metric]['precision'],
                                      'recall': stats_perc[metric]['recall'],
                                      'f1': stats_perc[metric]['f1']},
					'dist_params': dist_params, 'thr_strategy': thr_strategy
				}
    
    print(s)
    print('='*80)
    

# %%

# =============================================================================
# Plot final results. Onw-shot, few-shot, dynamic
# =============================================================================

print('Batch:', batch, '||', DIST_TO_GT_TP, DIST_TO_GT_FP)
best_one_shot_metrics = {}
for metric in metrics:
    best_one_shot = max(curve_results_default[metric].items(), key=lambda d: d[1]['f1'])
    if best_one_shot[1]['f1'] > best_f1_metric[metric]['f1']: best_f1_metric[metric] = (best_one_shot[1]['f1'], 'def '+str(best_one_shot[0]), (best_one_shot[1]))
    print('{:<3} || {:.2f} || P {:.3f} | R {:.3f} | F1 {:.3f}'.format(metric, best_one_shot[0], best_one_shot[1]['precision'], best_one_shot[1]['recall'], best_one_shot[1]['f1']))
    best_one_shot_metrics[metric] = best_one_shot
print('='*80)
print('='*80)
print(filename_dp)
best_few_shot_metrics = {}
for metric in metrics:
    best_few_shot = max(curve_results_dp[metric].items(), key=lambda d: d[1]['f1'])
    if best_few_shot[1]['f1'] > best_f1_metric[metric]['f1']: best_f1_metric[metric] = (best_few_shot[1]['f1'], filename_dp.split('/')[-1], (best_few_shot[1]))
    print('{:<3} || {:.2f} || P {:.3f} | R {:.3f} | F1 {:.3f}'.format(metric, best_few_shot[0], best_few_shot[1]['precision'], best_few_shot[1]['recall'], best_few_shot[1]['f1']))
    best_few_shot_metrics[metric] = best_few_shot
print('='*80)
print('='*80)
for metric in metrics:
    print('{:<3} || P {:.3f} | R {:.3f} | F1 {:.3f}  ||| {}'.format(metric, 
        best_f1_metric[metric]['res']['precision'], best_f1_metric[metric]['res']['recall'], 
        best_f1_metric[metric]['res']['f1'], best_f1_metric[metric]['res']))
    print(best_f1_metric[metric]['thr_strategy'])


# %%

# =============================================================================
# Plot default curves vs. dist_params
# =============================================================================

import matplotlib.pyplot as plt

curve_results = curve_results_default

fig, ax = plt.subplots(1,2, dpi=300)
plt.tight_layout()

for num_plt, metric in enumerate(metrics):
    ax[num_plt].plot([ curve_results_default[metric][thr]['recall'] for thr in thresholds ], 
              [ curve_results_default[metric][thr]['precision'] for thr in thresholds ], label='one-shot (m = 1)')
    
    
    ax[num_plt].plot([ curve_results_dp[metric][thr]['recall'] for thr in thresholds ], 
              [ curve_results_dp[metric][thr]['precision'] for thr in thresholds ], label='few-shot (m = 3)', linestyle='--')
	
	
    best_f1 = max(curve_results_default[metric].items(), key=lambda x: x[1]['f1'])
    ax[num_plt].scatter([ best_f1[1]['recall'] ], [ best_f1[1]['precision'] ], s=8, 
						zorder=15, marker ='x', c='r', label='Best P/R trade-off')
    best_f1 = max(curve_results_dp[metric].items(), key=lambda x: x[1]['f1'])
    ax[num_plt].scatter([ best_f1[1]['recall'] ], [ best_f1[1]['precision'] ], s=8, 
                        zorder=15, marker ='x', c='r')
    	
	
    ax[num_plt].set(xlabel='recall', ylabel='precision')
	
    ax[num_plt].set_xlim(0,1)
    ax[num_plt].set_ylim(0,1)
    ax[num_plt].set_title(metric)
    ax[num_plt].grid(linestyle='dashed')
    ax[num_plt].set_aspect('equal', adjustable='box')


fig.legend(loc = 'lower left', ncol=6)

    # plt.savefig(path_model + 'PR_curve_'+metric+'.png')
plt.savefig(args.path_model + 'PR_curve_few_shot_TP{}_FP{}_FBA{}_B{}.png'.format(DIST_TO_GT_TP, DIST_TO_GT_FP, FRAMES_BEFOR_ANCHOR, batch), 
				bbox_inches = 'tight', pad_inches = 0
				)
plt.show()


# %%
# %%
# %%

# =============================================================================
# Calculate metric per class. One-shot, few-shot, dynamic
# =============================================================================


stats_filename_final_base = args.path_model + 'TP{}_FP{}_FBA{}_B{}_best-{}.json'.format(DIST_TO_GT_TP, DIST_TO_GT_FP, FRAMES_BEFOR_ANCHOR, batch, '{}')




# One-shot
if not os.path.isfile(stats_filename_final_base.format('oneshot')):
	metric_thr = { m: {'med': best_one_shot_metrics[m][0], 'good': 0.0, 'excel': 0.0} for m in metrics }
	stats_one_shot = em.get_therapies_metrics(model, actions_data_v2, video_skels_v2, metric_thr, 
	                                          model_params, skip_empty_targets, metrics=metrics, 
	                                          batch=batch,
	                                          dist_params={}, thr_strategy=None,
	                                          in_memory_callback=in_memory_callback, cache=cache, 
	                                          FRAMES_BEFOR_ANCHOR=FRAMES_BEFOR_ANCHOR, 
	                                          DIST_TO_GT_TP=DIST_TO_GT_TP, DIST_TO_GT_FP=DIST_TO_GT_FP)
	for metric in metrics:
		stats_one_shot[metric]['dist_params'] = {}
		stats_one_shot[metric]['metric_thr'] = metric_thr
		stats_one_shot[metric]['thr_strategy'] = None
	json.dump(stats_one_shot, open(stats_filename_final_base.format('oneshot'), 'w'))
else:
	stats_one_shot = json.load(open(stats_filename_final_base.format('oneshot'), 'r'))
print('* Calculated:', stats_filename_final_base.format('oneshot'))

# Few-shot
if not os.path.isfile(stats_filename_final_base.format('fewshot')):
	metric_thr = { m: {'med': best_few_shot_metrics[m][0], 'good': 0.0, 'excel': 0.0} for m in metrics }
	stats_few_shot = em.get_therapies_metrics(model, actions_data_v2, video_skels_v2, metric_thr, 
	                                          model_params, skip_empty_targets, metrics=metrics, 
	                                          batch=batch,
	                                          dist_params=dist_params, thr_strategy=None,
	                                          in_memory_callback=in_memory_callback, cache=cache, 
	                                          FRAMES_BEFOR_ANCHOR=FRAMES_BEFOR_ANCHOR, 
	                                          DIST_TO_GT_TP=DIST_TO_GT_TP, DIST_TO_GT_FP=DIST_TO_GT_FP)
	for metric in metrics:
		stats_few_shot[metric]['dist_params'] = dist_params
		stats_few_shot[metric]['metric_thr'] = metric_thr
		stats_few_shot[metric]['thr_strategy'] = None	
	json.dump(stats_few_shot, open(stats_filename_final_base.format('fewshot'), 'w'))
else:
	stats_few_shot = json.load(open(stats_filename_final_base.format('fewshot'), 'r'))
print('* Calculated:', stats_filename_final_base.format('fewshot'))


# Dynamic Few-shot
if not os.path.isfile(stats_filename_final_base.format('fewshotdyn')):
	metric_thr = { m: {'med': best_few_shot_metrics[m][0], 'good': 0.0, 'excel': 0.0} for m in metrics }
	stats_few_shot_dyn = {}
	for metric in metrics:
# 		metric_thr = { metric: {'med':None, 'good':None, 'excel':None} for metric in metrics }
		meric_thr = { metric: {'med': best_few_shot_metrics[metric][0], 'good': 0.0, 'excel': 0.0} }
		stats_m = em.get_therapies_metrics(model, actions_data_v2, video_skels_v2, metric_thr, 
	                                          model_params, skip_empty_targets, metrics=[metric], 
	                                          batch=batch,
	                                          dist_params=best_f1_metric[metric]['dist_params'], 
											  thr_strategy=best_f1_metric[metric]['thr_strategy'],
	                                          in_memory_callback=in_memory_callback, cache=cache, 
	                                          FRAMES_BEFOR_ANCHOR=FRAMES_BEFOR_ANCHOR, 
	                                          DIST_TO_GT_TP=DIST_TO_GT_TP, DIST_TO_GT_FP=DIST_TO_GT_FP)
		stats_few_shot_dyn[metric] = stats_m[metric]

		stats_few_shot_dyn[metric]['dist_params'] = best_f1_metric[metric]['dist_params']
		stats_few_shot_dyn[metric]['metric_thr'] = metric_thr
		stats_few_shot_dyn[metric]['thr_strategy'] = best_f1_metric[metric]['thr_strategy']
	json.dump(stats_few_shot_dyn, open(stats_filename_final_base.format('fewshotdyn'), 'w'))
else:
	stats_few_shot_dyn = json.load(open(stats_filename_final_base.format('fewshotdyn'), 'r'))
print('* Calculated:', stats_filename_final_base.format('fewshotdyn'))
	


#%%

import pandas as pd

# =============================================================================
# Per-class tables. Open the DataFrame "class_data_metric" to copy-paste
# =============================================================================

print('** Evaluation per_class results')

class_data_metric = {}
for metric in metrics:
	class_data = []
	class_data.append({  k:v['f1'] for k,v in stats_one_shot[metric]['per_class_stats'].items() })
	class_data.append({  k:v['f1'] for k,v in stats_few_shot[metric]['per_class_stats'].items() })
	class_data.append({  k:v['f1'] for k,v in stats_few_shot_dyn[metric]['per_class_stats'].items() })
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


# %%

# =============================================================================
# Ablation table
# =============================================================================


res = '\t' + '\t\t\t'.join(metrics) + '\n'
res += '\tThreshold\tPrecision\tRecall\tF1'*2 + '\n'
res += 'One-shot\t' + '\t'.join([ '{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(stats_one_shot[metric]['metric_thr'][metric]['med'], stats_one_shot[metric]['precision'], stats_one_shot[metric]['recall'], stats_one_shot[metric]['f1']) for metric in metrics ])
res += '\nFew-shot\t' + '\t'.join([ '{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(stats_few_shot[metric]['metric_thr'][metric]['med'], stats_few_shot[metric]['precision'], stats_few_shot[metric]['recall'], stats_few_shot[metric]['f1']) for metric in metrics ])
res += '\nDynamic Few-shot\t' + '\t'.join([ '{}\t{:.3f}\t{:.3f}\t{:.3f}'.format(stats_few_shot_dyn[metric]['thr_strategy'], stats_few_shot_dyn[metric]['precision'], stats_few_shot_dyn[metric]['recall'], stats_few_shot_dyn[metric]['f1']) for metric in metrics ])
res = res.replace('.', ',')

print(res)

print(); print()
for metric in metrics: print('{: <3}: {:.3f} | {:.3f} | {:.3f}'.format(metric, stats_one_shot[metric]['f1'], stats_few_shot[metric]['f1'], stats_few_shot_dyn[metric]['f1']))


