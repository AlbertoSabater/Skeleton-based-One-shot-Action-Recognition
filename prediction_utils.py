#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 19:46:16 2020

@author: asabater
"""

import json
import os
import pickle


def get_weights_filename(path_model, loss_name, verbose=False, num_file=None):
    weights = sorted([ w for w in os.listdir(path_model + '/weights') if 'index' in w ])
    if verbose: print(weights)
    if loss_name is not None:
        weights = [ w for w in weights  if loss_name in w ][0]
    else:
        if num_file is not None:
            weights = weights[num_file]
        # weights = weights[0]
        elif 'mon' in weights[0]: 		# and False
            if verbose:  print('weights by monitor')
            weights = max(weights, key=lambda w: [ float(s[3:]) for s in w.replace('.ckpt.index', '').split('-') if s.startswith('mon') ][0])
        elif 'val_loss' not in weights[0]:
            if verbose: print('weights by last')
            weights = weights[-1]
        else:
            if verbose: print('weights by val_loss')
            losses = [ float(w.split('-')[2][8:15]) for w in weights ]
            weights = weights[losses.index(min(losses))]
    weights = weights[:-6]
    return path_model + '/weights/' + weights


def load_model(path_model, return_sequences=True, num_file=None, loss_name=None):
    model_params = json.load(open(path_model + '/model_params.json'))
    
    weights = get_weights_filename(path_model, loss_name, verbose=True)
    print(weights)
    
    if model_params.get('use_gru',False) == True and 'decoder_v2' not in path_model:
        model_params['use_gru'] = False
    

    from models.TCN_classifier import TCN_clf
    model = TCN_clf(prediction_mode=return_sequences, **model_params)
        
        
    print(' ** Model created')
    model.load_weights(weights).expect_partial()
    print(' ** Weights loaded:', weights)
    
    
    
    scale_data = model_params.get('scale_data', False) or model_params.get('use_scaler', False)
    if scale_data: 
        print(' * Loading data scaler')
        model_params['scaler'] = pickle.load(open(path_model + 'scaler.pckl', 'rb'))
    else: model_params['scaler'] = None
    
    
    for data_key in ['use_jcd_features', 'use_speeds', 'use_coords_raw', 
                     'use_coords', 'use_jcd_diff', 'use_bone_angles',
                     'tcn_batch_norm', 'use_bone_angles_cent']:
        if data_key not in model_params: model_params[data_key] = False
        
    if 'average_wrong_skels' not in model_params: model_params['average_wrong_skels'] = True
    
    return model, model_params