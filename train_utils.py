#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:20:14 2020

@author: asabater
"""

import os
import datetime
import shutil



def create_folder_if_not_exists(folder_path):
    if not os.path.isdir(folder_path): os.makedirs(folder_path)

def get_model_name(model_dir):
    folder_num = len(os.listdir(model_dir))
    return '/{}_model_{}/'.format(datetime.datetime.today().strftime('%m%d_%H%M'), folder_num)

def create_model_folder(path_results, folder_name):
    
    path_model = path_results + folder_name
    
    # Create logs and model main folder
    create_folder_if_not_exists(path_model)

    model_name = get_model_name(path_results + folder_name)
    path_model += model_name
    create_folder_if_not_exists(path_model)
    create_folder_if_not_exists(path_model + 'weights/')
    
    return path_model

def get_best_ckpt(model_folder):
    files = [ f for f in os.listdir(model_folder + '/weights/') if f.startswith('ep') and 'index' in f ]
    val_losses = [ [ float(v[8:]) for v in f[:-11].split('-') if v.startswith('val_loss') ][0] for f in files ]
    model_weights = model_folder + '/weights/' + files[val_losses.index(min(val_losses))]
    return model_weights

def get_model_path(path_results, dataset_name, model_num):
    return '{}{}/{}/'.format(path_results, dataset_name, 
                [ f for f in os.listdir(path_results + dataset_name) if f.endswith('model_{}'.format(model_num)) ][0])

