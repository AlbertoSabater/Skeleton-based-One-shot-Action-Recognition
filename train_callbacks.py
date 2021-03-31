#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:45:20 2020

@author: asabater
"""

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
from data_generator import load_scaler, get_pose_data_v2, get_body_skel, average_wrong_frame_skels
import numpy as np
import tensorflow as tf
import pickle
from tensorboard import summary as summary_lib
from joblib import Parallel, delayed
from tqdm import tqdm

def dummy_callback(writer):
    def aux(epoch, logs):
        summary_lib.scalar("foo", np.random.rand(), step=epoch)
        with writer.as_default():
            tf.summary.scalar('bar', np.random.random(), epoch)
            writer.flush()
    return aux



def load_pose(f, model_params):
    filename, label = f.split()
    
    pose_raw = np.load(filename, allow_pickle=True).item()  #['skel_body0']
    

    p = get_body_skel(pose_raw, validation=True)
    if model_params['average_wrong_skels']: p = average_wrong_frame_skels(p)
   
    pose_data = get_pose_data_v2(p, validation=True, **model_params)
    # print('target', pose_data.shape)
    pose_len = len(pose_data)
    
    return pose_len, label, pose_data

def get_lr_metric(optimizer):
	def lr(y_true, y_pred, sample_weights=None):
		return optimizer.lr
	return lr

