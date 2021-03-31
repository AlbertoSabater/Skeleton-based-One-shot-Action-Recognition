#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:10:29 2020

@author: asabater
"""

from scipy.special import comb
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LambdaCallback
import json

from data_generator import triplet_data_generator, get_scaler_filename, get_num_feats
from train_callbacks import get_lr_metric  # eval_one_shot_callback, eval_one_shot_therapies_callback, 
import train_utils
from shutil import copyfile

from models.TCN_classifier import TCN_clf
# tf.config.experimental_run_functions_eagerly(True)

from dataset_scripts.ntu120_utils.triplet_ntu_callback import eval_ntu_one_shot_triplets_callback
from dataset_scripts.therapies.triplet_therapies_callback import eval_therapies_triplet_callback

from remove_suboptimal_weights import remove_path_weights



np.random.seed(123)
tf.random.set_seed(123)



def main(model_params):
    train_verbose = 2

    model_params.update({
                'path_model': train_utils.create_model_folder(model_params['path_results'], model_params['model_name']),
                'num_jcd_feats': int(comb(model_params['joints_num'],2)), 
                'num_feats': int(comb(model_params['joints_num'],2)) + model_params['joints_dim']*model_params['joints_num'],
            })
    model_params['num_feats'] = get_num_feats(**model_params)
    json.dump(model_params, open(model_params['path_model']+'model_params.json', 'w'))
    print(model_params)
    
    with open(model_params['train_annotations'], 'r') as f: num_train_files = len(f.read().splitlines())
    if model_params['val_annotations']  == '': num_val_files = 0
    else:
        with open(model_params['val_annotations'], 'r') as f: num_val_files = len(f.read().splitlines())
    print(num_train_files, num_val_files)
    
    if model_params['scale_data']:
        scaler_filename = get_scaler_filename(**model_params)
        copyfile(scaler_filename, model_params['path_model'] + '/scaler.pckl')    
    
    
    model = TCN_clf(**model_params)
    
    
    # Build model
    model.build((None, None, model_params['num_feats']))
    
    # Initialize inputs and outputs
    dummy_inpt = (np.random.rand(model_params['batch_size'], max(abs(model_params['max_seq_len']), 123), model_params['num_feats']))
    print(' * dummy_shape:', dummy_inpt.shape)
    dummy_pred = model(dummy_inpt);
    print(' * dummy_pred shape', [ p.shape for p in dummy_pred ])
    dummy_pred = model.predict(dummy_inpt);
    print(' * dummy_pred predict shape', [ p.shape for p in dummy_pred ])
    dummy_emb = model.get_embedding(dummy_inpt);
    print(' * dummy_emb shape', dummy_emb.shape)
    
    
    optimizer = Adam(model_params['init_lr'], clipnorm=1.)
    losses, metrics, loss_weights, sample_weights_mode = {}, {}, {}, {}
    losses['output_1'] = tf.keras.losses.CategoricalCrossentropy()
    loss_weights['output_1'] = 0.4
    # loss_weights = None
    # loss_weights = [ 1.0 ]
    metrics = [ 'accuracy', get_lr_metric(optimizer) ]
        

    print(' * losses:', losses)
    print(' * loss_weights:', loss_weights)
    if sample_weights_mode == {}: sample_weights_mode = None
    print(' * sample_weights_mode:', sample_weights_mode)

    model.summary(100)

    
    monitor = model_params.get('monitor', 'val_loss')
    print(' * Monitor:', monitor)
    model_chkpt_path = 'ep{epoch:03d}-loss{loss:.5f}-' + monitor + '{' + monitor + ':.5f}.ckpt'
    callbacks = [ 
                    TensorBoard(log_dir = model_params['path_model'], profile_batch=0),
                    ModelCheckpoint(model_params['path_model'] + 'weights/' + model_chkpt_path,
                                             monitor=monitor, save_weights_only=True, 
                                             save_best_only=True, save_freq='epoch'),
                    ReduceLROnPlateau(monitor=monitor, min_delta=0.001, factor=0.1, patience=3, verbose=1, min_lr=1e-7),
                    EarlyStopping(monitor=monitor, min_delta=0.001, patience=6, verbose=1),
                ]



    file_writer = tf.summary.create_file_writer(model_params['path_model'] + "/metrics")
    file_writer.set_as_default()
        

    if model_params['eval_ntu']: 
        callbacks = [LambdaCallback(_supports_tf_logs = True, 
                                    on_epoch_end=eval_ntu_one_shot_triplets_callback(model, model_params.copy(), file_writer))] + callbacks
    if model_params['eval_therapies']: 
        callbacks = [LambdaCallback(_supports_tf_logs = True, 
                                    on_epoch_end=eval_therapies_triplet_callback(model, model_params.copy(), file_writer, 'full'))] + callbacks
        callbacks = [LambdaCallback(_supports_tf_logs = True, 
                                    on_epoch_end=eval_therapies_triplet_callback(model, model_params.copy(), file_writer, 'sample'))] + callbacks


    print(callbacks)
    
    
    print(' * metrics:', metrics)
    print(' * sample_weights_mode:', sample_weights_mode)
    

    model.compile(optimizer=optimizer,
                  loss = losses,
                  metrics = metrics,
                  loss_weights = loss_weights,
                  sample_weight_mode=sample_weights_mode
                  )


    # Save model
    model.save(model_params['path_model'] + 'model')
    

    
    train_gen = triplet_data_generator(pose_annotations_file=model_params['train_annotations'], 
                           validation=False, 
                           in_memory_generator=model_params['in_memory_generator_train'],
                           **model_params)
    if model_params['val_annotations'] == '': val_gen = None
    else:
        val_gen = triplet_data_generator(pose_annotations_file=model_params['val_annotations'], 
                           validation=True, 
                           in_memory_generator=model_params['in_memory_generator_val'],
                           **model_params)

    print(train_gen, val_gen)
    
    model.fit(
            train_gen,
            validation_data = val_gen,
            steps_per_epoch = num_train_files//model_params['batch_size'],
            validation_steps = None if num_val_files == 0 else num_val_files//model_params['batch_size'],
            epochs = 300, 
            # steps_per_epoch = 10,         # num_val_files//model_params['batch_size'],
            # validation_steps = 10,
            # epochs = 50, 
            verbose = train_verbose,
            callbacks = callbacks,
        )

    del train_gen; del val_gen
    del callbacks


    
    model.summary(100)
    
    
    # Remove suboptimal weights
    remove_path_weights(model_params['path_model'], model_params['monitor'], model_params['min_monitor'])
    



if __name__ == "__main__":
    
    model_params = {
        "path_results": "./pretrained_models/",

# # NTU-120 Data sets to optimize the therapy data
# "train_annotations": "./ntu_annotations/one_shot_aux_set_train_full8.txt",
# "val_annotations": "./ntu_annotations/one_shot_aux_set_val_full8.txt",
# "eval_therapies": True,       ### Therapy data needed for its evaluation
# "eval_therapies_triplets_dataset": "./therapies_annotations/triplets/triplets_dataset.pckl",
# "eval_therapies_triplets_bgnd_dataset": "./therapies_annotations/triplets/triplets_ther_pat_bgnd_dataset.pckl",
# "eval_therapies_video_skels": "./therapies_annotations/video_skels.pckl",
# "h_flip": True,
# "skip_frames": [2, 3],

# NTU-120 Data sets to optimize the NTU one-shot benchmark
"train_annotations": "./ntu_annotations/one_shot_aux_set.txt",
"val_annotations": "",
"eval_therapies": False,
"h_flip": False,
"monitor": "ntu_one_shot_acc_euc",
"skip_frames": [2],


"in_memory_generator_train": False,
"in_memory_generator_val": True,
"in_memory_callback": True,

"eval_ntu": True,
"eval_ntu_one_shot_eval_anchors_file": "./ntu_annotations/one_shot_eval_anchors.txt",
"eval_ntu_one_shot_eval_set_file": "./ntu_annotations/one_shot_eval_set.txt",


"joints_num": 25,
"joints_dim": 3,
"init_lr": 0.0001,
"max_seq_len": -32,

# Set True to use a fitted data scaler. The one from the pre-trained models can also be used
"scale_data": False,       
"lstm_recurrent_dropout": 0.0,
"lstm_dropout": 0.2,
"num_layers": 2,
"num_neurons": 256,
"batch_size": 64,
"masking": True,
"center_skels": True,
"scale_by_torso": True,
"temporal_scale": [0.8, 1.2],
"classification": True, "triplet": False, "decoder": False, "reverse_decoder": False,
"num_classes": 120,
"clf_neurons": 0,

"model_name": "train_TCN",
"conv_params": [256, 4, 2, True, "causal", [4]],
"is_tcn": False,
"use_jcd_features": True,
"use_speeds": False,
"use_coords_raw": False,
"use_coords": True,
"use_jcd_diff": False,
"use_bone_angles": True,
"use_bone_angles_cent": False,
"average_wrong_skels": True,
        }
    
    main(model_params)
    