#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 19:32:22 2020

@author: asabater
"""

import os
from shutil import rmtree



def remove_path_weights(path_model, monitor='val_loss', min_monitor=True):
    if not os.path.isdir(path_model + '/weights'): 
        print('No weights folder:', path_model)
        return
    
    weights = [ w for w in os.listdir(path_model + '/weights') if w.endswith('.index') and monitor in w ]
    if len(weights) == 0: 
        print('Removing:', path_model)
        rmtree(path_model)
        return
    
    if min_monitor: best = min(weights, key=lambda w: [ float(s.replace(monitor, '')) for s in w.replace('.ckpt.index', '').split('-') if monitor in s ][0] )
    else: best = max(weights, key=lambda w: [ float(s.replace(monitor, '')) for s in w.replace('.ckpt.index', '').split('-') if monitor in s ][0] )
    best_epoch = [ s for s in best.split('-') if s.startswith('ep') ][0]
    remove = [ w for w in os.listdir(path_model + '/weights') if best_epoch not in w and 'ep' in w and monitor in w ]
    
    
    for r in remove: 
        if os.path.isfile(path_model + '/weights/' + r): os.remove(path_model + '/weights/' + r)



