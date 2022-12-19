import time
import numpy as np
from dataset import Dataset
from neural_network import train_bp_model, format_time, get_op_lr, show_model
import json

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


tr_size = 2**13
vl_size = 2**10
lr_size = 2**10
mini_batch_size = 32
use_predefined_DS = True

epochs = 100
single_process_size = 2**10
max_procs = 10
max_training_counter = 100



def get_data(x, y, size):
    ds = np.load(f'database/{x}.npy', mmap_mode='r')
    indicies = np.random.choice(ds.shape[0], (size,))
    xd = ds[indicies]
    del ds

    ds = np.load(f'database/{y}.npy', mmap_mode='r')
    yd = ds.T[indicies]
    yd = yd.T
    del ds

    return xd, [yd[i] for i in range(yd.shape[0])]

# show_model()

if __name__ == '__main__':
    db = Dataset()
    with open('model_history/version_history.json', 'r') as file:
        training_counter = len(json.load(file))
    _lr = 0.001

    start=True
    if not use_predefined_DS:
        st = time.time()
        print("Creating validation dataset...")
        vlx, vly = db.get_data_set(
            batch_size=vl_size, max_procs=max_procs, single_process_size=single_process_size, buy=True, validation=True, test=False)
        print(
            f"Dataset created with {vlx.shape[0]} validation points in {format_time(time.time()-st)}")

        st = time.time()
        print("Creating validation dataset...")
        lrx, lyy = db.get_data_set(
            batch_size=lr_size, max_procs=max_procs, single_process_size=single_process_size, buy=True, validation=False, test=False)
        print(
            f"Dataset created with {lrx.shape[0]} Training points for estimating lr in {format_time(time.time()-st)}")
    else:
        print("Retriving saved data...")
        inputs, outputs = get_data('inputs', 'outputs', tr_size)
        vlx, vly = get_data('vlx', 'vly', vl_size)
        lrx, lry = get_data('lrx', 'lry', lr_size)
        print("Done")

    while training_counter < max_training_counter:
        print("Training started for: ", training_counter)
        if not use_predefined_DS:
            st = time.time()
            print("Creating training dataset...")
            inputs, outputs = db.get_data_set(
                batch_size=tr_size, max_procs=max_procs, single_process_size=single_process_size, buy=True, validation=False, test=False)
            print(
                f"Dataset created with {inputs.shape[0]} training points in {format_time(time.time()-st)}")
        else:
            if np.random.rand()>0.5 and not start:
                print("Retriving saved data...")
                inputs, outputs = get_data('inputs', 'outputs', tr_size)
                print("Done")
            else:
                start=False

        if np.random.rand() > 0.5:
            lr = get_op_lr(lrx=lrx, lry=lry, mini_batch_size=mini_batch_size)
        else:
            _lr =  0.05*0.9**(12*np.log(training_counter+1))
            lr = _lr

        print("Uisng lr= ", lr)
        train_bp_model(inputs=inputs, outputs=outputs,
                       validation_data=(vlx, vly), epochs=epochs, verbose=1, lr=lr, mini_batch_size=mini_batch_size, patience=25)

        training_counter+=1
