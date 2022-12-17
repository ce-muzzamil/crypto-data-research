import time
import numpy as np
from dataset import Dataset
from neural_network import train_bp_model, format_time
import json

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


tr_size = 2**15
vl_size = 2**12
epochs = 100
single_process_size = 2**10
max_procs = 10
max_training_counter = 10


if __name__ == '__main__':
    db = Dataset()
    training_counter = 0
    st = time.time()
    print("Creating validation dataset...")
    vlx, vly = db.get_data_set(
        batch_size=vl_size, max_procs=max_procs, single_process_size=single_process_size, buy=True, validation=True, test=False)
    print(
        f"Dataset created with {vlx.shape[0]} validation points in {format_time(time.time()-st)}")

    while training_counter < max_training_counter:
        st = time.time()
        print("Creating training dataset...")
        inputs, outputs = db.get_data_set(
            batch_size=tr_size, max_procs=max_procs, single_process_size=single_process_size, buy=True, validation=False, test=False)
        print(
            f"Dataset created with {inputs.shape[0]} training points in {format_time(time.time()-st)}")

        train_bp_model(inputs=inputs, outputs=outputs,
                       validation_data=(vlx, vly), epochs=epochs, verbose=1)

        time.sleep(60.0)
