import time
import numpy as np
from dataset import Dataset
from neural_network import buy_predictor
import json

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


tr_size = 2**15
vl_size = 2**12
epochs = 50
single_process_size = 2**8
max_procs = 8
max_training_counter = 10


def unique_identifier():
    r = np.random.randint(0, 10, size=(8)).tolist()
    r = [str(i) for i in r]
    r = ''.join(r)
    return r


def format_time(seconds):
    if seconds % 60 == seconds:
        return f'{np.round(seconds, 2)} secs'
    else:
        return f'{np.floor(seconds/60.)} mins & {np.round(seconds%60., 2)} secs'


model = buy_predictor()

if __name__ == '__main__':
    db = Dataset()
    training_counter = 0
    
    while training_counter<max_training_counter:
        st = time.time()
        print("Creating dataset...")
        inputs, outputs = db.get_data_set(
            batch_size=tr_size, max_procs=max_procs, single_process_size=single_process_size, buy=True, validation=False, test=False)
        vlx, vly = db.get_data_set(
            batch_size=vl_size, max_procs=max_procs, single_process_size=single_process_size, buy=True, validation=True, test=False)
        print(
            f"Dataset created with {inputs.shape[0]} training points and {vlx.shape[0]} validation points in {format_time(time.time()-st)}")

        st = time.time()
        hist = model.fit(inputs, outputs, epochs=epochs,
                        validation_data=(vlx, vly), verbose=1)

        print(f"Neural network trained in {format_time(time.time()-st)}")

        identifier = unique_identifier()
        model.save(f'model_history/{identifier}.h5')

        history = {"id": identifier, "time": time.time()}
        for key in hist.history.keys():
            history[key] = float(np.mean(hist.history[key]))

        with open('model_history/version_history.json', 'r') as file:
            version_history = json.load(file)

        version_history.append(history)
        with open('model_history/version_history.json', 'w') as file:
            json.dump(version_history, file)

        time.sleep(120.0)
