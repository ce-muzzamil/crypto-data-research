import json
import time
import glob
import gc
import os
import numpy as np
from neural_network import train_bp_model


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


mini_batch_size = 512
epochs = 100

def get_data(*vars, symbol='BTCUSDT'):
    outs = []
    for var in vars:
        ds = np.load(f'database/{symbol}/{var}.npy', mmap_mode='r')
        outs.append(ds)
        del ds
    return outs

def normalized_data(symbol):
    training = get_data('trx1', 'trx2', 'try1', 'try2', 'try3', 'try4', symbol=symbol)
    validation = get_data('vlx1', 'vlx2', 'vly1', 'vly2', 'vly3', 'vly4', symbol=symbol)
    testing = get_data('tsx1', 'tsx2', 'tsy1', 'tsy2', 'tsy3', 'tsy4', symbol=symbol)

    x_norm = np.load(f'database/{symbol}/norm.npy')
    norm = [x_norm, 1.0, 288.0, x_norm[0], 288.0, x_norm[2]]

    training = [training[i]/norm[i] for i in range (len(training))]
    validation = [validation[i]/norm[i] for i in range (len(validation))]
    testing = [testing[i]/norm[i] for i in range (len(testing))]
    return training, validation, testing

def wrapper(symbol, lr):
    training, validation, _ = normalized_data(symbol)
    length = train_bp_model(symbol, inputs=training[:2], outputs=training[2:], validation_data=(
    validation[:2], validation[2:]), epochs=epochs, verbose=1, lr=lr, mini_batch_size=mini_batch_size, patience=12, shuffle=True, allow_base=False)
    if length<epochs:
        return True
    return False

lrs = [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]

if __name__=='__main__':
    for symbol in [r.split('\\')[-1] for r in glob.glob('database/*') if '.' not in r]:
        print('for ', symbol)
        if not os.path.isdir(f'database/{symbol}/model_history'):
            os.mkdir(f'database/{symbol}/model_history')
            with open(f'database/{symbol}/model_history/version_history.json', 'w') as file:
                json.dump([], file)

        with open(f'database/{symbol}/model_history/version_history.json') as file:
            h = json.load(file)

        i = [i for i in range(len(lrs)) if lrs[i]==h[-1]['lr']] if len(h)>0 else []
        if len(i)==0:
            i=0
        else:
            i=i[-1]
            if h[-1]['epochs']<epochs:
                i+=1

        while i < len(lrs):
            if lrs[i]==lrs[-1]:
                break
            print('using lr = ', lrs[i])
            
            is_not_exhausted = wrapper(symbol, lrs[i])
            i = i+1 if is_not_exhausted else i
            gc.collect()
            time.sleep(60)
        

    
