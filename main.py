import json
import time
import glob
import gc
import os
import numpy as np
from matplotlib import pyplot as plt
import scipy as sci
from neural_network import train_bp_model, get_bp_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


patience=12
epochs=100
mini_batch_size=512

def get_data(*vars, symbol='BTCUSDT'):
    outs = []
    for var in vars:
        ds = np.load(f'database/buy/{symbol}/{var}.npy', mmap_mode='r')
        outs.append(ds)
        del ds
    return outs

def normalized_data(symbol, mtype):
    training = get_data('trx1', 'trx2', 'try1', 'try2', 'try3', 'try4', symbol=symbol)
    validation = get_data('vlx1', 'vlx2', 'vly1', 'vly2', 'vly3', 'vly4', symbol=symbol)
    testing = get_data('tsx1', 'tsx2', 'tsy1', 'tsy2', 'tsy3', 'tsy4', symbol=symbol)

    x_norm = np.load(f'database/buy/{symbol}/norm.npy')
    norm = [x_norm, 288.0, 288.0, x_norm[0], 288.0, x_norm[2]]

    rindices = np.random.randint(0, training[0].shape[0], int(training[0].shape[0]/1))
    training = [(training[i]/norm[i])[rindices] for i in range (len(training))]
    validation = [validation[i]/norm[i] for i in range (len(validation))]
    testing = [testing[i]/norm[i] for i in range (len(testing))]

    if mtype=='ch':
        out = [5]
    elif mtype=='bs':
        out = [2,4]
    elif mtype=='p':
        out = [3]
    return [training[i] for i in [0,1,*out]], [validation[i] for i in [0,1,*out]], [testing[i] for i in [0,1,*out]]

def wrapper(symbol, lr, mtype):
    training, validation, _ = normalized_data(symbol, mtype)
    
    length = train_bp_model(symbol,mtype=mtype,lr=lr, inputs=training[:2], outputs=training[2:], validation_data=(
    validation[:2], validation[2:]), epochs=epochs, verbose=1, mini_batch_size=mini_batch_size, patience=patience, shuffle=False, allow_base=False)
    if length<epochs:
        return True
    return False

def test(symbol, mtype):
    model = get_bp_model(symbol, 3e-4, mtype, allow_base=False)
    _,_,data = normalized_data(symbol, mtype)
    yp = model.predict(data[:2], batch_size=512, verbose=0)
    ya = data[2:]
    r2 = []
    if type(yp)==list:
        for i in range(len(yp)):
            slope, intercept, r_value, p_value, std_err = sci.stats.linregress(ya[i], yp[i].flatten())
            r2.append(r_value**2)
    else:
        slope, intercept, r_value, p_value, std_err = sci.stats.linregress(ya, yp.flatten())
        r2.append(r_value**2)
    return np.mean(r2)

lrs = [3e-3, 3e-4, 3e-6, 1e-2, 1e-3, 5e-4, 1e-4]
mtypes = ['ch', 'bs', 'p']

if __name__=='__main__':
    for symbol in [r.split('\\')[-1] for r in glob.glob('database/buy/*') if '.' not in r]:
        if symbol in ["ADAUSDT", "BNBUSDT", "BTCUSDT"]:
            continue
        print('for ', symbol)
        for mtype in mtypes:
            if not os.path.isdir(f'database/buy/{symbol}/{mtype}_history'):
                os.mkdir(f'database/buy/{symbol}/{mtype}_history')
                with open(f'database/buy/{symbol}/{mtype}_history/version_history.json', 'w') as file:
                    json.dump([], file)

            with open(f'database/buy/{symbol}/{mtype}_history/version_history.json') as file:
                h = json.load(file)

            i = [i for i in range(len(lrs)) if lrs[i]==h[-1]['lr']] if len(h)>0 else []
            if len(i)==0:
                i=0
            else:
                i=i[-1]
                if h[-1]['epochs']<epochs:
                    i+=1

            r2 = test(symbol, mtype)
            if r2>0.85:
                print(symbol, " ", mtype, " doesnot need finetune for now: acc", r2)
                continue
            else:
                print("retraining as acc:", r2)

            while i < len(lrs):
                print(symbol, mtype)
                print('using lr = ', lrs[i])
                is_not_exhausted = wrapper(symbol, lrs[i], mtype)
                i = i+1 if is_not_exhausted else i
                gc.collect()
                r2 = test(symbol, mtype)
                if r2>0.85:
                    print(symbol, " ", mtype, " doesnot need further finetune for now: acc", r2)
                    break
                else:
                    print("retraining as acc:", r2)
                time.sleep(60)
            
    
