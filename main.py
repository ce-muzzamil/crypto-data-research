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


patience = 25
epochs = 100
mini_batch_size = 728
min_r2 = 0.9


def get_data(*vars, symbol='BTCUSDT', trans='buy'):
    outs = []
    for var in vars:
        ds = np.load(f'database/{trans}/{symbol}/{var}.npy', mmap_mode='r')
        outs.append(ds)
        del ds
    return outs


def normalized_test_data(symbol, mtype, trans='buy'):
    if trans == 'buy':
        testing = get_data('tsx1', 'tsx2', 'tsy1', 'tsy2',
                           'tsy3', 'tsy4', symbol=symbol, trans='buy')

        x_norm = np.load(f'database/{trans}/{symbol}/norm.npy')
        norm = [x_norm, 288.0, 288.0, x_norm[0], 288.0, x_norm[2]]

        testing = [testing[i]/norm[i] for i in range(len(testing))]

        if mtype == 'ch':
            out = [5]
        elif mtype == 'b':
            out = [2]
        elif mtype == 's':
            out = [4]
        elif mtype == 'p':
            out = [3]

    elif trans == 'sell':
        testing = get_data('tsx1', 'tsx2', 'tsy1', 'tsy2',
                           symbol=symbol, trans='sell')

        x_norm = np.load(f'database/{trans}/{symbol}/norm.npy')
        norm = [x_norm, 288.0, 288.0, x_norm[0]]

        testing = [testing[i]/norm[i] for i in range(len(testing))]

        if mtype == 's':
            out = [2]
        elif mtype == 'p':
            out = [3]

    return [testing[i] for i in [0, 1, *out]]


def normalized_data(symbol, mtype, trans='buy'):
    if trans == 'buy':
        training = get_data('trx1', 'trx2', 'try1', 'try2',
                            'try3', 'try4', symbol=symbol, trans='buy')
        validation = get_data('vlx1', 'vlx2', 'vly1', 'vly2',
                              'vly3', 'vly4', symbol=symbol, trans='buy')
        testing = get_data('tsx1', 'tsx2', 'tsy1', 'tsy2',
                           'tsy3', 'tsy4', symbol=symbol, trans='buy')

        x_norm = np.load(f'database/{trans}/{symbol}/norm.npy')
        norm = [x_norm, 288.0, 288.0, x_norm[0], 288.0, x_norm[2]]

        rindices = np.random.randint(
            0, training[0].shape[0], int(training[0].shape[0]/1))
        training = [(training[i]/norm[i])[rindices]
                    for i in range(len(training))]
        validation = [validation[i]/norm[i] for i in range(len(validation))]
        testing = [testing[i]/norm[i] for i in range(len(testing))]

        if mtype == 'ch':
            out = [5]
        elif mtype == 'b':
            out = [2]
        elif mtype == 's':
            out = [4]
        elif mtype == 'p':
            out = [3]

    elif trans == 'sell':
        training = get_data('trx1', 'trx2', 'try1', 'try2',
                            symbol=symbol, trans='sell')
        validation = get_data('vlx1', 'vlx2', 'vly1',
                              'vly2', symbol=symbol, trans='sell')
        testing = get_data('tsx1', 'tsx2', 'tsy1', 'tsy2',
                           symbol=symbol, trans='sell')

        x_norm = np.load(f'database/{trans}/{symbol}/norm.npy')
        norm = [x_norm, 288.0, 288.0, x_norm[0]]

        rindices = np.random.randint(
            0, training[0].shape[0], int(training[0].shape[0]/1))
        training = [(training[i]/norm[i])[rindices]
                    for i in range(len(training))]
        validation = [validation[i]/norm[i] for i in range(len(validation))]
        testing = [testing[i]/norm[i] for i in range(len(testing))]

        if mtype == 's':
            out = [2]
        elif mtype == 'p':
            out = [3]

    return [training[i] for i in [0, 1, *out]], [validation[i] for i in [0, 1, *out]], [testing[i] for i in [0, 1, *out]]


def wrapper(symbol, lr, mtype, trans='buy', earlystopping='v'):
    training, validation, _ = normalized_data(symbol, mtype, trans=trans)

    length = train_bp_model(symbol, mtype=mtype, lr=lr, inputs=training[:2], outputs=training[2:], validation_data=(
        validation[:2], validation[2:]), epochs=epochs, verbose=1, mini_batch_size=mini_batch_size, patience=patience, shuffle=False, allow_base=False, trans=trans, earlystopping=earlystopping)
    if length < epochs:
        return True
    return False


def test(symbol, mtype, trans='buy'):
    model = get_bp_model(symbol, 3e-4, mtype, allow_base=False, trans=trans)
    _, _, data = normalized_data(symbol, mtype, trans=trans)
    yp = model.predict(data[:2], batch_size=512, verbose=0)
    ya = data[2:]
    r2 = []
    if type(yp) == list:
        for i in range(len(yp)):
            slope, intercept, r_value, p_value, std_err = sci.stats.linregress(
                ya[i], yp[i].flatten())
            r2.append(r_value**2)
    else:
        slope, intercept, r_value, p_value, std_err = sci.stats.linregress(
            ya, yp.flatten())
        r2.append(r_value**2)
    return np.mean(r2)


def itter_train(trans='buy', earlystopping='v'):
    lrs = [1e-3, 1e-4]
    if trans == 'buy':
        mtypes = ['ch', 'b', 's', 'p']
    elif trans == 'sell':
        mtypes = ['s', 'p']

    for symbol in [r.split('\\')[-1] for r in glob.glob(f'database/{trans}/*') if '.' not in r]:
        print('for ', symbol)
        for mtype in mtypes:
            if not os.path.isdir(f'database/{trans}/{symbol}/{mtype}_history'):
                os.mkdir(f'database/{trans}/{symbol}/{mtype}_history')
                with open(f'database/{trans}/{symbol}/{mtype}_history/version_history.json', 'w') as file:
                    json.dump([], file)

            with open(f'database/{trans}/{symbol}/{mtype}_history/version_history.json') as file:
                h = json.load(file)

            i = [i for i in range(len(lrs)) if lrs[i] ==
                 h[-1]['lr']] if len(h) > 0 else []
            if len(i) == 0:
                i = 0
            else:
                i = i[-1]
                if h[-1]['epochs'] < epochs:
                    i += 1

            r2 = test(symbol, mtype, trans=trans)
            if r2 > min_r2:
                print(symbol, " ", mtype,
                      " doesnot need finetune for now: acc", r2)
                continue
            else:
                print("retraining as acc:", r2)

            while i < len(lrs):
                print(symbol, mtype)
                print('using lr = ', lrs[i])
                is_not_exhausted = wrapper(
                    symbol, lrs[i], mtype, trans=trans, earlystopping=earlystopping)
                i = i+1 if is_not_exhausted else i
                gc.collect()
                r2 = test(symbol, mtype, trans=trans)
                if r2 > min_r2:
                    print(symbol, " ", mtype,
                          " doesnot need further finetune for now: acc", r2)
                    break
                else:
                    print("retraining as acc:", r2)
                time.sleep(60)


if __name__ == '__main__':
    # itter_train(trans='buy', earlystopping='v')
    itter_train(trans='sell', earlystopping='v')
