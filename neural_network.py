import json
import time
import numpy as np
import tensorflow as tf


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


def feature_extractor(inputs, finetune=False):
    gcfe = tf.keras.layers.GRU(
        64, return_sequences=True, trainable=(not finetune))(inputs)
    gcfe = tf.keras.layers.GRU(
        128, return_sequences=False, trainable=(not finetune))(gcfe)

    gcfe = tf.keras.layers.Dropout(0.5)(gcfe)
    extracted_features = tf.keras.layers.Dense(
        256, activation='relu', trainable=(not finetune))(gcfe)
    return extracted_features


def regression(features, name, helpers=[]):
    extracted_features = tf.keras.layers.Dense(128, activation='relu')(features)
    extracted_features = tf.keras.layers.Concatenate(axis=1)([extracted_features, *helpers])
    extracted_features = tf.keras.layers.Dense(128, activation='relu')(extracted_features)
    extracted_features = tf.keras.layers.Dropout(0.25)(extracted_features)
    out = tf.keras.layers.Dense(1, name=name)(extracted_features)
    return out


def ch_predictor(input1_shape=(64, 10), input2_shape=(1,), finetune=False):
    inputs1 = tf.keras.layers.Input(input1_shape)
    inputs2 = tf.keras.layers.Input(input2_shape)

    features = feature_extractor(inputs1)
    buytime = regression(features, 'y1', [inputs2])
    selltime = regression(features, 'y3', [inputs2, buytime])
    change = regression(features, 'y4', [buytime, selltime])

    model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=[change])
    return model

def bs_predictor(input1_shape=(64, 10), input2_shape=(1,), finetune=False):
    inputs1 = tf.keras.layers.Input(input1_shape)
    inputs2 = tf.keras.layers.Input(input2_shape)

    features = feature_extractor(inputs1)
    buytime = regression(features, 'y1', [inputs2])
    selltime = regression(features, 'y3', [inputs2, buytime])

    model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=[
                           buytime, selltime])
    return model

def p_predictor(input1_shape=(64, 10), input2_shape=(1,), finetune=False):
    inputs1 = tf.keras.layers.Input(input1_shape)
    inputs2 = tf.keras.layers.Input(input2_shape)

    features = feature_extractor(inputs1)
    buytime = regression(features, 'y1', [inputs2])
    buyprice = regression(features, 'y2', [inputs2, buytime])

    model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=[buyprice])
    return model


def get_bp_model(symbol, lr, mtype, allow_base=True, finetune=False):
    """mtype: ch, bs, p"""

    with open(f'database/buy/{symbol}/{mtype}_history/version_history.json', 'r') as file:
        version_history = json.load(file)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    if mtype=='ch':
        model = ch_predictor(finetune=finetune)
    elif mtype=='bs':
        model = bs_predictor(finetune=finetune)
    elif mtype=='p':
        model = p_predictor(finetune=finetune)
    else:
        raise Exception("Invalid mtype")

    if len(version_history) < 1:
        model.compile(optimizer=optimizer, loss='mae')
        print("Generated New Model")
        return model
    else:
        sdata = version_history[-1]
        for data in version_history[:-1]:
            if data['loss'] < sdata['loss']:
                if not allow_base and data['id'] == 'Base':
                    pass
                else:
                    sdata = data

        model_name = sdata['id']
        print(f"{model_name} selected")
        model.load_weights(f'database/buy/{symbol}/{mtype}_history/{model_name}.h5')

        model.compile(optimizer=optimizer, loss='mae')

        print("Retrived Trained Model")
        return model


def train_bp_model(symbol, mtype='ch', inputs=None, outputs=None, validation_data=None, verbose=None, epochs=None, lr=3e-4, save=True, patience=20, mini_batch_size=32, allow_base=True, shuffle=False, finetune=False):
    st = time.time()
    model = get_bp_model(symbol, lr, mtype, allow_base, finetune=finetune)
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss' if save else 'loss', patience=patience, restore_best_weights=True)

    hist = model.fit(inputs, outputs, epochs=epochs,
                     validation_data=validation_data, verbose=verbose, callbacks=[callback], batch_size=mini_batch_size, shuffle=shuffle)

    print(f"Neural network trained in {format_time(time.time()-st)}")

    if save:
        identifier = unique_identifier()
        model.save_weights(f'database/buy/{symbol}/{mtype}_history/{identifier}.h5')

        history = {"id": identifier, "time": time.time()}
        for key in hist.history.keys():
            history[key] = float(np.mean(hist.history[key]))

        history['lr'] = float(lr)
        history['epochs'] = len(hist.history['loss'])

        with open(f'database/buy/{symbol}/{mtype}_history/version_history.json', 'r') as file:
            version_history = json.load(file)

        version_history.append(history)

        with open(f'database/buy/{symbol}/{mtype}_history/version_history.json', 'w') as file:
            json.dump(version_history, file)
        return len(hist.history['loss'])
    else:
        return hist


# def get_op_lr(lrx=None, lry=None, mini_batch_size=32):
#     print("Selecting optimal learning rate")
#     st = time.time()
#     def lrs(epoch): return 1e-4 * 10**(epoch/20)
#     lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lrs)
#     history = train_bp_model(inputs=lrx, outputs=lry, epochs=60, verbose=0,
#                              save=False, patience=20, mini_batch_size=mini_batch_size)

#     losses = history.history['loss']
#     epochs = np.array([i for i in range(len(losses))])
#     x = lrs(epochs)
#     print(
#         f"Optimal lr = {np.round(x[np.argmin(losses)], 4)} in {format_time(time.time()-st)}")
#     return x[np.argmin(losses)]
