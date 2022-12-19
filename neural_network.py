import numpy as np
import tensorflow as tf
import json
import time


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


def clock_loss(y_true, y_pred):
    '''On a scale of 5 min interval this loss gives 0.0 as the minimum most loss while a gap of 5 mins in decission increases the loss to 17.403757, max loss is set by highest panelty'''
    maxtime = 18.0
    highest_panelty = 1000.0
    up_scaler = 64.0

    _y_true = maxtime-y_true
    _y_pred = maxtime-y_pred
    is_y_pred_out_of_bounds = tf.logical_or(_y_pred < 0.0, y_pred < 0.0)

    error = tf.where(y_true > maxtime/2.0, _y_true, y_true) - \
        tf.where(y_pred > maxtime/2.0, _y_pred, y_pred)
    error = tf.abs(error)

    # return tf.where(is_y_pred_out_of_bounds, highest_panelty, up_scaler*tf.math.exp(tf.math.pow(error, tf.math.pow(0.95, error))))
    return tf.where(is_y_pred_out_of_bounds, highest_panelty, up_scaler*tf.math.log(error+1.0))


def gc_block(inputs, gru_units=32, return_sequences=True):
    g1 = tf.keras.layers.GRU(gru_units, dropout=0.25,
                             return_sequences=return_sequences)
    g2 = tf.keras.layers.Bidirectional(g1)(inputs)
    return g2


def cn_block(inputs, conv_filters=64, is_last=False):
    c1 = tf.keras.layers.Conv1D(
        conv_filters, 3, padding='same')(inputs)
    bn1 = tf.keras.layers.BatchNormalization()(c1)
    ac1 = tf.keras.layers.ReLU()(bn1)
    m1 = tf.keras.layers.MaxPool1D(3)(ac1)
    m1_ = tf.keras.layers.Dropout(0.25)(m1)
    if is_last:
        return tf.keras.layers.Flatten()(m1_)
    return m1_


# def gc_feature_extractor(inputs):
#     c1 = cn_block(inputs, 32)
#     gc1 = gc_block(c1, 64)

#     c2 = cn_block(gc1, 256)
#     gc2 = gc_block(c2, 512, return_sequences=False)
#     return gc2

def gc_feature_extractor(inputs):
    # c1 = cn_block(inputs, 64)
    # c2 = cn_block(c1, 128,is_last=True)
    # c3 = cn_block(c2, 128)
    # c4 = cn_block(c3, 256)
    # c5 = cn_block(c4, 512, is_last=True)
    # gc1 = gc_block(c4, 512, return_sequences=False)
    gc1 = gc_block(inputs, 64, return_sequences=False)
    return gc1


def buy_predictor(input_shape=(288, 12)):
    inputs = tf.keras.layers.Input(input_shape)
    gcfe = gc_feature_extractor(inputs)

    extracted_features = tf.keras.layers.Dense(128, activation='relu')(gcfe)
    extracted_features_ = tf.keras.layers.Dropout(0.25)(extracted_features)
    buytime = tf.keras.layers.Dense(1, name="y1")(extracted_features_)

    selling_features = tf.keras.layers.Concatenate(
        axis=1)([extracted_features, buytime])
    selling_features_ = tf.keras.layers.Dropout(0.25)(selling_features)
    selltime = tf.keras.layers.Dense(1, name="y2")(selling_features_)

    change_features = tf.keras.layers.Concatenate(
        axis=1)([selling_features, selltime])
    change_features_ = tf.keras.layers.Dropout(0.25)(change_features)
    aquired_change = tf.keras.layers.Dense(1, name="y3")(change_features_)

    model = tf.keras.Model(inputs=inputs, outputs=[
                           buytime, selltime, aquired_change])

    return model


def get_bp_model(lr):
    with open('model_history/version_history.json', 'r') as file:
        version_history = json.load(file)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model = buy_predictor()

    if len(version_history) < 1:
        model.compile(optimizer=optimizer, loss={
            'y1': clock_loss, 'y2': clock_loss, 'y3': 'mae'})
        print("Generated New Model")
        return model
    else:
        sdata = version_history[-1]
        for data in version_history[:-1]:
            if data['loss'] < sdata['loss']:
                sdata = data

        model_name = sdata['id']
        print(f"{model_name} selected")
        model.load_weights(f'model_history/{model_name}.h5')

        model.compile(optimizer=optimizer, loss={
            'y1': clock_loss, 'y2': clock_loss, 'y3': 'mae'})

        print("Retrived Trained Model")
        return model


def show_model():
    print(get_bp_model(3e-4).summary())


def train_bp_model(inputs=None, outputs=None, validation_data=None, verbose=None, epochs=None, lr=3e-4, save=True, patience=20, mini_batch_size=32):
    st = time.time()
    model = get_bp_model(lr)
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss' if save else 'loss', patience=patience, restore_best_weights=False)

    hist = model.fit(inputs, outputs, epochs=epochs,
                     validation_data=validation_data, verbose=verbose, callbacks=[callback], batch_size=mini_batch_size)

    print(f"Neural network trained in {format_time(time.time()-st)}")

    if save:
        identifier = unique_identifier()
        model.save_weights(f'model_history/{identifier}.h5')

        history = {"id": identifier, "time": time.time()}
        for key in hist.history.keys():
            history[key] = float(np.mean(hist.history[key]))

        history['lr'] = float(lr)

        with open('model_history/version_history.json', 'r') as file:
            version_history = json.load(file)

        version_history.append(history)
        with open('model_history/version_history.json', 'w') as file:
            json.dump(version_history, file)
    else:
        return hist


def get_op_lr(lrx=None, lry=None, mini_batch_size=32):
    print("Selecting optimal learning rate")
    st = time.time()
    def lrs(epoch): return 1e-4 * 10**(epoch/20)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lrs)
    history = train_bp_model(inputs=lrx, outputs=lry, epochs=60, verbose=0,
                             save=False, patience=20, mini_batch_size=mini_batch_size)

    losses = history.history['loss']
    epochs = np.array([i for i in range(len(losses))])
    x = lrs(epochs)
    print(
        f"Optimal lr = {np.round(x[np.argmin(losses)], 4)} in {format_time(time.time()-st)}")
    return x[np.argmin(losses)]
