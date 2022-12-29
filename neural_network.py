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


# def clock_loss(y_true, y_pred):
#     '''On a scale of 5 min interval this loss gives 0.0 as the minimum most loss while a gap of 5 mins in decission increases the loss to 17.403757, max loss is set by highest panelty'''
#     maxtime = 63.0
#     highest_panelty = 1000.0
#     up_scaler = 64.0

#     _y_true = maxtime-y_true
#     _y_pred = maxtime-y_pred
#     is_y_pred_out_of_bounds = tf.logical_or(_y_pred < 0.0, y_pred < 0.0)

#     error = tf.where(y_true > maxtime/2.0, _y_true, y_true) - \
#         tf.where(y_pred > maxtime/2.0, _y_pred, y_pred)
#     error = tf.abs(error)

#     # highest_panelty = y_pred*0.0 + highest_panelty

#     # return tf.where(is_y_pred_out_of_bounds, highest_panelty, up_scaler*tf.math.exp(tf.math.pow(error, tf.math.pow(0.95, error))))
#     # return tf.where(is_y_pred_out_of_bounds, highest_panelty, up_scaler*tf.math.log(error+1.0))
#     return tf.where(is_y_pred_out_of_bounds, highest_panelty, tf.math.pow(error, 2.5))


# def gru_res_block(inputs, skip=3, gru_entry_units=32, gru_exit_units=64, return_sequences=False):

#     for i in range(skip):
#         gr = tf.keras.layers.GRU(gru_entry_units if i < (skip-1) else gru_exit_units,
#                                  return_sequences=True if i < (skip-1) else return_sequences)(gr if i > 0 else inputs)

#     sk = tf.keras.layers.GRU(gru_exit_units,
#                              return_sequences=return_sequences)(inputs)

#     return tf.keras.layers.Add()([gr, sk])


# def cn_block(inputs, conv_filters=32, kernel=3, is_last=False):
#     c1 = tf.keras.layers.Conv1D(
#         conv_filters, kernel, padding='same')(inputs)
#     bn1 = tf.keras.layers.BatchNormalization()(c1)
#     ac1 = tf.keras.layers.ReLU()(bn1)
#     if is_last:
#         return tf.keras.layers.GlobalAveragePooling1D()(ac1)
#     return ac1


# def cnn_res_block(inputs, skip=3, conv_entry_filters=32, conv_exit_filters=64, is_last=False):

#     kernel = [3 for _ in range(skip)]
#     kernel[0] = 1
#     kernel[-1] = 1
#     for i in range(skip):
#         cr = cn_block(cr if i > 0 else inputs, conv_entry_filters if i < (skip-1) else conv_exit_filters,
#                       kernel=kernel[i], is_last=False if i < (skip-1) else is_last)

#     sk = cn_block(inputs, conv_exit_filters, kernel=1, is_last=is_last)

#     return tf.keras.layers.Add()([cr, sk])


# def gc_feature_extractor(inputs):
#     cb1 = cn_block(inputs, conv_filters=32, kernel=7, is_last=False)

#     crb1 = cnn_res_block(cb1, conv_entry_filters=32, skip=3,
#                          conv_exit_filters=64, is_last=False)

#     cb2_gbp = cn_block(crb1, conv_filters=128, kernel=3, is_last=True)

#     crb2_gbp = cnn_res_block(crb1, conv_entry_filters=64, skip=3,
#                              conv_exit_filters=128, is_last=True)

#     crc1 = tf.keras.layers.Add()([crb2_gbp, cb2_gbp])

#     grb1 = gru_res_block(inputs, skip=3, gru_entry_units=32,
#                         gru_exit_units=64, return_sequences=True)

#     g1 = tf.keras.layers.GRU(128, return_sequences=False)(grb1)

#     grb2_flat = gru_res_block(inputs, skip=3, gru_entry_units=64,
#                         gru_exit_units=128, return_sequences=False)

#     grg1 =tf.keras.layers.Add()([grb2_flat, g1])

#     return tf.keras.layers.Concatenate(axis=1)([crc1, grg1])


def buy_predictor(input1_shape=(64, 10), input2_shape=(1,)):
    inputs1 = tf.keras.layers.Input(input1_shape)
    
    inputs2 = tf.keras.layers.Input(input2_shape)

    # gcfe = gc_feature_extractor(inputs)

    gcfe = tf.keras.layers.GRU(64, return_sequences=True)(inputs1)
    gcfe = tf.keras.layers.GRU(128, return_sequences=False)(gcfe)

    extracted_features_ = tf.keras.layers.Dense(256, activation='relu')(gcfe)
    # extracted_features_ = tf.keras.layers.Dropout(0.1)(extracted_features)

    extracted_features = tf.keras.layers.Dense(
        128, activation='relu')(extracted_features_)
    extracted_features = tf.keras.layers.Concatenate(
        axis=1)([extracted_features, inputs2])
    extracted_features = tf.keras.layers.Dropout(0.25)(extracted_features)
    buytime = tf.keras.layers.Dense(1, name="y1")(extracted_features)

    extracted_features = tf.keras.layers.Dense(
        128, activation='relu')(extracted_features_)
    selling_features = tf.keras.layers.Concatenate(
        axis=1)([extracted_features, inputs2, buytime])
    selling_features = tf.keras.layers.Dropout(0.25)(selling_features)
    selltime = tf.keras.layers.Dense(1, name="y2")(selling_features)

    extracted_features = tf.keras.layers.Dense(
        128, activation='relu')(extracted_features_)
    change_features = tf.keras.layers.Concatenate(
        axis=1)([extracted_features, buytime, selltime])
    change_features = tf.keras.layers.Dropout(0.25)(change_features)
    aquired_change = tf.keras.layers.Dense(1, name="y3")(change_features)

    model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=[
                           buytime, selltime, aquired_change])

    return model


def get_bp_model(lr):
    with open('model_history/version_history.json', 'r') as file:
        version_history = json.load(file)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model = buy_predictor()

    if len(version_history) < 1:
        model.compile(optimizer=optimizer, loss={
            'y1': 'mae', 'y2': 'mae', 'y3': 'mae'})
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
            'y1': 'mae', 'y2': 'mae', 'y3': 'mae'})

        print("Retrived Trained Model")
        return model


def show_model():
    print(get_bp_model(3e-4).summary())


def train_bp_model(inputs=None, outputs=None, validation_data=None, verbose=None, epochs=None, lr=3e-4, save=True, patience=20, mini_batch_size=32):
    st = time.time()
    model = get_bp_model(lr)
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss' if save else 'loss', patience=patience, restore_best_weights=True)

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
