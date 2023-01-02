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


def buy_predictor(input1_shape=(64, 10), input2_shape=(1,)):
    inputs1 = tf.keras.layers.Input(input1_shape)
    
    inputs2 = tf.keras.layers.Input(input2_shape)

    gcfe = tf.keras.layers.GRU(64, return_sequences=True)(inputs1)
    gcfe = tf.keras.layers.GRU(128, return_sequences=False)(gcfe)

    gcfe = tf.keras.layers.Dropout(0.5)(gcfe)
    extracted_features_ = tf.keras.layers.Dense(256, activation='relu')(gcfe)

    extracted_features = tf.keras.layers.Dense(
        128, activation='relu')(extracted_features_)
    extracted_features = tf.keras.layers.Concatenate(
        axis=1)([extracted_features, inputs2])
    extracted_features = tf.keras.layers.Dropout(0.5)(extracted_features)
    buytime = tf.keras.layers.Dense(1, name="y1")(extracted_features)

    extracted_features = tf.keras.layers.Dense(
        128, activation='relu')(extracted_features_)
    buy_price = tf.keras.layers.Concatenate(
        axis=1)([extracted_features, inputs2, buytime])
    buy_price = tf.keras.layers.Dropout(0.5)(buy_price)
    buy_price = tf.keras.layers.Dense(1, name="y2")(buy_price)
    

    extracted_features = tf.keras.layers.Dense(
        128, activation='relu')(extracted_features_)
    selling_features = tf.keras.layers.Concatenate(
        axis=1)([extracted_features, inputs2, buytime])
    selling_features = tf.keras.layers.Dropout(0.5)(selling_features)
    selltime = tf.keras.layers.Dense(1, name="y3")(selling_features)

    extracted_features = tf.keras.layers.Dense(
        128, activation='relu')(extracted_features_)
    change_features = tf.keras.layers.Concatenate(
        axis=1)([extracted_features, buytime, selltime])
    change_features = tf.keras.layers.Dropout(0.5)(change_features)
    aquired_change = tf.keras.layers.Dense(1, name="y4")(change_features)

    model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=[
                           buytime, buy_price, selltime, aquired_change])

    return model


def get_bp_model(symbol, lr, allow_base=True):
    with open(f'database/{symbol}/model_history/version_history.json', 'r') as file:
        version_history = json.load(file)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model = buy_predictor()

    if len(version_history) < 1:
        model.compile(optimizer=optimizer, loss={
            'y1': 'mae', 'y2': 'mae', 'y3': 'mae', 'y4': 'mae'})
        print("Generated New Model")
        return model
    else:
        sdata = version_history[-1]
        for data in version_history[:-1]:
            if data['loss'] < sdata['loss']:
                if not allow_base and data['id']=='Base':
                    pass
                else:
                    sdata = data

        model_name = sdata['id']
        print(f"{model_name} selected")
        model.load_weights(f'database/{symbol}/model_history/{model_name}.h5')

        model.compile(optimizer=optimizer, loss={
            'y1': 'mae', 'y2': 'mae', 'y3': 'mae', 'y4': 'mae'})

        print("Retrived Trained Model")
        return model


def show_model():
    print(get_bp_model(3e-4).summary())

    
def train_bp_model(symbol, inputs=None, outputs=None, validation_data=None, verbose=None, epochs=None, lr=3e-4, save=True, patience=20, mini_batch_size=32, allow_base=True, shuffle=False):
    st = time.time()
    model = get_bp_model(symbol, lr, allow_base)
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss' if save else 'loss', patience=patience, restore_best_weights=True)

    hist = model.fit(inputs, outputs, epochs=epochs,
                     validation_data=validation_data, verbose=verbose, callbacks=[callback], batch_size=mini_batch_size, shuffle=shuffle)

    print(f"Neural network trained in {format_time(time.time()-st)}")

    if save:
        identifier = unique_identifier()
        model.save_weights(f'database/{symbol}/model_history/{identifier}.h5')

        history = {"id": identifier, "time": time.time()}
        for key in hist.history.keys():
            history[key] = float(np.mean(hist.history[key]))

        history['lr'] = float(lr)
        history['epochs'] = len(hist.history['loss'])

        with open(f'database/{symbol}/model_history/version_history.json', 'r') as file:
            version_history = json.load(file)

        version_history.append(history)

        with open(f'database/{symbol}/model_history/version_history.json', 'w') as file:
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
