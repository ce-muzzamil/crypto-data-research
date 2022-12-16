import tensorflow as tf


def clock_loss(y_true, y_pred):
    '''On a scale of 5 min interval this loss gives 0.0 as the minimum most loss while a gap of 5 mins in decission increases the loss to 17.403757, max loss is set by highest panelty'''
    maxtime = 18.0
    highest_panelty = 1000.0
    log_up_scaler = 64.0

    _y_true = maxtime-y_true
    _y_pred = maxtime-y_pred
    is_y_pred_out_of_bounds = tf.logical_or(_y_pred < 0.0, y_pred < 0.0)

    error = tf.where(y_true > maxtime/2.0, _y_true, y_true) - \
        tf.where(y_pred > maxtime/2.0, _y_pred, y_pred)
    error = tf.abs(error)

    return tf.where(is_y_pred_out_of_bounds, highest_panelty, log_up_scaler*tf.math.log(error+1.0))


def gc_block(inputs, gru_units=32, conv_filters=64, is_last=False):
    g1 = tf.keras.layers.GRU(gru_units, dropout=0.25, return_sequences=True)
    g2 = tf.keras.layers.Bidirectional(g1)(inputs)

    c1 = tf.keras.layers.Conv1D(
        conv_filters, 3, padding='same', activation='relu')(g2)
    c1_ = tf.keras.layers.BatchNormalization()(c1)
    m1 = tf.keras.layers.MaxPool1D(3)(c1)
    m1_ = tf.keras.layers.Dropout(0.25)(m1)
    if is_last:
        return tf.keras.layers.Flatten()(m1_)
    return m1_


def cn_block(inputs, conv_filters=64, is_last=False):
    c1 = tf.keras.layers.Conv1D(
        conv_filters, 3, padding='same', activation='relu')(inputs)
    c1_ = tf.keras.layers.BatchNormalization()(c1)
    m1 = tf.keras.layers.MaxPool1D(3)(c1)
    m1_ = tf.keras.layers.Dropout(0.25)(m1)
    if is_last:
        return tf.keras.layers.Flatten()(m1_)
    return m1_


def gc_feature_extractor(inputs):
    gc1 = gc_block(inputs, 32, 64)
    gc2 = gc_block(gc1, 64, 128)
    gc3 = gc_block(gc1, 128, 256)

    c4 = cn_block(gc3, 256)
    c5 = cn_block(c4, 512, is_last=True)
    return c5


def buy_predictor(input_shape=(288, 12)):
    inputs = tf.keras.layers.Input(input_shape)
    gcfe = gc_feature_extractor(inputs)

    extracted_features = tf.keras.layers.Dense(512, activation='relu')(gcfe)
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

    initial_learning_rate = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=5,
    decay_rate=0.5,
    staircase=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss={
                  'y1': clock_loss, 'y2': clock_loss, 'y3': 'mae'})
    return model


def save_model(path, model):
    model.save(path)