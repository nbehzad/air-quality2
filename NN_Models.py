import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, AveragePooling1D, MaxPooling1D, Flatten, Input, concatenate, LSTM, Dropout, \
    Conv1D, add


def create_lstm_base(input_shape=(240, 12), output_dim=24):
    input_layer = Input(shape=input_shape)
    lstm_layer1 = LSTM(32, dropout=0.4, return_sequences=True, activation='tanh')(input_layer)
    lstm_layer2 = LSTM(16, dropout=0.2, activation='relu')(lstm_layer1)
    hidden_layer = Dense(units=100, activation='relu')(lstm_layer2)
    output_layer = Dense(units=output_dim)(hidden_layer)
    # kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mae', metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    print(model.summary())
    return model


def create_lstm_with_pooling(input_shape=(240, 12), output_dim=24, pooling=1):
    input_layer = Input(shape=input_shape)
    if pooling > 1:
        pooling_layer = AveragePooling1D(pool_size=pooling, strides=pooling, padding="valid",
                                         data_format="channels_last")(input_layer)
        lstm_layer = LSTM(32, dropout=0.2, activation='tanh')(pooling_layer)
    else:
        lstm_layer = LSTM(32, dropout=0.2, activation='tanh')(input_layer)

    hidden_layer = Dense(units=100, activation='relu')(lstm_layer)
    output_layer = Dense(units=output_dim)(hidden_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mae', metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    print(model.summary())
    return model


def create_cnn_base(input_shape=(240, 12), output_dim=48):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=100, kernel_size=2, strides=1, padding="valid"))
    model.add(Dropout(rate=0.2))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    model.add(Flatten())
    model.add(Dense(units=output_dim))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mae', metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    print(model.summary())
    return model


def create_cnn(input_shape=(240, 12), output_dim=48, kernel=4, stride=1):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=100, kernel_size=kernel, strides=stride, padding="valid"))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=output_dim))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mae', metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    print(model.summary())
    return model


def create_cnn_lstm(input_shape=(240, 12), output_dim=48, kernel=4, stride=1):
    input_layer = Input(shape=input_shape)
    cnn_layer = Conv1D(100, kernel_size=kernel, padding='valid', strides=stride)(input_layer)
    cnn_layer = Dropout(0.2)(cnn_layer)
    lstm_layer = LSTM(32)(cnn_layer)
    output_layer = Dense(units=output_dim)(lstm_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mae', metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    print(model.summary())
    return model


def create_lstm_cnn(input_shape=(240, 12), output_dim=48, kernel=4, stride=1):
    input_layer = Input(shape=input_shape)
    lstm_layer = LSTM(32, return_sequences=True)(input_layer)
    cnn_layer = Conv1D(100, kernel_size=kernel, padding='valid', strides=stride)(lstm_layer)
    cnn_layer = Dropout(0.2)(cnn_layer)
    pooling_layer = Flatten()(cnn_layer)
    output_layer = Dense(units=output_dim)(pooling_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mae', metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    print(model.summary())
    return model


def create_lstm_cnn2(input_shape=(240, 12), output_dim=48, kernel=4, stride=1):
    input_layer = Input(shape=input_shape)
    lstm_layer = LSTM(32, return_sequences=False)(input_layer)
    lstm_dense = Dense(output_dim)(lstm_layer)

    cnn_layer = Conv1D(100, kernel_size=kernel, padding='valid', strides=stride)(input_layer)
    cnn_layer = Dropout(0.2)(cnn_layer)
    cnn_layer = Flatten()(cnn_layer)
    cnn_dense = Dense(output_dim)(cnn_layer)

    output_layer = Dense(output_dim)(add([lstm_dense, cnn_dense]))

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mae', metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    print(model.summary())
    return model


def create_mlp(input_shape=(240, 12), output_dim=48):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Flatten())
    model.add(Dense(units=100))
    model.add(Dense(units=output_dim))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mae', metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    print(model.summary())
    return model


def create_cnn_mlp_concat(input_shape=(240, 12), input_dim=1440, output_dim=48):
    mlp_input = Input(shape=input_dim)

    cnn_input = Input(shape=input_shape)
    cnn_layer = Conv1D(100, kernel_size=4, padding='valid', strides=4)(cnn_input)
    cnn_layer = Dropout(0.2)(cnn_layer)
    cnn_layer = Flatten()(cnn_layer)

    output_layer = Dense(output_dim)(concatenate([mlp_input, cnn_layer]))

    model = Model(inputs={'serial_input': cnn_input, 'vector_input': mlp_input}, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mae', metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    print(model.summary())
    return model


def create_cnn_mlp_attention(input_shape=(240, 12), input_dim=1440, output_dim=48):
    mlp_input = Input(shape=input_dim)
    mlp_output = Dense(units=100)(mlp_input)
    mlp_output = Dropout(.3)(mlp_output)

    cnn_input = Input(shape=input_shape)
    cnn_layer = Conv1D(100, kernel_size=4, padding='valid', strides=4)(cnn_input)
    cnn_layer = Dropout(0.3)(cnn_layer)
    cnn_layer = Flatten()(cnn_layer)
    cnn_output = Dense(units=100)(cnn_layer)

    output_layer = Dense(output_dim)(add([mlp_output, cnn_output]))

    model = Model(inputs={'serial_input': cnn_input, 'vector_input': mlp_input}, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mae', metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    print(model.summary())
    return model


def create_lstm_mlp_concat(input_shape=(240, 12), input_dim=1440, output_dim=48):
    mlp_input = Input(shape=input_dim)
    mlp_output = Dense(units=100)(mlp_input)
    mlp_output = Dropout(.3)(mlp_output)

    lstm_input = Input(shape=input_shape)
    lstm_output = LSTM(32, return_sequences=False)(lstm_input)

    output_layer = Dense(output_dim)(concatenate([lstm_output, mlp_output]))

    model = Model(inputs={'serial_input': lstm_input, 'vector_input': mlp_input}, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mae', metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    print(model.summary())
    return model


def create_lstm_mlp_atention(input_shape=(240, 12), input_dim=1440, output_dim=48):
    mlp_input = Input(shape=input_dim)
    mlp_output = Dense(units=100)(mlp_input)
    mlp_output = Dropout(.3)(mlp_output)

    lstm_input = Input(shape=input_shape)
    lstm_output = LSTM(32, return_sequences=False)(lstm_input)
    lstm_output = Dense(units=100)(lstm_output)
    lstm_output = Dropout(0.3)(lstm_output)

    output_layer = Dense(output_dim)(add([lstm_output, mlp_output]))

    model = Model(inputs={'serial_input': lstm_input, 'vector_input': mlp_input}, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mae', metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    print(model.summary())
    return model