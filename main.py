from Dataset import Dataset
from NN_Models import *
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
import os
import numpy as np

STEP = 1
BATCH_SIZE = 512
EPOCHS = 20
BUFFER_SIZE = 30000


def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def plot_train_history(history, file_name):
    # summarize history for accuracy
    plt.figure(figsize=(8, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('train history')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.draw()
    plt.savefig(file_name)
    plt.close()


def create_time_steps(length):
    return list(range(-length, 0))


def multi_step_plot(history, true_future, prediction, index=0):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, -1]), label='History')
    plt.plot(np.arange(num_out) / STEP, np.array(true_future), 'g-',
             label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out) / STEP, np.array(prediction), 'r-',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.savefig('predict-plot-' + str(index) + '.png')
    plt.close()


def train2input(x_train, y_train, x_test, y_test, model_dir='models/LSTM'):
    train_data = tf.data.Dataset.from_tensor_slices(({"serial_input": x_train[0], "vector_input": x_train[1]}, y_train))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data = tf.data.Dataset.from_tensor_slices(({"serial_input": x_test[0], "vector_input": x_test[1]}, y_test))
    val_data = val_data.batch(BATCH_SIZE).repeat()

    create_directory(model_dir)
    checkpoint = ModelCheckpoint(model_dir + '/weights-{epoch:03d}-{val_loss:.4f}-.hdf5', monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='auto')

    model = create_cnn_mlp_attention(input_shape=x_train[0].shape[-2:], input_dim=x_train[1].shape[-1],
                                   output_dim=y_train.shape[-1])
    h = model.fit(train_data, validation_data=val_data, steps_per_epoch=int(x_train[0].shape[0]/BATCH_SIZE) + 1,
                  validation_steps=int(x_train[0].shape[0]/BATCH_SIZE) + 1, verbose=1, callbacks=[checkpoint],
                  epochs=EPOCHS)
    plot_train_history(h, model_dir + '/model-history.png')


def train2(x_train, y_train, x_test, y_test, model_dir='models/LSTM'):
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_data = val_data.batch(BATCH_SIZE).repeat()

    create_directory(model_dir)
    checkpoint = ModelCheckpoint(model_dir + '/weights-{epoch:03d}-{val_loss:.4f}-.hdf5', monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='auto')

    # model = create_mlp(input_dim=x_train.shape[1], output_dim=y_train.shape[-1])
    # model = create_lstm(input_shape=x_train.shape[-2:], output_dim=y_train.shape[-1])
    # model = create_lstm_cnn(input_shape=x_train.shape[-2:], output_dim=y_train.shape[-1])
    # model = create_cnn_timestep(input_shape=x_train.shape[-2:], output_dim=y_train.shape[-1])
    h = model.fit(train_data, validation_data=val_data, steps_per_epoch=int(x_train.shape[0]/BATCH_SIZE) + 1,
                  validation_steps=int(x_train.shape[0]/BATCH_SIZE) + 1, verbose=1, callbacks=[checkpoint],
                  epochs=EPOCHS)
    plot_train_history(h, model_dir + '/model-history.png')

    '''
    print('train2 evaluating...')

    i = 0
    for x, y in val_data.take(10):
        multi_step_plot(x[0], y[0], model.predict(x)[0], i)
        i += 1
    '''


def train(data_file_name='./data/Basaksehir.csv', model_dir='models/LSTM', bach_size=512):
    ds = Dataset(data_file_name)
    ds.read_csv_data()
    ds.normalize()
    x_train, y_train, x_test, y_test = ds.get_train_test(window_size=240, predict_period=48)

    create_directory(model_dir)

    checkpoint = ModelCheckpoint(model_dir + '/weights-{epoch:03d}-{val_loss:.4f}-.hdf5', monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='auto')

    model = create_lstm(input_shape=x_train.shape[-2:], lstm_dim=64, output_dim=y_train.shape[-1], dropout=.2)
    h = model.fit(x_train, y_train, validation_data=[x_test, y_test], batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                  shuffle=True, callbacks=[checkpoint])

    plot_train_history(h, model_dir + '/model-history.png')


def evaluate(x_test, y_test, model_dir='models/LSTM', plot_count=10, plot_dir='plots/', is2input=True):

    weights = os.listdir(model_dir)
    best_val = 100
    best_weights = ''
    for weight in weights:
        if len(weight.split('-')) > 2:
            val_loss = weight.split('-')
            val_loss = float(val_loss[2])
            if val_loss < best_val:
                best_val = val_loss
                best_weights = weight

    trained_model = load_model(os.path.join(model_dir, best_weights))
    if type(x_test) == list and is2input:
        pred = trained_model.predict({'serial_input': x_test[0], 'vector_input': x_test[1]}, verbose=1)
        x_test = x_test[0]
    elif type(x_test) == list and not is2input:
        pred = trained_model.predict(x_test[0], verbose=1)
        x_test = x_test[0]
    else:
        pred = trained_model.predict(x_test, verbose=1)

    x_axis = np.hstack((np.arange(x_test.shape[1])[::-1]*-1, np.arange(y_test.shape[1] + 1)[1:]))
    plot_instance_indices = np.array([i * 512 for i in range(plot_count)])
    # np.random.randint(x_test.shape[0], size=plot_count)

    for index in plot_instance_indices:
        if index < x_test.shape[0]:
            plt.figure(figsize=(12, 6))
            plt.plot(x_axis, np.hstack((x_test[index, :, -1], pred[index])), label="predicted")
            plt.plot(x_axis, np.hstack((x_test[index, :, -1], y_test[index])), label="Observation")
            plt.title('LSTM model result on test instance ' + str(index))
            plt.xlabel('time')
            plt.ylabel('Normalized Concentration')
            plt.legend()
            plt.draw()
            create_directory(plot_dir)
            plt.savefig(plot_dir + str(index))
            plt.close()


ds = Dataset('./data/Basaksehir.csv')
ds.read_csv_data()
ds.normalize()
x_train, y_train, x_test, y_test = ds.get_train_test(window_size=240, predict_period=48)

# x_train = ds.get_up_scale_data(x_train, 4)
# x_test_feature = ds.get_up_scale_data(x_test, 4)

x_train_feature = ds.get_up_scale_data_parallel(x_train, 4)
x_test_feature = ds.get_up_scale_data_parallel(x_test, 4)

np.random.seed(13)
tf.random.set_seed(13)
model_dir = 'models/cnn100-mlp100drop0.3-attention-all-features'
# train2(x_train, y_train, x_test_feature, y_test, model_dir=model_dir)
train2input([x_train, x_train_feature], y_train, [x_test, x_test_feature], y_test, model_dir=model_dir)
evaluate([x_test, x_test_feature], y_test, model_dir=model_dir, plot_dir='plots/cnn100-mlp100drop0.3-attention-all-features/', plot_count=20)

