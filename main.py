from Dataset import Dataset
from NN_Models import *
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import argparse

STEP = 1
BATCH_SIZE = 512
EPOCHS = 20
BUFFER_SIZE = 30000
SEED = 13


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


def train(x_train, y_train, x_test, y_test, model_name='lstm', data_name='all', window_size=None, stride_size=1):
    model_dir = 'models/' + data_name + '/' + model_name
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_data = val_data.batch(BATCH_SIZE).repeat()

    create_directory(model_dir)
    checkpoint = ModelCheckpoint(model_dir + '/weights-{epoch:03d}-{val_loss:.4f}-.hdf5', monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='auto')

    if str.lower(model_name).startswith('mlp'):
        model = create_mlp(input_shape=x_train.shape[-2:], output_dim=y_train.shape[-1])
    elif str.lower(model_name).startswith('lstm-with-pooling'):
        model = create_lstm_with_pooling(input_shape=x_train.shape[-2:], output_dim=y_train.shape[-1],
                                         pooling=window_size)
    elif str.lower(model_name).startswith('cnn-base'):
        model = create_cnn_base(input_shape=x_train.shape[-2:], output_dim=y_train.shape[-1])
    elif str.lower(model_name).startswith('cnn-with-kernel'):
        model = create_cnn(input_shape=x_train.shape[-2:], output_dim=y_train.shape[-1], kernel=window_size,
                           stride=stride_size)
    elif str.lower(model_name).startswith('cnn-lstm'):
        model = create_cnn_lstm(input_shape=x_train.shape[-2:], output_dim=y_train.shape[-1], kernel=window_size,
                                stride=stride_size)
    elif str.lower(model_name).startswith('lstm-cnn'):
        model = create_lstm_cnn(input_shape=x_train.shape[-2:], output_dim=y_train.shape[-1], kernel=window_size,
                                stride=stride_size)
    else:
        print('Invalid model name')
        return

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


def evaluate(x_test, y_test, model_name='lstm', data_name='all', plot_count=10, is2input=True):
    model_dir = 'models/' + data_name + '/' + model_name
    plot_dir = 'plots/' + data_name + '/' + model_name + '/'
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
        pred = trained_model.predict(x_test[1], verbose=1)
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
            plt.title('{} model on instance {}'.format(model_name, str(index)))
            plt.xlabel('Time(h)')
            plt.ylabel('Normalized Ozone Concentration')
            plt.legend()
            plt.draw()
            create_directory(plot_dir)
            plt.savefig(plot_dir + str(index))
            plt.close()


def evaluate_yearly(x_test, y_test, model_name='lstm', data_name='all'):
    def moving_average(x, s):
        return [np.mean(x[j:j + s]) for j in range(0, len(x), s)]

    model_dir = 'models/' + data_name + '/' + model_name
    plot_dir = 'plots/' + data_name + '/' + model_name + '/'
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
    if type(x_test) == list:
        pred = trained_model.predict({'serial_input': x_test[1], 'vector_input': x_test[1]}, verbose=1)
    else:
        pred = trained_model.predict(x_test, verbose=1)

    pred = [np.pad(pred[i], (i, pred.shape[0] - 1 - i), 'constant', constant_values=(np.nan,))
            for i in range(pred.shape[0])]
    pred = np.nanmean(pred, axis=0)

    y_test = [np.pad(y_test[i], (i, y_test.shape[0] - 1 - i), 'constant', constant_values=(np.nan,))
              for i in range(y_test.shape[0])]
    y_test = np.nanmean(y_test, axis=0)

    pred = moving_average(pred, 24)
    y_test = moving_average(y_test, 24)

    x_axis = np.arange(len(y_test) + 1)[1:]

    plt.figure(figsize=(16, 6))
    plt.plot(x_axis, pred, label="predicted", linewidth=0.8)
    plt.plot(x_axis, y_test, label="Observation", linewidth=0.8, alpha=0.5)
    plt.title('{} model'.format(model_name))
    plt.xlabel('Time(24h)')
    plt.ylabel('Normalized Ozone Concentration')
    plt.legend()
    plt.draw()
    create_directory(plot_dir)
    plt.savefig(plot_dir + model_name + '-yearly-prediction-plot')
    plt.close()


def run_mlp_experiment(data_set):
    ds = Dataset(data_set)
    ds.read_csv_data()
    ds.normalize()
    x_train, y_train, x_test, y_test = ds.get_train_test(window_size=240, predict_period=48)

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    data_set_name = os.path.splitext(os.path.basename(data_set))[0]

    print('\n\n\nTraining begins for model MLP...\n')
    train(x_train, y_train, x_test, y_test, model_name='MLP', data_name=data_set_name)
    evaluate(x_test, y_test, model_name='MLP', data_name=data_set_name, plot_count=20)
    evaluate_yearly(x_test, y_test, model_name='MLP', data_name=data_set_name)

    for i in [4, 8, 16, 24]:
        print('\n\n\nTraining begins for model MLP-with-pooling{}...\n'.format(str(i)))
        x_train_strided = ds.get_up_scale_data(x_train, i)
        x_test_strided = ds.get_up_scale_data(x_test, i)
        train(x_train_strided, y_train, x_test_strided, y_test, model_name='MLP-with-pooling{}'.format(str(i))
              , data_name=data_set_name)
        evaluate([x_test, x_test_strided], y_test, model_name='MLP-with-pooling{}'.format(str(i)),
                 data_name=data_set_name, plot_count=20,
                 is2input=False)
        evaluate_yearly(x_test_strided, y_test, model_name='MLP-with-pooling{}'.format(str(i)), data_name=data_set_name)


def run_lstm_experiment(data_set):
    ds = Dataset(data_set)
    ds.read_csv_data()
    ds.normalize()
    x_train, y_train, x_test, y_test = ds.get_train_test(window_size=240, predict_period=48)

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    data_set_name = os.path.splitext(os.path.basename(data_set))[0]

    for i in [1]:
        print('\n\n\nTraining begins for model LSTM-with-pooling{}...\n'.format(str(i)))
        train(x_train, y_train, x_test, y_test, model_name='LSTM-with-pooling{}'.format(str(i)),
              data_name=data_set_name, window_size=i)
        evaluate(x_test, y_test, model_name='LSTM-with-pooling{}'.format(str(i)),
                 data_name=data_set_name, plot_count=20)
        evaluate_yearly(x_test, y_test, model_name='LSTM-with-pooling{}'.format(str(i)), data_name=data_set_name)


def run_cnn_experiment(data_set):
    ds = Dataset(data_set)
    ds.read_csv_data()
    ds.normalize()
    x_train, y_train, x_test, y_test = ds.get_train_test(window_size=240, predict_period=48)

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    data_set_name = os.path.splitext(os.path.basename(data_set))[0]

    print('\n\n\nTraining begins for model CNN-base ...\n')
    train(x_train, y_train, x_test, y_test, model_name='CNN-base', data_name=data_set_name)
    evaluate(x_test, y_test, model_name='CNN-base', data_name=data_set_name, plot_count=20)
    evaluate_yearly(x_test, y_test, model_name='CNN-base', data_name=data_set_name)

    for i in [2, 4, 8, 16, 24]:
        print('\n\nTraining begins for model CNN-with-kernel{} ...\n'.format(str(i)))
        train(x_train, y_train, x_test, y_test, model_name='CNN-with-kernel{}'.format(str(i)), data_name=data_set_name,
              window_size=i, stride_size=i)
        evaluate(x_test, y_test, model_name='CNN-with-kernel{}'.format(str(i)), data_name=data_set_name, plot_count=20)
        evaluate_yearly(x_test, y_test, model_name='CNN-with-kernel{}'.format(str(i)), data_name=data_set_name)


def run_cnn_lstm_experiment(data_set):
    ds = Dataset(data_set)
    ds.read_csv_data()
    ds.normalize()
    x_train, y_train, x_test, y_test = ds.get_train_test(window_size=240, predict_period=48)

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    data_set_name = os.path.splitext(os.path.basename(data_set))[0]

    for i in [2, 4, 8, 16, 24]:
        print('\n\n\nTraining begins for model CNN-LSTM-with-kernel{}...\n'.format(str(i)))
        train(x_train, y_train, x_test, y_test, model_name='CNN-LSTM-with-kernel{}'.format(str(i)),
              data_name=data_set_name, window_size=i, stride_size=i)
        evaluate(x_test, y_test, model_name='CNN-LSTM-with-kernel{}'.format(str(i)), data_name=data_set_name,
                 plot_count=20)
        evaluate_yearly(x_test, y_test, model_name='CNN-LSTM-with-kernel{}'.format(str(i)), data_name=data_set_name)


def run_lstm_cnn_experiment(data_set):
    ds = Dataset(data_set)
    ds.read_csv_data()
    ds.normalize()
    x_train, y_train, x_test, y_test = ds.get_train_test(window_size=240, predict_period=48)

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    data_set_name = os.path.splitext(os.path.basename(data_set))[0]

    for i in [2, 4, 8, 16, 24]:
        print('\n\n\nTraining begins for model LSTM-CNN-with-kernel{}...\n'.format(str(i)))
        train(x_train, y_train, x_test, y_test, model_name='LSTM-CNN-with-kernel{}'.format(str(i)),
              data_name=data_set_name, window_size=i, stride_size=i)
        evaluate(x_test, y_test, model_name='LSTM-CNN-with-kernel{}'.format(str(i)), data_name=data_set_name,
                 plot_count=20)
        evaluate_yearly(x_test, y_test, model_name='LSTM-CNN-with-kernel{}'.format(str(i)), data_name=data_set_name)


def main(args):
    parser = argparse.ArgumentParser(
        description='run ozone forecasting models on Istanbul data sets')
    parser.add_argument('-dataset', help='The name of data set existing in data directory', default='all')
    parser.add_argument('-model', help='The name of the model including mlp, lstm, cnn, lstm-cnn, cnn-lstm',
                        default='all')

    args = vars(parser.parse_args())
    dataset_name = './data/' + args['dataset'] + '.csv'
    model_name = args['model']

    print('\n\nExperiment is started for dataset {} using {} model ...'.format(dataset_name, str.upper(model_name)))

    if model_name == 'mlp' and args['dataset'] != 'all':
        run_mlp_experiment(dataset_name)
    elif model_name == 'lstm' and args['dataset'] != 'all':
        run_lstm_experiment(dataset_name)
    elif model_name == 'cnn' and args['dataset'] != 'all':
        run_cnn_experiment(dataset_name)
    elif model_name == 'cnn-lstm' and args['dataset'] != 'all':
        run_cnn_lstm_experiment(dataset_name)
    elif model_name == 'lstm-cnn' and args['dataset'] != 'all':
        run_lstm_cnn_experiment(dataset_name)
    elif model_name == 'all' and args['dataset'] != 'all':
        print('All experiments are running...')
        run_mlp_experiment(dataset_name)
        run_lstm_experiment(dataset_name)
        run_cnn_experiment(dataset_name)
        run_cnn_lstm_experiment(dataset_name)
        run_lstm_cnn_experiment(dataset_name)
    elif model_name == 'all' and args['dataset'] == 'all':
        for data_name in ['Alibeykoy', 'Basaksehir', 'Besiktas', 'Esenyurt', 'Kadikoy', 'Kagithane', 'Sultanbeyli',
                          'Sultangazi']:

            print('All experiments are running for {}...'.format(data_name))
            dataset_name = './data/' + data_name + '.csv'
            run_mlp_experiment(dataset_name)
            run_lstm_experiment(dataset_name)
            run_cnn_experiment(dataset_name)
            run_cnn_lstm_experiment(dataset_name)
            run_lstm_cnn_experiment(dataset_name)
    else:
        print('Invalid parameters')

    print()


if __name__ == '__main__':
    args = sys.argv
    main(args)
