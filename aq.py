#         ****  This is the final version of LSTM RNN (for Istanbul)  ****

#       Time series forecasting of ozone on Istanbul dataset using LSTM RNN


#  Code ref:      https://habr.com/ru/post/495884/
#  See also,      https://www.kaggle.com/nicapotato/keras-timeseries-multi-step-multi-output?select=multi-output-timesteps.h5
#                 https://www.tensorflow.org/tutorials/structured_data/time_series
#                 https://hk29.hatenablog.jp/entry/2020/05/17/214955?utm_source=feed
#          *****  https://towardsdatascience.com/why-lstms-work-well-with-volatile-time-series-1bd2fd4ade62
#          *****  https://www.michael-grogan.com/hotel-modelling/articles/lstm_adr

#                 https://towardsdatascience.com/simplifying-grus-lstm-and-rnns-in-general-8f0715c20228
#                 https://towardsdatascience.com/temporal-convolutional-networks-the-next-revolution-for-time-series-8990af826567

#  select learning rate:   ((((((((((Adaptive Computation and Machine Learning series.pdf

import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import gc
import os
import pandas as pd
from keras.callbacks import CSVLogger, EarlyStopping

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# ========================================== Input the file

'''
a= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ds = tf.data.Dataset.from_tensor_slices(a)
ds = ds.batch(2).repeat()
aa = list(ds.take(3).as_numpy_iterator())
'''


csv_path = "./data/Basaksehir.csv"
df = pd.read_csv(csv_path)

# Setting seed to ensure reproducibility.
tf.random.set_seed(13)

TRAIN_SPLIT = 35053  # 80% train (35055/43814)

# ========================================== Data preaparation

# Define the features that should used in the model
features_considered = ['Pressure', 'RH', 'Temperature', 'Precipitation', 'Wind_direction',
                       'Wind_speed', 'Solar_radiation', 'NO2', 'NOX', 'Average_8h_NOX', 'Max_8h_NOX', 'O3']

features = df[features_considered]
# features.index = df['Date']
features.head()
# features.plot(subplots=True)


dataset = features.values

# Here a part of dataset was defined as train dataset
#train_data = dataset[:TRAIN_SPLIT]

# ================== Get the max and min values of train dataest

"""
The input dataset should be scaled between 0 and 1 using the max and min values 
of train dataset features. So, the max and min values of train dataset features 
should be saved to use in scaling of the dataset. 
In this section I create a list (min_max_data) to keep max and min values of train 
dataset before scaling.
"""
'''
lister = []
max_values = train_data.max(axis=0)  # get max values of train data features
lister.append(max_values)

min_values = train_data.min(axis=0)  # get min values of train data features
lister.append(min_values)

merge_values = np.vstack(lister)  # np.vstack()  => stack arrays in sequence vertically (row wise)
min_max_data = pd.DataFrame(merge_values)  # min and max values of train dataset features
min_max_data.columns = features_considered
'''
# ================== Scaling the train and validation datasets

# Scaling dataset between 0 to 1, based on max and min values of train features 
# The last line of the for loop is infact the MinMaxScaler formula
'''
for i in range(dataset.shape[1]):  # for column
    for j in range(dataset.shape[0]):  # for row
        if dataset[j][i] > min_max_data.iat[0, i]:
            dataset[j][i] = min_max_data.iat[0, i]
        if dataset[j][i] < min_max_data.iat[1, i]:
            dataset[j][i] = min_max_data.iat[1, i]
        dataset[j][i] = (dataset[j][i] - min_max_data.iat[1, i]) / (min_max_data.iat[0, i] - min_max_data.iat[1, i])
'''

    # ================== Creating sample windows

"""
In a single step setup, the model learns to predict a single point in the future 
based on some history provided.
The below function performs the same windowing task as below, however, here it 
samples the past observation based on the step size given.
"""


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


"""
past_history = the model uses past data qeual to "past_history" to predict the 
future. For example, in a hourly dataset if we want to use 7 days data to train
the model, past_history equals to 168 (7*24).
future_target = is the point thet we want to predict (in single prediction). 
STEP = shows the time stpes of prediction. For example, if we have hourly dataset
and we want to have hourly predictions STEP equals to 1.



past_history = 120        # 24 hr * 7 days
future_target = 6
STEP = 1

# In "dataset[:, 7]" 7 is the index of ozone column data, which should be predicted.

x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 9], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 9],
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=True)

print(x_train_single.shape)                                                   
# Let's look at a single data-point.
print ('Single window of past history : {}'.format(x_train_single[0].shape))
print(x_train_single.shape[-2:])                                              
"""
# ========================================== Multivariate single step prediction

# Let's now use tf.data to shuffle, batch, and cache the dataset.
"""
tf data is designed to work with possibly infinite sequences, so it doesn't attempt 
to shuffle the entire sequence in memory. Instead,it maintains a buffer in which 
it shuffles elements.
"""
"""
For perfect shuffling, set the buffer size equal to the full size of the dataset.
Ref:  https://www.tensorflow.org/api_docs/python/tf/data/experimental/shuffle_and_repeat

Bach size = 32 is a good default value, with values above 10 taking advantage 
of the speed-up of matrixmatrix products over matrix-vector products. 
Ref:  sPractical Recommendations for Gradient-Based Training of Deep Architecture, Bengio (2012)


BATCH_SIZE = 256
BUFFER_SIZE = 20000


train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

single_step_model = tf.keras.models.Sequential()   
single_step_model.add(tf.keras.layers.LSTM(32,           # LSTM layer (input)
                                           input_shape=x_train_single.shape[-2:]))   # ".shape[-2:]"  means all rows and columns of a window 
single_step_model.add(tf.keras.layers.Dense(1))    # output

single_step_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae', metrics=['accuracy'])

print(single_step_model.summary())

# Let's check out a sample prediction.
for x, y in val_data_single.take(1):
  print(single_step_model.predict(x).shape)

"""
# ================== Training the model

"""
steps_per_epoch: the default None is equal to the number of samples in your dataset 
divided by the batch size, or 1 if that cannot be determined.

validation_steps: Total number of steps (batches of samples) to draw before stopping 
when performing validation at the end of every epoch. If 'validation_steps' is None, 
validation will run until the validation_data dataset is exhausted. If 'validation_steps' 
is specified and only part of the dataset will be consumed, the evaluation will start 
from the beginning of the dataset at each epoch. This ensures that the same validation 
samples are used every time.

Adjustment in:
    https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit


EPOCHS = 1
EVALUATION_INTERVAL = 200


#print(f"Evaluation Threshold: {EVALUATION_INTERVAL}", f"Epochs: {EPOCHS}", sep="\n")

early_stopping = EarlyStopping(monitor='val_loss', patience = 3, restore_best_weights=True)
single_step_history = single_step_model.fit(train_data_single,
                                            epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            callbacks=[early_stopping],
                                            validation_steps=50)

"""


def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


# plot_train_history(single_step_history, 'Single Step Training and validation loss')


# ================== Predict a single step future:

"""
Now that the model is trained, let's make a few sample predictions. The model is 
given the history of three features over the past five days sampled every hour 
(120 data-points), since the goal is to predict the temperature, the plot only 
displays the past temperature. The prediction is made one day into the future 
(hence the gap between the history and prediction).
"""


def create_time_steps(length):
    return list(range(-length, 0))

'''
def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'ro', 'gx']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()

    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel('Time-Step')
    return plt



for x, y in val_data_single.take(3):
  plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                    single_step_model.predict(x)[0]], 12,
                   'Single Step Prediction')
  plot.show()


del single_step_history, val_data_single, train_data_single                   
'''
# ========================================== Multivariate multi step prediction

"""
There are three built-in RNN layers in Keras:
    1. keras.layers.SimpleRNN, a fully-connected RNN where the output from 
       previous timestep is to be fed to next timestep.
    2. keras.layers.GRU, first proposed in Cho et al., 2014.
    3. keras.layers.LSTM, first proposed in Hochreiter & Schmidhuber, 1997.

Ref:   https://www.tensorflow.org/guide/keras/rnn
       https://www.youtube.com/watch?v=wY0dyFgNCgY
"""


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




past_history = 240  # 24 hr * 10 days
future_target = 48  # 80
STEP = 1

dataset = (dataset - dataset.min(0)) / dataset.ptp(0)

x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, -1], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, -1],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)

print('Single window of past history : {}'.format(x_train_multi[0].shape))
print('\n Target ozone to predict : {}'.format(y_train_multi[0].shape))

BATCH_SIZE = 512
BUFFER_SIZE = 30000
train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()


multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32, activation='tanh',  # "activation='relu'"  ben ekledim
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[
                                                      -2:]))  # ".shape[-2:]"  means all rows and columns of a window
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(48))

# To select beta values:   https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
multi_step_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mae',
                         metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])
print(multi_step_model.summary())


EPOCHS = 10
EVALUATION_INTERVAL = 200

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
multi_step_history = multi_step_model.fit(train_data_multi,
                                          epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=EVALUATION_INTERVAL,
                                          callbacks=[early_stopping])

plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

i = 0
for x, y in val_data_multi.take(20):
    multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0], i)
    i += 1

del multi_step_model, val_data_multi, train_data_multi
_ = gc.collect()                                                                 