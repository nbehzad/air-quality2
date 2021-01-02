from Dataset import Dataset
from keras.models import load_model
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from scipy.stats import chisquare
import pandas as pd



def evaluate_monthly(x_test, y_test, model_name='lstm', data_name='all'):
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

    #pred = moving_average(pred, 24)                 # Daily average prediction
    #y_test = moving_average(y_test, 24)             # Daily average prediction
    #pred = np.array(pred).reshape(355, 1)            # Daily average prediction
    #y_test = np.array(y_test).reshape(355, 1)        # Daily average prediction

    pred = np.array(pred).reshape(8520, 1)          # Hourly prediction
    y_test = np.array(y_test).reshape(8520, 1)      # Hourly prediction

    combined_df = np.concatenate((pred, y_test), axis=1)

    ### Daily average prediction/observe monthly subsets
    #jan = combined_df[0:31]
    #feb = combined_df[31:59]
    #mar = combined_df[59:90]
    #apr = combined_df[90:120]
    #may = combined_df[120:151]
    #jun = combined_df[151:181]
    #jul = combined_df[181:212]
    #aug = combined_df[212:243]
    #sep = combined_df[243:273]
    #oct = combined_df[273:304]
    #nov = combined_df[304:334]
    #dec = combined_df[334:365]

    ### Hourly prediction/observe monthly subsets
    jan = combined_df[0:744]
    feb = combined_df[744:1416]
    mar = combined_df[1416:2160]
    apr = combined_df[2160:2880]
    may = combined_df[2880:3624]
    jun = combined_df[3624:4344]
    jul = combined_df[4344:5088]
    aug = combined_df[5088:5832]
    sep = combined_df[5832:6552]
    oct = combined_df[6552:7296]
    nov = combined_df[7296:8016]
    dec = combined_df[8016:8760]

    ### Hourly prediction/observe seasonal subset
    #winter_part_1 = combined_df[0:1896]
    #winter_part_2 = combined_df[8520:]
    #winter = np.concatenate((winter_part_1,winter_part_2), axis=0)
    #spring = combined_df[1896:4104]
    #summer = combined_df[4104:6360]
    #fall = combined_df[6360:8520]

    subset_list = [jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec]      # Monthly
    subset_name = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                  'November', 'December']                                          # Monthly

    #subset_list = [winter, spring, summer, fall]                                   # Seasonal
    #subset_name = ['winter', 'spring', 'summer', 'fall']                           # Seasonal

    results_list = []

    for i in range(len(subset_list)):
        prediction = subset_list[i][:, 0]
        observed = subset_list[i][:, 1]

        numerator = sum(abs(prediction - observed))
        denominator = sum(observed)
        acc = round(1 - (numerator/denominator), 4)         # Accuracy
        results_list.append(acc)
        #print('Acc: ' + str(acc))

        numerator = sum((prediction - observed) ** 2)
        denominator = len(observed)
        MSE = round(numerator/denominator, 4)               # Mean Square Error
        results_list.append(MSE)
        #print('MSE: ' + str(MSE))

        RMSE = round((MSE) ** 0.5, 4)                       # Root Mean Square Error
        results_list.append(RMSE)
        #print('RMSE: ' + str(RMSE))

        numerator = sum(abs(prediction - observed))
        denominator = len(observed)
        MAE = round(numerator/denominator, 4)               # Mean Absolute Error
        results_list.append(MAE)
        #print('MAE: ' + str(MAE))

        correlation_matrix = np.corrcoef(observed, prediction)
        correlation_xy = correlation_matrix[0, 1]
        r2 = round(correlation_xy ** 2, 4)                   # Determination Coefficient (R2)
        results_list.append(r2)
        #print('R2: ' + str(r2))

        chi_square = chisquare(f_obs=observed, f_exp=prediction, axis=0)
        chi2 = round(chi_square[0], 4)                       # Chi-square
        critical_value = round(chi_square[1], 4)             # Critical value of chi_square test
        results_list.append(chi2)
        results_list.append(critical_value)
        #print('X2: ' + str(chi2))
        #print('C.V: ' + str(critical_value))

        max_y = prediction.max()
        y_place = max_y
        min_x = observed.min()
        x_place = min_x + 0.01
        month = subset_name[i]
        p = sns.regplot(x=observed, y=prediction, ci=None, color='k', scatter_kws={'s': 5, 'facecolors': 'b', 'edgecolor': 'b'},
                        line_kws={"lw": 1})
        plt.xlabel('Observed')
        plt.ylabel('Predicted')
        p.text(x_place, y_place, month, horizontalalignment='left', size='medium',                           # Add month on plot
               color='black', weight='semibold')

        p.text(x_place, y_place - 0.015, 'Acc. = ' + str(acc), horizontalalignment='left', size='medium',     # Add Acc. value on plot
               color='black', weight='semibold')

        plt.savefig(data_name + '_regplot_' + month + '.png')
        #plt.show()
        plt.close()

    #results_arr = np.array(results_list).reshape(12,7)                          # Monthly
    results_arr = np.array(results_list).reshape(4, 7)
    results_arr = results_arr.transpose()
    results_df = pd.DataFrame(data=results_arr, index=['Acc.', 'MSE', 'RMSE', 'MAE', 'R2', 'X2', 'C.V.'],
                                  columns=subset_name)
    results_df.to_csv('final_results_' + data_set_name + '.csv')

data_set = './data/Sultangazi.csv'

ds = Dataset(data_set)
ds.read_csv_data()
ds.normalize()
x_train, y_train, x_test, y_test = ds.get_train_test(window_size=240, predict_period=48)

data_set_name = os.path.splitext(os.path.basename(data_set))[0]

evaluate_monthly(x_test, y_test, model_name='CNN-with-kernel4', data_name=data_set_name)