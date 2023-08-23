import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.tree import DecisionTreeRegressor
from scipy.signal import butter, lfilter
import scipy.stats
import os
import functions_mea_prediction as func

directory = '/Volumes/Elements/spring2022/final/week4/'
cultures = ['NS', '8020', '5050']

ff = 0
#R2_array = np.empty([540,3])
#data_real = np.empty([2000000,3])
#data_pred = np.empty([2000000, 3])

for folder in cultures:
    ff += 1
    population = os.path.join(directory, folder)
    os.chdir(population)
    aa = os.getcwd()
    R2 = []
    data_real = np.zeros([1,1])
    data_pred = np.zeros([1,1])

    for meas in os.listdir(aa):
        mea_num = os.path.join(aa, meas)
        if not meas.startswith('.'):
            print(meas)
            os.chdir(mea_num)
            ii = 0
            for ch_data in os.listdir(mea_num):

                if ch_data.endswith('data.mat') and ch_data.startswith('ch'):
                    ii += 1
                    data = sio.loadmat(ch_data)
                    mdata = data['data']
                    timestamps = np.arange(1, len(mdata))
                    mdata = mdata.reshape(len(mdata), 1)
                    mdtype = mdata.dtype
                    lowcut = 300
                    highcut = 3000
                    fs = 25000
                    y = func.butter_lowpass_filter(mdata, lowcut, fs, order=5)
                    df = pd.DataFrame(y, columns=['Voltage'])
                    del (data, mdata)
                    downsampled_df = df.iloc[::100, :]  # from 25kHz to 250 Hz
                    del (df)
                    downsampled_df['y'] = downsampled_df['Voltage'].shift(-1)

                    train = downsampled_df[0:round(0.7 * len(downsampled_df))]  # 70% split
                    test = downsampled_df[round(0.7 * len(downsampled_df)) + 1:len(downsampled_df)]
                    test = test.drop(test.tail(1).index)

                    X_train = train['Voltage'].values.reshape(-1, 1)
                    y_train = train['y'].values.reshape(-1, 1)
                    X_test = test['Voltage'].values.reshape(-1, 1)
                    # Initialize the model
                    dt_reg = DecisionTreeRegressor(random_state=42)

                    # Fit the model
                    dt_reg.fit(X=X_train, y=y_train)

                    # Make predictions
                    dt_pred = dt_reg.predict(X_test)


                    r2 = scipy.stats.pearsonr(test['y'], dt_pred)

                    R2.append(r2.statistic)

                    fig, ax = plt.subplots(figsize=(16, 11))

                    ax.plot(np.arange(0, round(0.7 * len(downsampled_df)), 1),
                            downsampled_df[0: round(0.7 * len(downsampled_df))], marker='.', color='black', label='Train')
                    ax.plot(np.arange(round(0.7 * len(downsampled_df)) + 1, len(downsampled_df) - 1, 1), X_test, marker='.',
                            color='blue', label='Actual')
                    ax.plot(np.arange(round(0.7 * len(downsampled_df)) + 1, len(downsampled_df) - 1, 1), dt_pred,
                            marker='^', color='green', label='Decision Tree predictions')
                    ax.set_xticks([1 * len(downsampled_df) / 5, 2 * len(downsampled_df) / 5, 3 * len(downsampled_df) / 5,
                                   4 * len(downsampled_df) / 5, 5 * len(downsampled_df) / 5])
                    ax.set_xticklabels(['1', '2', '3', '4', '5'])
                    ax.set_xlabel('Time (min)')
                    ax.set_ylabel('Voltage')
                    plt.xlim([0, len(downsampled_df)])
                    plt.legend()
                    #plt.show()
                    plt.draw()
                    figure_name = 'prediction_' + str(folder) + '_' + str(meas) + '_ch' + str(ch_data) + '_250Hz.pdf'
                    fig.savefig(figure_name, dpi=300)
                    plt.close()

                    dt_pred = dt_pred.reshape(len(dt_pred),1)
                    data_pred = np.vstack((data_pred, dt_pred))

                    data_real = np.concatenate((data_real, X_test))

    filename_r2 = 'R2_prediction_' + cultures[ff-1] + '_250Hz.csv'
    filename_pred = 'Pred_prediction_' + cultures[ff-1] + '_250Hz.csv'
    filename_real = 'Real_prediction_' + cultures[ff-1] + '_250Hz.csv'
    np.savetxt(directory + 'Predictor/' + filename_r2, R2, delimiter=',')
    np.savetxt(directory + 'Predictor/' + filename_pred, data_pred, delimiter=',')
    np.savetxt(directory + 'Predictor/' + filename_real, data_real, delimiter=',')







