from sklearn.linear_model import LinearRegression
import numpy as np

### Detrend function
def detrend(confirmed_case,window_size):
    i=0
    train_list_detrended = np.empty((0))
    num_days = list()
    trend_list = np.empty((0))
   
    while(i+window_size<=confirmed_case.shape[0]):

        # fit linear model
        X = list(range(0,window_size))
        X = np.reshape(X, (len(X), 1))
        y = confirmed_case[i:i+window_size]
        model = LinearRegression()
        model.fit(X, y)
        
        # calculate trend
        trend = model.predict(X)
        
        # plot trend
        trend_list= np.append(trend_list,trend)
        
        # detrend
        detrended = [y[j]-trend[j] for j in range(0, window_size)]
        
        # plot detrended
        train_list_detrended = np.append(train_list_detrended,detrended)
        i += window_size

    return train_list_detrended, trend_list