# COMPARISON OF FITNESS BANDS

# MI band

# Data Preprocessing


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from mpl_toolkits.mplot3d import Axes3D
from sklearn.impute import SimpleImputer  

# Importing Datasets
dataset_activity = pd.read_csv('ACTIVITY_a.csv').drop_duplicates()
dataset_activity['date'] = dataset_activity.date.astype('datetime64[ns]')
dataset_sleep = pd.read_csv('SLEEP_a.csv').drop_duplicates()

# Average heartbeat calculation
date_hb = pd.read_csv('HEARTRATE_a.csv').drop(['time'], axis = 1)
date_hb.to_csv('HEARTRATE.csv', index = False)
hb = pd.read_csv('HEARTRATE.csv', parse_dates = ['date'], index_col = 'date')
avg_hb = hb.resample('D').mean()    
avg_hb = avg_hb.reset_index()

# Merge average heartbeat/day to main dataset
hb_merged = dataset_activity.merge(avg_hb, on = 'date')
hb_merged['lastSyncTime'] = pd.to_datetime(hb_merged['lastSyncTime'],unit='s')
hb_merged = hb_merged.drop(['lastSyncTime'], axis = 1)
hb_merged.to_csv("hb_merge.csv")

# Merging dataset_sleep to dataset_activity
dataset_sleep['date'] = dataset_sleep['date'].astype('datetime64[ns]')
dataset_merged = hb_merged.merge(dataset_sleep, on = 'date').drop('lastSyncTime', axis =1)
dataset_merged['start'] = pd.to_datetime(dataset_merged['start'], unit='s')
dataset_merged['stop'] = pd.to_datetime(dataset_merged['stop'], unit='s')


# For 0 values in hb_merged (activity columns)
x = hb_merged.iloc[:,1:5].values
imputer_zero = SimpleImputer(missing_values = 0, strategy = "mean")
imputer_zero = imputer_zero.fit(x)
x = (imputer_zero.transform(x))
x = x.astype('int64')
x = np.array(x)

# For nan values in hb_merged (heartbeat column)
y = hb_merged.iloc[:,5:6].values
imputer_nan = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer_nan = imputer_nan.fit(y[:,:])
y = (imputer_nan.transform(y[:,:]))
y = y.astype('int64')
y = np.array(y)

# Dataframe with 0 and nan values removed
z = np.concatenate((x,y), axis=1)
z = pd.DataFrame(z)


# Machine Learning

X = z.iloc[:, :3].values # Independent variable matrix
Y = z.iloc[:, 3].values # Dependent variable martix

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) 

# Fitting Multiple Linear Regression into Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the test set results
Y_pred = regressor.predict(X_test)
Y_pred = Y_pred.astype('int64')


# Backward Elimination Algorithm
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((X.shape[0], 1)).astype(int), values = X, axis = 1)  # first column must have value 1 for b0 constant

# Removing independent variables that are statistically insignificant
X_opt = X[:,[0,1,2]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() 
regressor_OLS.summary()

X_opt = X[:,[0,2]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

# Most Significant Feature
X_opt = X[:,[1]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()


# Visualising the regression results


# Steps VS Calories
plt.scatter(X[:,1], Y, color = 'red')
plt.plot(X[:,1], Pre, color = 'blue')
plt.title('Fitness Analysis')
plt.xlabel('Steps')
plt.ylabel('Calories')
plt.show()

indep=X[:,1]
Pre=regressor_OLS.predict(X[:,1])

fig, ax = plt.subplots()
line, = ax.plot(indep, Pre, color='k')



def update(num, x, y, line):
    line.set_data(indep[:num],Pre[:num] )
    line.axes.axis([2500, 11000, 350, 2000])
    return line,

ani = animation.FuncAnimation(fig, update, len(X[:,1]), fargs=[X[:,1],regressor_OLS.predict(X[:,1]) , line],
                              interval=500, blit=False)

ani.save('sVc.mp4')



# Distance VS Calories
regressor_OLS = sm.OLS(endog = Y, exog = X[:,[2]]).fit()
plt.scatter(X[:,2], Y, color = 'red')
plt.plot(X[:,2], regressor_OLS.predict(X[:,2]), color = 'black')
plt.title('Fitness Analysis')
plt.xlabel('Distance')
plt.ylabel('Calories')
plt.show()

indep2=X[:,2]
Pre2= regressor_OLS.predict(X[:,2])
fig, ax = plt.subplots()
line, = ax.plot(indep, Pre, color='k')

def update(num, x, y, line):
    line.set_data(indep[:num],Pre[:num] )
    line.axes.axis([3900, 14200, 350, 1800])
    return line,

ani = animation.FuncAnimation(fig, update, len(X[:,1]), fargs=[X[:,1],regressor_OLS.predict(X[:,1]) , line],
                              interval=500, blit=False)

ani.save('dVc.mp4')





# 3D graph (Multiple Regression)

fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.scatter(dataset_merged['distance'], dataset_merged['steps'], dataset_merged['calories'], c='blue', marker = 'o', s = 10)
ax.view_init(30, 185)
#plt.xlabel('Distance')
#plt.ylabel('Steps')
#plt.zlabel('Calories')
ax.set_xlabel('DISTANCE')
ax.set_ylabel('STEPS')
ax.set_zlabel('CALORIES')

plt.show()


# Fitbit 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fitbit_dataset = pd.read_csv('fitbit_activity_a.csv')
fitbit_dataset['Steps'] = fitbit_dataset["Steps"].str.replace(",","").astype(float)
fitbit_dataset['Date'] = fitbit_dataset.Date.astype('datetime64[ns]')
fitbit_dataset['Calories'] = fitbit_dataset["Calories"].str.replace(",","").astype(float)
fitbit_dataset['Distance'] = fitbit_dataset['Distance'] * 1000

X = fitbit_dataset.iloc[:, 1:3].values # Independent variable matrix
Y = fitbit_dataset.iloc[:, 3].values # Dependent variable martix

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) 

# Fitting Multiple Linear Regression into Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the test set results
Y_pred = regressor.predict(X_test)
Y_pred = Y_pred.astype('int64')

# Backward Elimination Algorithm
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((X.shape[0], 1)).astype(int), values = X, axis = 1)  # first column must have value 1 for b0 constant

# Removing independent variables that are statistically insignificant
X_opt = X[:,[0,1,2]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() 
regressor_OLS.summary()

X_opt = X[:,[0,1]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

# Most Significant Feature
X_opt = X[:,[1]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()


# Visualising the regression results

Pre= regressor_OLS.predict(X[:,1])


# Distance VS Calories
plt.scatter(X[:,1], Y, color = 'red')
plt.plot(X[:,1], regressor_OLS.predict(X[:,1]), color = 'blue')
plt.title('Distance VS Calories')
plt.xlabel('Distance')
plt.ylabel('Calories')
plt.show()


indep1=X[:,1]
Pre1= regressor_OLS.predict(X[:,1])
fig, ax = plt.subplots()
line, = ax.plot(indep, Pre, color='k')

def update(num, x, y, line):
    line.set_data(indep[:num],Pre[:num] )
    line.axes.axis([3900, 14200, 350, 1800])
    return line,

ani = animation.FuncAnimation(fig, update, len(X[:,1]), fargs=[X[:,1],regressor_OLS.predict(X[:,1]) , line],
                              interval=500, blit=False)

ani.save('dVc.mp4')




# Steps VS Calories
regressor_OLS = sm.OLS(endog = Y, exog = X[:,2]).fit()
plt.scatter(X[:,2], Y, color = 'red')
plt.plot(X[:,2], regressor_OLS.predict(X[:,2]), color = 'purple')
plt.title('Steps VS Calories')
plt.xlabel('Steps')
plt.ylabel('Calories')
plt.show()


indep=X[:,2]
Pre=regressor_OLS.predict(X[:,2])
fig, ax = plt.subplots()
line, = ax.plot(indep, Pre, color='k')



def update(num, x, y, line):
    line.set_data(indep[:num],Pre[:num] )
    line.axes.axis([2500, 11000, 350, 2000])
    return line,

ani = animation.FuncAnimation(fig, update, len(X[:,1]), fargs=[X[:,1],regressor_OLS.predict(X[:,1]) , line],
                              interval=500, blit=False)

ani.save('sVc.mp4')




# 3D graph (Multiple Regression)

fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.scatter(fitbit_dataset['Distance'], fitbit_dataset['Steps'], fitbit_dataset['Calories'], c='blue', marker = 'o', s = 10)
ax.view_init(30, 185)


#
#plt.xlabel('Distance')
#plt.ylabel('Steps')
#plt.zlabel('Calories')


ax.set_xlabel('DISTANCE')
ax.set_ylabel('STEPS')
ax.set_zlabel('CALORIES')


plt.show()



