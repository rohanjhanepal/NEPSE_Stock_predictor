import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
print("Importing packages")

def appender(da):
    global dataset  , data
    dataset = np.concatenate([dataset , np.array([[da]])])
    data = data.append(pd.DataFrame({"Close":[da]}) , ignore_index = True)

def scaleit():
    global scaled_data
    scaled_data = scaler.fit_transform(dataset)

def tester():
    global text_data , x_test , y_test  ,training_data_len
    training_data_len = math.ceil(len(dataset) * 0.8)
    test_data = scaled_data[training_data_len - 60: , :]
    x_test = []
    y_test = dataset[training_data_len: , :]
    for i in range(60 , len(test_data)):
        x_test.append(test_data[i-60:i , 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test , (x_test.shape[0] , x_test.shape[1] , 1))
    
def plot_me():
    global data , training_data_len
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    plt.figure(figsize=(32,16))
    plt.title('Moded')
    plt.xlabel('Date' , fontsize=18)
    plt.ylabel('Close $' , fontsize=18)
    plt.plot(train['Close'] )
    plt.plot(valid[['Close' , 'Predictions']])
    plt.legend(['Train' , 'Actual' , 'Predictions'] , loc = 'lower right')
    plt.show()

def predict_next():
    global model , scaler , x_test , predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    pre = predictions[-1][0]
    print(pre)
    print(len(x_test))
    appender(pre)
    scaleit()
    tester()
  
    
if __name__ == "__main__" :
    df = pd.read_csv('upper.csv')
    #df.set_index("Date")
    print(df.head())
    data=df.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * 0.8)
    print("loading data")
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:training_data_len , :]
    
    x_train = []
    y_train = []
    for i in range(60 , len(train_data)):
        x_train.append(train_data[i-60:i , 0])
        y_train.append(train_data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    print("Modelling ... ")
    #model 
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50 , return_sequences = False))
    model.add(Dense(25))
    model.add(Dense(1))
    print("Training")
    #Train
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
    print("Scaling data")
    test_data = scaled_data[training_data_len - 60: , :]
    x_test = []
    y_test = dataset[training_data_len: , :]
    for i in range(60 , len(test_data)):
        x_test.append(test_data[i-60:i , 0])
        
    x_test = np.array(x_test)
    x_test = np.reshape(x_test , (x_test.shape[0] , x_test.shape[1] , 1))
    print("predicting")
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    print(predictions[-1][0])

    for i in range(30):
        predict_next()
    
    plot_me()
