import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import gc


def data_acquisition_preprocessing():
    boston = load_boston()
    boston_predictor = boston.data
    boston_target = boston.target
    complete_boston = np.column_stack((boston_predictor, boston_target))

    global predictor
    predictor = np.delete(complete_boston, 0, 1)
    global target 
    target = complete_boston[:,0]

def run_relu_mlp_model_standard_split(validation_size:float,no_hidden_layer,no_neuron:int):
    model = Sequential()
    model.add(Dense(12, input_dim=13, kernel_initializer='normal', activation='relu'))
    while no_hidden_layer != 0:
        model.add(Dense(no_neuron, activation='relu'))
        no_hidden_layer = no_hidden_layer - 1
    model.add(Dense(1, activation='linear'))
    model.summary()

    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

    history = model.fit(predictor,target, epochs=500, batch_size=50,  verbose=1, validation_split=validation_size)

    print(history.history.keys())
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss 80%-w Train Data')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def kfold_mlp(k:int,no_hidden_layer,no_neuron:int):
    cv = KFold(n_splits=k, random_state=42, shuffle=False)
    cvscores = []

    for train, test in cv.split(predictor, target):
    # create model
        model_kfold = Sequential()
        model_kfold.add(Dense(12, input_dim=13, kernel_initializer='normal', activation='relu'))
        while no_hidden_layer != 0:
            model_kfold.add(Dense(no_neuron, activation='relu'))
            no_hidden_layer = no_hidden_layer - 1
        model_kfold.add(Dense(1, activation='linear'))
        model_kfold.summary()
        # Compile model
        model_kfold.compile(loss='mse', optimizer='adam', metrics=['mse'])
        # Fit the model
        model_kfold.fit(predictor[train], target[train], epochs=500, batch_size=50,  verbose=1)
        # evaluate the model
        scores = model_kfold.evaluate(predictor[test], target[test], verbose=1)
        print("%s: %.2f" % (model_kfold.metrics_names[1], scores[1]))
        cvscores.append(scores[1])

    print("%.2f (+/- %.2f)" % (np.mean(cvscores), np.std(cvscores)))    



data_acquisition_preprocessing()
run_relu_mlp_model_standard_split(0.2,2,10)
kfold_mlp(5,2,10)

gc.collect()
