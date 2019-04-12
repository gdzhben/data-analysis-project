import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras import regularizers
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


def generate_penalty_range():
    return_value = []
    start = 0.01
    end = 1
    while(start < end):
        return_value.append(start)
        if(start < 10):
            return_value.append(start+start*2)
        else:
            return_value.append(start+3)
        start = start*10
    return np.array(return_value)


def data_acquisition_preprocessing():
    scaler = MinMaxScaler(feature_range=(0, 1))
    boston = load_boston()
    boston_predictor = boston.data
    boston_target = boston.target
    complete_boston = np.column_stack((boston_predictor, boston_target))

    global predictor
    predictor = scaler.fit_transform(np.delete(complete_boston, 0, 1))
    global target
    target = scaler.fit_transform(complete_boston[:, 0].reshape(-1, 1))


def run_relu_mlp_model_standard_split(validation_size: float, no_hidden_layer: int, no_neuron: int, l1: float, l2: float, regularizer: int, epoch: int, bat_size: int):
    info_hidden = no_hidden_layer
    model = Sequential()
    if regularizer == 1:
        regularizer_name = 'L1 regularizer'
        model.add(Dense(12, input_dim=13, use_bias=True,
                        kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l1(l=l1)))
        while no_hidden_layer != 0:
            model.add(Dense(no_neuron, activation='relu', use_bias=True,
                            kernel_regularizer=regularizers.l1(l=l1)))
            no_hidden_layer = no_hidden_layer - 1
        model.add(Dense(1, use_bias=True, activation='linear',
                        kernel_regularizer=regularizers.l1(l=l1)))
    elif regularizer == 2:
        regularizer_name = 'L2 regularizer'
        model.add(Dense(12, input_dim=13, use_bias=True,
                        kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(l=l2)))
        while no_hidden_layer != 0:
            model.add(Dense(no_neuron, activation='relu', use_bias=True,
                            kernel_regularizer=regularizers.l2(l=l2)))
            no_hidden_layer = no_hidden_layer - 1
        model.add(Dense(1, use_bias=True, activation='linear',
                        kernel_regularizer=regularizers.l2(l=l2)))
    elif regularizer == 3:
        regularizer_name = 'L1+L2 regularizer'
        model.add(Dense(12, input_dim=13, use_bias=True,
                        kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
        while no_hidden_layer != 0:
            model.add(Dense(no_neuron, activation='relu', use_bias=True,
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            no_hidden_layer = no_hidden_layer - 1
        model.add(Dense(1, use_bias=True, activation='linear',
                        kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
    model.summary()
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    history = model.fit(predictor, target, epochs=epoch, batch_size=bat_size,
                        verbose=1, validation_split=validation_size)
    if regularizer == 1:
        l_str = "L1:%.2f" % (l1)
    elif regularizer == 2:
        l_str = "L2:%.2f" % (l2)
    else:
        l_str = "L1:%.2f, L2:%.2f" % (l1, l2)
    info = "Model Loss-"+str(info_hidden)+" hidden layers with "+str(no_neuron) + \
        " neurons" + "\nAfter " + \
        str(epoch)+" sepochs "+l_str

    mlp_history.append(history)
    mlp_info.append(info)
    # print(history.history.keys())
    # "Loss"


def kfold_mlp(k: int, no_hidden_layer, no_neuron: int, l1: float, l2: float, regularizer: int, epoch: int):
    cv = KFold(n_splits=k, random_state=42, shuffle=False)
    info_hidden = no_hidden_layer
    cvscores = []
    for train, test in cv.split(predictor, target):
        # create model
        model_kfold = Sequential()
        if regularizer == 1:
            regularizer_name = 'L1 regularizer'
            model_kfold.add(
                Dense(12, input_dim=13, kernel_regularizer=regularizers.l1(l=l1), kernel_initializer='normal', activation='relu'))
            while no_hidden_layer != 0:
                model_kfold.add(Dense(no_neuron, activation='relu',
                                      use_bias=True, kernel_regularizer=regularizers.l1(l=l1)))
                no_hidden_layer = no_hidden_layer - 1
            model_kfold.add(Dense(1, activation='linear',
                                  kernel_regularizer=regularizers.l1(l=l1)))
        elif regularizer == 2:
            regularizer_name = 'L2 regularizer'
            model_kfold.add(
                Dense(12, input_dim=13, kernel_regularizer=regularizers.l2(
                    l=l2), kernel_initializer='normal', activation='relu'))
            while no_hidden_layer != 0:
                model_kfold.add(Dense(no_neuron, activation='relu',
                                      use_bias=True, kernel_regularizer=regularizers.l2(l=l2)))
                no_hidden_layer = no_hidden_layer - 1
            model_kfold.add(Dense(1, activation='linear',
                                  kernel_regularizer=regularizers.l2(l=l2)))
        elif regularizer == 3:
            regularizer_name = 'L1+L2 regularizer'
            model_kfold.add(
                Dense(12, input_dim=13, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2), kernel_initializer='normal', activation='relu'))
            while no_hidden_layer != 0:
                model_kfold.add(Dense(no_neuron, activation='relu',
                                      use_bias=True, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
                no_hidden_layer = no_hidden_layer - 1
            model_kfold.add(Dense(1, activation='linear',
                                  kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))

        model_kfold.summary()
        # Compile model
        model_kfold.compile(loss='mse', optimizer='adam', metrics=['mse'])
        # Fit the model
        model_kfold.fit(predictor[train], target[train],
                        epochs=epoch, batch_size=50,  verbose=1)
        # evaluate the model
        scores = model_kfold.evaluate(predictor[test], target[test], verbose=1)
        print("%s: %.2f" % (model_kfold.metrics_names[1], scores[1]))
        cvscores.append(scores[1])
    return str(info_hidden)+" hidden layers with "+str(no_neuron) + " neurons" + "- After "+str(epoch)+" sepochs, Result from "+regularizer_name+" with "+str(k)+" fold CV for l1:%.2f penalty MSE: %.2f (SD:+/- %.2f)\n" % (l1, np.mean(cvscores), np.std(cvscores))


mlp_history = []
mlp_info = []
data_acquisition_preprocessing()

# kfold_result = ""
# for reg_index in range(0, 3):
#     for i in np.nditer(generate_penalty_range()):
#         kfold_result += kfold_mlp(5, 2, 10, l1=i, l2=i,
#                                   regularizer=reg_index+1, epoch=500)
#     kfold_result += "----------------------------------------------------------\n"

# print(kfold_result)

# for i in np.nditer(generate_penalty_range()):
for reg_index in range(0, 3):
    run_relu_mlp_model_standard_split(validation_size=0.2, no_hidden_layer=2,
                                    no_neuron=10, l1=0.1, l2=0.1, regularizer=reg_index+1, epoch=100, bat_size=50)
for index in range(0,len(mlp_history),3):
    plt.plot(mlp_history[index].history['loss'])
    plt.plot(mlp_history[index].history['val_loss'])
    plt.plot(mlp_history[index+1].history['loss'])
    plt.plot(mlp_history[index+1].history['val_loss'])
    plt.plot(mlp_history[index+2].history['loss'])
    plt.plot(mlp_history[index+2].history['val_loss'])
    plt.title(mlp_info[index])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train-L1', 'Validation-L1','Train-L2', 'Validation-L2','Train-L1+L2', 'Validation-L1+L2'], loc='upper left')
    # plt.savefig(mlp_info[index]+".png")
    # plt.close()
    plt.show()

gc.collect()
