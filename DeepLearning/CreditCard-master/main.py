# https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset

from classifier import *
from classifier import algorithms
import matplotlib.pyplot as plt

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = Manager.load_data(normalization=False)
    (nor_x_train, nor_y_train), (nor_x_test, nor_y_test) = Manager.load_data(normalization=True)
    input_shape = x_train[0].shape

    algorithms.nn.make_model('nn', input_shape, batch_size=32, num_epochs=15, x_train=x_train, y_train=y_train)
    algorithms.lstm.make_model('lstm', input_shape, batch_size=32, num_epochs=15, x_train=x_train, y_train=y_train)
    algorithms.dnn.make_model('dnn', input_shape, batch_size=32, num_epochs=15, x_train=x_train, y_train=y_train)
    algorithms.nn.make_model('nor_nn', input_shape, batch_size=32, num_epochs=15, x_train=nor_x_train, y_train=nor_y_train)
    algorithms.lstm.make_model('nor_lstm', input_shape, batch_size=32, num_epochs=15, x_train=nor_x_train, y_train=nor_y_train)
    algorithms.dnn.make_model('nor_dnn', input_shape, batch_size=32, num_epochs=15, x_train=nor_x_train, y_train=nor_y_train)
    
    nn_pred, nn_acc = algorithms.nn.evaluate('nn', x_test, y_test)
    dnn_pred, dnn_acc = algorithms.nn.evaluate('dnn', x_test, y_test)
    lstm_pred, lstm_acc = algorithms.lstm.evaluate('lstm', x_test, y_test)
    nor_nn_pred, nor_nn_acc = algorithms.nn.evaluate('nor_nn', nor_x_test, nor_y_test)
    nor_dnn_pred, nor_dnn_acc = algorithms.nn.evaluate('nor_dnn', nor_x_test, nor_y_test)
    nor_lstm_pred, nor_lstm_acc = algorithms.lstm.evaluate('nor_lstm', nor_x_test, nor_y_test)
    
    print('nn acc : {0}'.format(nn_acc))
    print('dnn acc : {0}'.format(dnn_acc))
    print('lstm acc : {0}'.format(lstm_acc))
    print('nor_nn acc : {0}'.format(nor_nn_acc))
    print('nor_dnn acc : {0}'.format(nor_dnn_acc))
    print('nor_lstm acc : {0}'.format(nor_lstm_acc))

    # plt.plot(nn_pred, label='nn_pred')
    plt.plot(dnn_pred, label='dnn_pred')
    # plt.plot(lstm_pred, label='lstm_pred')
    # plt.plot(nor_nn_pred, label='nor_nn_pred')
    plt.plot(nor_dnn_pred, label='nor_dnn_pred')
    # plt.plot(nor_lstm_pred, label='nor_lstm_pred')
    plt.legend()
    plt.show()

    ensemble_models = [nor_nn_pred, nor_dnn_pred, nor_lstm_pred]

    ens_pred, end_acc = Manager.ensemble_evaluate(ensemble_models, y_test)
    print('ens acc : {0}'.format(end_acc))
