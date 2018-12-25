from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model

import numpy as np
import os

from . import Const


def load_data(normalization):
    """
    DNN 기법으로 데이터를 학습함.

    Parameters
    ----------
    normalization : bool
        데이터를 0과 1사이의 값으로 정규화를 하는지

    Returns
    -------
    (np.array, np.array), (np.array, np.array)
        (x_train, y_train), (x_test, y_test)
    """

    file_path = os.path.join(Const.DATA_DIR, 'UCI_Credit_Card.csv')
    data = np.loadtxt(fname=file_path, delimiter=',')

    if normalization == True:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)

    x = data[:, 0:-1]
    y = data[:, [-1]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return (x_train, y_train), (x_test, y_test)


def load_model_manager(model_name):
    """
    모델을 로드함

    Parameters
    ----------
    model_name : str
        모델의 이름(저장된 파일 이름)

    Returns
    -------
    model
        로드한 모델

    Raises
    -------
    RuntimeError
        Model 을 찾을 수 없을 때
    """

    file_path = os.path.join(Const.MODEL_DIR, model_name + '.hdf5')
    model = load_model(file_path)

    return model


def ensemble_evaluate(model_preds, y_test):
    """
    앙상블 기법으로 모델을 평가함

    Parameters
    ----------
    model_preds : list(model)
        모델의 결과값들을 모아놓음
    y_test : np.array
        test 데이터의 출력

    Returns
    -------
    acc, list(float)
        정확성과 앙상블의 결과값들

    Raises
    -------
    RuntimeError
        Model 을 찾을 수 없을 때
    """

    pred = np.mean(model_preds, axis=0)
    pred_max = np.argmax(pred, axis=1)
    pred_max = np.expand_dims(pred_max, axis=1)
    acc = np.sum(np.equal(pred_max, y_test)) / len(y_test)

    return pred, acc