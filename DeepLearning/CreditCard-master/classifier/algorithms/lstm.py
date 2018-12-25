from keras.layers.core import Activation, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers.recurrent import LSTM
from .. import Const
from .. import Manager

import os


def make_model(model_name, input_shape, batch_size, num_epochs, x_train, y_train):
    """
    DNN 기법으로 데이터를 학습함.

    Parameters
    ----------
    model_name : str
        모델의 이름(저장되는 파일 이름)
    input_shape : shape
        input data 의 shape
    batch_size : int
        batch 크기
    num_epochs : int
        epoch 횟수
    x_train : np.array
        train 데이터의 입력
    y_train : np.array
        train 데이터의 출력

    Returns
    -------
    model, history
        생성된 모델과 학습 과정을 기록한 history
    """

    x_train = x_train.reshape([len(x_train), len(x_train[0]), 1])

    model = Sequential()
    model.name = model_name
    model.add(LSTM(128, input_shape=(len(x_train[0]), 1)))
    model.add(Dense(64))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    if Const.DEBUG_MODE == 1:
        model.summary()

    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"])

    file_path = os.path.join(Const.MODEL_DIR, model_name + '.hdf5')
    checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=0, save_weights_only=True, save_best_only=True,
                                 mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs, verbose=Const.DEBUG_MODE,
                        callbacks=[checkpoint, tensor_board], validation_split=0.2)
    model.save(file_path)

    return model, history


def evaluate(model_name, x_test, y_test):
    """
    DNN 기법으로 평가함

    Parameters
    ----------
    model_name : str
        모델의 이름(저장되는 파일 이름)
    x_test : np.array
        test 데이터의 입력
    y_test : np.array
        test 데이터의 출력

    Returns
    -------
    np.array, acc
        모델이 예측한 값들, 정확도

    Raises
    -------
    RuntimeError
        Model 을 찾을 수 없을 때
    """

    model = Manager.load_model_manager(model_name)
    x_test = x_test.reshape([len(x_test), len(x_test[0]), 1])

    score, acc = model.evaluate(x_test, y_test, verbose=Const.DEBUG_MODE)
    pred = model.predict(x_test)
    if Const.DEBUG_MODE == 1:
        print("Test score: %.3f, accuracy: %.3f" % (score, acc))

    return pred, acc