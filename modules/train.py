from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from modules.data_preprocess import data_preprocess


def train(category_name):
    ## 모델 피팅
    ##
    df, vectors_메인카피, vectors_서브카피, vectors_이미지 = data_preprocess(category_name)

    input1 = Input(shape=(100,), name='input1')
    input2 = Input(shape=(100,), name='input2')
    input3 = Input(shape=(100,), name='input3')

    x = tf.keras.layers.concatenate([input1, input2, input3])

    x = Dense(350, activation='relu')(x)
    x = Dense(250, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(30, activation='relu')(x)
    main_output1 = Dense(1, activation='linear', name='main_output1')(x)
    main_output2 = Dense(1, activation='linear', name='main_output2')(x)
    main_output3 = Dense(1, activation='linear', name='main_output3')(x)

    model = Model(inputs=[input1, input2, input3], outputs=[main_output1, main_output2, main_output3])

    model.compile(loss=['mse', 'mse', 'mse'], optimizer='adam', metrics=['mae'])

    model.fit([np.array(vectors_메인카피), np.array(vectors_서브카피), np.array(vectors_이미지)],
              [np.array(df['총 전환율_scaled']), np.array(df['CTR_scaled']), np.array(df['CPC_scaled'])], epochs=150,
              batch_size=5, verbose=2)

    model.save('model_' + str(category_name) + '.h5')

if __name__ == '__main__':
    train()