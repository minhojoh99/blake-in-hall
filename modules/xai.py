from sklearn.model_selection import train_test_split
#-*- coding: utf-8 -*-
import pandas as pd
import pickle
from konlpy.tag import Hannanum
import numpy as np
import os
import unicodedata
import re

from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from modules.data_preprocess import file_read
from modules.config import *
# LSTM으로 좋은 전환율의 메인카피 분류하기

# 패키지 준비
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.family'] = 'NanumBarunGothic'
plt.rcParams['font.size'] = 14.5
plt.rcParams['figure.figsize'] = (14,4)

hannanum = Hannanum()   # 한나눔 한글 형태소 분석기
label = 'main'
global tokenizer_main, max_len, loaded_model
with open("XAI/stopwords_temp", "rb") as fp:  # stopwords를 가져옵니다
    stopwords = pickle.load(fp)
stopwords = stopwords + ['!', '!!', '?', '??', '?!', '!?', '이거', '=', '-', 'ㅋ', 'ㅋㅋ', 'ㅋㅋㅋ', 'ㅋㅋㅋㅋ', 'ㅋㅋㅋㅋㅋ', '#']


def clean_data(comment_list):  # 불용어 제거 및 데이터를 준비합니다

    words_list = []
    for comment in comment_list:
        words_comment = hannanum.nouns(str(comment))  # 명사만 추출합니단
        stopped_comment = [c for c in list(set(words_comment)) if
                           not c in stopwords]  # spowords에 포함된 명사들과 'ㅋ'가 포함된 단어들을 제거합니다
        stopped_comment2 = [c2 for c2 in stopped_comment if len(c2) > 1]  # 한 글자의 명사는 제거합니다
        stopped_comment2 = [c3 for c3 in stopped_comment2 if c3.isdigit() == False]  # 숫자 제거
        words_list.append(stopped_comment2)
    return words_list

# 중첩 리스트를 하나의 리스트로 변환해주는 함수
def flatten_list(list_name):
    one_list = []
    for lists in list_name:
        for list in lists:
            one_list.append(list)
    return one_list

# 정수 인코딩 값을 구하는 함수 (lstm 학습에 활용)
def int_encoding(train, test):
    # 단어 집합 생성 (각 단어마다 인덱스 값을 부여함, 인덱스 값이 클수록 단어 빈도가 낮음)
    X_train = np.array(train)[:, 0]
    y_train = np.array(train)[:, 1]
    X_test = np.array(test)[:, 0]
    y_test = np.array(test)[:, 1]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    # print(tokenizer.word_index)
    # 등장 빈도수가 3 미만인 단어들의 분포 확인 / 빈도수가 2 이하인 단어들은 중요하지 않다고 가정했습니다. Threshold 값을 수정하면 이 또한 바꿀 수 있습니다.
    threshold = 3
    total_cnt = len(tokenizer.word_index) # 단어의 수
    rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합
    for key, value in tokenizer.word_counts.items():   # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
        total_freq = total_freq + value
        if(value < threshold):  # 단어의 등장 빈도수가 threshold보다 작으면
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value
    print('단어 집합(vocabulary)의 크기 :',total_cnt)
    print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
    print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
    print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
    # 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
    # 0번 패딩 토큰을 고려하여 + 1
    vocab_size = total_cnt - rare_cnt + 1
    print('단어 집합의 크기 :',vocab_size)
    # keras tokenizer의 인자로 넘겨: 텍스트 시퀸스 -> 숫자 시퀸스 변환
    tokenizer = Tokenizer(vocab_size)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    # 빈 샘플(empty samples) 제거
    # 각 샘플들의 길이를 확인해서 길이가 0인 샘플들의 인덱스 받아오기
    drop_train = [index for index, sentence in enumerate(X_train_seq) if len(sentence) < 1]
    drop_test = [index for index, sentence in enumerate(X_test_seq) if len(sentence) < 1]
    # 빈 샘플들 제거
    X_train_seq2 = np.delete(X_train_seq, drop_train, axis=0)
    y_train2 = np.delete(y_train, drop_train, axis=0)
    X_test_seq2 = np.delete(X_test_seq, drop_test, axis=0)
    y_test2 = np.delete(y_test, drop_test, axis=0)
    return X_train_seq2, X_test_seq2, y_train2, y_test2, vocab_size, tokenizer

# 전체 샘플 중 길이가 max_len 이하인 샘플의 비율이 몇 %인지 확인
def below_threshold_len(max_len, nested_list):
    count = 0
    for sentence in nested_list:
        if(len(sentence) <= max_len):
            count = count + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))


def lstm_train(vocab_size, X_train, X_test, y_train, y_test, label, category):
    embedding_dim = 100  # 임베딩 벡터 차원
    hidden_units = 128  # 은닉 상태 크기

    # 모델 설계
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(LSTM(hidden_units))
    model.add(Dense(1, activation='sigmoid'))

    # 모델 검증
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint(f'best_model_{label}_{category}.h5', monitor='val_acc', mode='max', verbose=1,
                         save_best_only=True)

    # 모델 훈련
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(X_train, y_train, validation_data=[X_test, y_test], epochs=15, callbacks=[es, mc],
                        batch_size=64)

    # 테스트 정확도 측정
    loaded_model = load_model(f'best_model_{label}_{category}.h5')
    #     X_test = X_test.tolist()
    print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

def xai_train(category_name):
    global tokenizer_main, max_len
    files_path = []
    for path, subdirs, files in os.walk(input_excel_path):
        for name in files:
            if 'csv' in name or 'xlsx' in name:
                files_path.append(unicodedata.normalize('NFC', os.path.join(path, name)))

    files_path = list(filter(lambda x: unicodedata.normalize('NFC', category_name) in x, files_path))

    print(files_path)
    total_main = []
    total_sub = []
    total_cvr = []
    main = []
    sub = []
    cvr = []
    for file_path in files_path:  # 모든 csv파일의 데이터를 불러옵니다
        temp_lists = file_read(file_path)
        df = pd.DataFrame(temp_lists).T.drop_duplicates().reset_index(drop=True)
        if temp_lists:
            main.append(df.loc[:, 0].values)
            sub.append(df.loc[:, 1].values)
            cvr.append(df.loc[:, 2].values)
            total_main.extend(df.loc[:, 0].values)
            total_sub.extend(df.loc[:, 1].values)
            total_cvr.extend(df.loc[:, 2].values)


    main_tokenized_list = []
    for m in main:  # 메인카피 데이터 정리
        temp_tokenized = clean_data(m)
        main_tokenized_list.append(temp_tokenized)
    sub_tokenized_list = []
    for m2 in sub:  # 서브카피 데이터 정리
        temp_tokenized = clean_data(m2)
        sub_tokenized_list.append(temp_tokenized)


    # 메인카피 서브카피 중첩 리스트를 하나의 리스트로 변환해줍니다
    main_flatten = flatten_list(main_tokenized_list)  # lstm 학습에 사용될 데이터들 입니다
    sub_flatten = flatten_list(sub_tokenized_list)  # lstm 학습에 사용될 데이터들 입니다

    # count_words(main,'메인카피')    # 메인카피 값들을 그대로 넣어 동시출현 단어 빈도 분석 값을 구합니다
    # count_words(sub,'서브카피')     # 서브카피 값들을 그대로 넣어 동시출현 단어 빈도 분석 값을 구합니다
    print("len(main_flatten):", len(main_flatten))
    print("len(sub_flatten):", len(sub_flatten))
    print("len(total_cvr):", len(total_cvr))

    main_select = []
    sub_select = []
    for main_f, cvr in zip(main_flatten, total_cvr):
        if main_f:
            main_select.append([main_f, cvr])

    for sub_f, cvr in zip(main_flatten, total_cvr):
        if sub_f:
            sub_select.append([sub_f, cvr])

    main_train, main_test = train_test_split(main_select, test_size=0.2)
    sub_train, sub_test = train_test_split(sub_select, test_size=0.2)

    X_main_train_encoded, X_main_test_encoded, Y_main_cvr_train_encoded, Y_main_cvr_test_encoded, vocab_size_main, tokenizer_main = int_encoding(
        main_train, main_test)
    X_sub_train_encoded, X_sub_test_encoded, Y_sub_cvr_train_encoded, Y_sub_cvr_test_encoded, vocab_size_sub, tokenizer_sub = int_encoding(
        sub_train, sub_test)

    # 패딩

    # 댓글 길이 확인 main (그래프)
    print('main 리뷰의 최대 길이 :', max(len(review) for review in X_main_train_encoded))
    print('main 리뷰의 평균 길이 :', sum(map(len, X_main_train_encoded)) / len(X_main_train_encoded))
    print('sub 리뷰의 최대 길이 :', max(len(review) for review in X_sub_train_encoded))
    print('sub 리뷰의 평균 길이 :', sum(map(len, X_sub_train_encoded)) / len(X_sub_train_encoded))

    # max_len이 n일때 샘플 비율 확인 / 위에 표로 확인 후 max_len 수정 가능
    max_len = 5
    below_threshold_len(max_len, X_main_train_encoded)
    below_threshold_len(max_len, X_sub_train_encoded)

    # 모든 샘플 길이를 max_len으로 맞춤
    X_main_train_encoded = pad_sequences(X_main_train_encoded, maxlen=max_len).astype(np.float32)
    X_main_test_encoded = pad_sequences(X_main_test_encoded, maxlen=max_len).astype(np.float32)
    X_sub_train_encoded = pad_sequences(X_sub_train_encoded, maxlen=max_len).astype(np.float32)
    X_sub_test_encoded = pad_sequences(X_sub_test_encoded, maxlen=max_len).astype(np.float32)
    Y_main_cvr_train_encoded = Y_main_cvr_train_encoded.astype(np.float32)
    Y_main_cvr_test_encoded = Y_main_cvr_test_encoded.astype(np.float32)

    # 아래 lstm_train()안에 main 이라고 들어가 있는 부분을 전부 sub로 바꾸면 서브카피의 lstm 모델 생성가능. 주석으로 아래 예시를 넣어두겠습니다.
    lstm_train(vocab_size_main, X_main_train_encoded, X_main_test_encoded, Y_main_cvr_train_encoded,
               Y_main_cvr_test_encoded, label='main', category=category_name)
    # lstm_train(vocab_size_sub,X_sub_train_encoded,X_sub_test_encoded,Y_sub_cvr_train_encoded,Y_cvr_test)

# 메인/서브카피 예측해보기
okt = Okt()
def cvr_predict(new_sentence,tokenizer,loaded_model):
    print(f"new_sentence : {new_sentence}")
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
    new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen=max_len) # 패딩
    score = float(loaded_model.predict(pad_new)) # 예측
    positive = True # 전환율이 잘 나오는 경우
    if(score > 0.5):
        print("{:.2f}% 확률로 전환율이 잘 나오는 리뷰입니다.\n".format(score * 100))    # 전환율이 잘 나올 확률값을 보여줌
    else:
        print("{:.2f}% 확률로 전환율이 잘 안나오는 리뷰입니다.\n".format((1 - score) * 100))   # 전환율이 잘 안나올 확률값을 보여줌
        positive = False    # 전환율이 잘 안나오는 경우
    return positive

# 데이터 프로세싱
def process_text(text):
    text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', text)
    text = okt.morphs(text, stem=True) # 토큰화
    text = [word for word in text if not word in stopwords] # 불용어 제거
    texts_clean = [word for word in text]
    return " ".join(texts_clean)

# LIME을 사용해서 어떤 단어들이 좋은/안좋은 전환율에 영향을 주는지 확인합니다
from lime.lime_text import LimeTextExplainer
class_names=['negative','positive']
explainer= LimeTextExplainer(class_names=class_names)

def pred(sentence):
    global tokenizer_main, max_len, loaded_model
    processed=[]
    for i in sentence:
        processed.append(process_text(i))
    encoded2 = tokenizer_main.texts_to_sequences(processed) # 정수 인코딩   #여기서 tokenizer_main을 tokenizer_sub로 바꾸면 서브카피에 적용 가능합니다
    pad_new2 = pad_sequences(encoded2, maxlen=max_len) # 패딩
    score2 = loaded_model.predict(pad_new2) # 예측
    returnable = []
    for i2 in score2:
        temp = i2[0]
        returnable.append(np.array([1-temp,temp]))
    return np.array(returnable)

def xai_inference(category_name):
    os.makedirs(f'static/xai/{category_name}', exist_ok=True)
    global tokenizer_main, max_len, loaded_model
    # 위에서 저장한 모델을 불러 옵니다. 한번 모델을 만들고 나면 위의 lstm_train라인을 주석처리하고 이 부분만 사용해서 만들어놓은 모델을 불러올 수 있습니다.
    loaded_model = load_model(f'best_model_{label}_{category_name}.h5')

    files_path = []
    for path, subdirs, files in os.walk(input_excel_path):
        for name in files:
            if 'csv' in name or 'xlsx' in name:
                files_path.append(unicodedata.normalize('NFC', os.path.join(path, name)))
    files_path = list(filter(lambda x: unicodedata.normalize('NFC', category_name) in x, files_path))
    # 테스트 할 데이터를 불러오는 부분입니다

    total_main = []
    for file_path in files_path:  # 모든 csv파일의 데이터를 불러옵니다
        temp_lists = file_read(file_path)
        df = pd.DataFrame(temp_lists).T.drop_duplicates().reset_index(drop=True)
        if temp_lists:
            total_main.extend(df.loc[:, 0].values)

    webscraped_comments = list(
        set(total_main))  # 메인카피를 테스트 해봅니다. 서브 카피를 테스트 하려면 df_test[]안에 있는 column 이름을 수정하면 됩니다.
    try:
        webscraped_comments.remove('nan')
    except:
        print('no nan value')
    positive_comments = []
    negative_comments = []
    webscraped_comments = np.array(webscraped_comments)
    webscraped_comments = webscraped_comments[~pd.isna(webscraped_comments)]
    webscraped_comments = webscraped_comments[webscraped_comments!='nan']
    print(f"webscraped_comments : {webscraped_comments}")
    for comm_i, comm in enumerate(webscraped_comments):
        if comm_i < 10:  # 처음 10개의 메인 카피만 테스트 해봅니다
            print(comm)  # 메인카피 데이터 확인
            bool_positive = cvr_predict(comm, tokenizer_main,
                                        loaded_model)  # 서브 카피를 테스트 하려면 cvr_predict()안에 tokenizer_main을 tokenizer_sub로 변경해주면 됩니다
            if bool_positive == True:
                positive_comments.append(comm)
            else:
                negative_comments.append(comm)
    print("positive_comments", positive_comments)  # 전환율이 잘 나오는 메인카피(또는 서브카피) 값들을 보여줍니다
    print("negative_comments", negative_comments)  # 전환율이 잘 나오는 메인카피(또는 서브카피) 값들을 보여줍니다
    print("len(positive_comments)", len(positive_comments))  # 전환율이 잘 나온다 예측된 데이터 수를 보여줍니다
    print("len(negative_comments)", len(negative_comments))  # 전환율이 잘 안나온다 예측된 데이터 수를 보여줍니다

    for comm_i, comm_text in enumerate(webscraped_comments[:10]):
        #     explainer.explain_instance(comm_text,pred).show_in_notebook(text=True)
        exp = explainer.explain_instance(comm_text, pred)
        fig = exp.as_pyplot_figure()

        fig.savefig(f'static/xai/{category_name}/lime_report_{comm_i}.jpg')
        plt.close()
    #     exp.save_to_file('lime_{}.html'.format(comm_i))