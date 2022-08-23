import pymysql
import pandas as pd
import unicodedata
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
from torchvision import transforms
from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')
global efficientnet

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))])

def file_read(filepath):
    if filepath[-3:] == 'csv':
        try:
            df = pd.read_csv(filepath)
        except Exception:
            try:
                df = pd.read_csv(filepath, encoding='UTF-8')
            except Exception:
                df = pd.read_csv(filepath, encoding='cp949')
    else:
        try:
            df = pd.read_excel(filepath)
        except:
            return
    return df


def create_table(curs):
    mk_table = '''create table if not exists db_blake(id int(11) NOT NULL AUTO_INCREMENT PRIMARY KEY,
                  상품분류 varchar(255), 
                  카테고리 varchar(255),
                  아이템 varchar(255), 
                  기장 varchar(255),
                  색상 varchar(255),
                  세부색상 varchar(255),
                  제품링크 varchar(255),
                  이미지경로 varchar(255))'''
    curs.execute(mk_table)

def load_data():
    bags = []
    shoes = []
    dresses = []
    for path, subdirs, files in os.walk(unicodedata.normalize('NFC', '가방')):
        for file in files:
            if 'csv' in file:
                #             print((os.path.join(path, file)))
                bags.append(file_read(os.path.join(path, file)))
    bagdf = pd.concat(bags)

    for path, subdirs, files in os.walk(unicodedata.normalize('NFC', '신발')):
        for file in files:
            if 'csv' in file:
                #             print((os.path.join(path, file)))
                shoes.append(file_read(os.path.join(path, file)))

    shoes_df = pd.concat(shoes)

    for path, subdirs, files in os.walk(unicodedata.normalize('NFC', '드레스')):
        for file in files:
            if 'csv' in file:
                #             print((os.path.join(path, file)))
                dresses.append(file_read(os.path.join(path, file)))

    dresses_df = pd.concat(dresses)

    dresses_df = dresses_df.drop('소매기장', axis=1)

    total_df = pd.concat([bagdf, shoes_df, dresses_df]).reset_index(drop=True)

    total_df.이미지경로 = total_df.이미지경로.map(lambda x: '/'.join(x.split('\\')[2:]))
    return total_df


def db_connect_and_getdata():
    con = pymysql.connect(host='localhost', user='root', password='djrakswkdwk1',
                           db='blake2', charset='utf8') # 한글처리 (charset = 'utf8')

    curs = con.cursor(pymysql.cursors.DictCursor)
    create_table(curs)
    curs.execute('select * from db_blake2')
    result = curs.fetchall()
    result_df = pd.DataFrame(result)

    if not len(result_df):
        sql = 'insert into db_blake2 (상품분류, 카테고리, 아이템, 기장, 색상, 세부색상, 제품링크, 이미지경로) values(%s, %s, %s, %s, %s, %s, %s, %s)'
        for idx in range(len(result_df)):
            curs.execute(sql, tuple(result_df.values[idx]))
        con.commit()

    big_categories = result_df.상품분류.unique()
    mid_categories = result_df.카테고리.unique()
    items = result_df.아이템.unique()
    return result_df, big_categories, mid_categories, items

## 카테고리에 맞는 엑셀들 선별 및 전처리
##

from modules.config import *
Threshold = 770




if __name__ == '__main__':
    data_preprocess()