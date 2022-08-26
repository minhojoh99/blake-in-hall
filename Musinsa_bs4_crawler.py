# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
import dload
from googletrans import Translator
import os
import csv

search_word = input("검색어를 입력하세요: ")
product_number = int(input("검색할 제품 수를 입력하세요: "))

page_range = int(int(product_number) / 90)  # 입력한 제품 수에 따라서 검색 페이지 수를 구합니다
if page_range == 0:
    page_range = 1
else:
    page_range += 1
translator = Translator()   # 사진 이름을 한국어로 저장하면 rgb값을 구할 때 imread에서 에러가 발생해서 검색어를 영어로 변역해서 사진 이름에 사용합니다
search_word_eng = translator.translate(search_word, src='ko',dest='en').text
print("영어 검색어:",search_word_eng)
HEADERS = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36"} # User Agent를 지정해줍니다

# 웹 색상을 구하기 위한 함수
def palette_perc(k_cluster):
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)
    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_)
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i]/n_pixels, 2)
    perc = dict(sorted(perc.items()))
    step = 0
    for idx, centers in enumerate(k_cluster.cluster_centers_):
        palette[:, step:int(step + perc[idx]*width+1), :] = centers
        step += int(perc[idx]*width+1)
    return palette, perc, k_cluster.cluster_centers_

# 제품들의 url을 가져오는 부분
def product_links(page_range):  # 검색 페이지 수를 받습니다
    all_links = []
    for pg in range(int(page_range)): # 검색 결과의 각 페이지마다 접속합니다
        url = "https://www.musinsa.com/search/musinsa/goods?q={}&list_kind=small&sortCode=pop&sub_sort=&page={}&display_cnt=0&saleGoods=false&includeSoldOut=false&setupGoods=false&popular=false&category1DepthCode=&category2DepthCodes=&category3DepthCodes=&selectedFilters=&category1DepthName=&category2DepthName=&brandIds=&price=&colorCodes=&contentType=&styleTypes=&includeKeywords=&excludeKeywords=&originalYn=N&tags=&campaignId=&serviceType=&eventType=&type=&season=&measure=&openFilterLayout=N&selectedOrderMeasure=&shoeSizeOption=&groupSale=false&d_cat_cd=&attribute=".format(search_word,pg+1)
        res = requests.get(url,headers=HEADERS) # 무신사 검색 결과 페이지에 접속합니다
        res.raise_for_status()  # 에러 체크
        soup = BeautifulSoup(res.text,"html")
        products_info = soup.find("div",attrs={"class":"n-search-contents"})    # 제품들이 들어있는 부분을 가져옵니다
        products = products_info.find_all('li',attrs={"class":"li_box"})    # 제품들만 가져옵니다
        for product in products:    # 각 제품들의 링크 주소를 가져옵니다
            p_temp = product.find('p',attrs={"class":"list_info"})
            a_tag = p_temp.find('a',href=True)
            link = a_tag['href']
            all_links.append(link)
            if len(all_links) == int(product_number):   # 요청한 제품 수 만큼의 링크 주소를 구했으면 그 이상은 찾지 않습니다
                break
    return all_links

def Musinsa_crawler(all_links):
    type_class = []
    categories = []
    items = []
    length = []
    color = []
    rgb = []
    clothing_class = ['상의','아우터','바지','원피스','스커트','스포츠/용품']
    shoes_class = ['스니커즈','신발']
    bag_class = ['가방','여성 가방']
    colors_class = ['인디고','indigo','블루','blue','레드','red','블랙','black','bk','화이트','white','에크루','ecru','bleach','그린','green',
                    '크림','cream','그레이','grey','gray','아이보리','ivory','카키','khaki','브라운','brown','네이비','navy','베이지','beige',
                    '브릭','brick','핑크','pink','차콜','charcoal','퍼플','purple','민트','mint','오트밀','oatmeal','노랑','옐로우','yellow',
                    '모카','mocha','머스타드','mustard','올리브','olive','실버','silver','골드','gold','하늘','sky','bl','샌드','sand','orange',
                    '오랜지','와인','wine','멜란지','melange','wht','피치','peach','iv','카멜','camel','레몬','lemon','butter','버터']
    for ind,one_link in enumerate(all_links):   # 각 제품의 정보를 가져옵니다
        print("Product number:",ind)
        res2 = requests.get(one_link,headers=HEADERS)   # 전에 구한 각 제품들의 링크에 접속합니다 
        res2.raise_for_status()  # 에러 체크
        soup = BeautifulSoup(res2.text, 'html.parser')
        # 사진
        image = soup.find('div',attrs={'class':'product-img'}).find('img')    # 사진이 들어있는 부분을 가져옵니다 (img 태그)
        img = 'https:' + image['src']   # 사진의 src값을 가져와 링크 주소를 만들어줍니다
        # 이미지 경로를 아래 3줄에서 지정해 주시면 됩니다. if not 부분 두줄은 지정 폴더 경로를 확인 및 없으면 생성해주고, dload부분은 사진을 저장해주는 부분입니다.
        if not os.path.exists(f'images/musinsa/{search_word_eng}'):    # 사진 저장 폴더 경로를 확인합니다
            os.makedirs(f'images/musinsa/{search_word_eng}')    # 사진 저장 폴더가 없으면 생성해줍니다
        dload.save(img,f'images/musinsa/{search_word_eng}/{search_word_eng}_{ind}.jpg')   # 사진을 저장합니다
        # 카테고리
        content = soup.find('div',attrs={'class':'right_contents section_product_summary'})
        all_a_tags = content.find_all('a')
        categories.append(all_a_tags[0].get_text())
        # 상품 분류
        if categories[-1] in clothing_class:    # 상품 카테고리에 따라서 상품을 '의류','신발','가방'으로 분류해줍니다
            type_class.append('의류')
        elif categories[-1] in shoes_class:
            type_class.append('신발')
        elif categories[-1] in bag_class:
            type_class.append('가방')
        # 아이템
        items.append(all_a_tags[1].get_text())
        # 기장
        try:
            size_table = content.find('table',attrs={'id':'size_table'})    # 사이즈 테이블을 가져옵니다
            sizes_temp = size_table.find_all('td',attrs={'class':'goods_size_val'}) # 사이즈 정보들을 모두 가져옵니다
            sizes = sizes_temp[0].get_text()    # 가장 작은 사이즈의 기장을 가져옵니다
        except:
            sizes = '상세페이지 참조'   # 기장 정보가 사이즈 테이블에 없는 경우
        length.append(sizes)
        # 색상
        all_tr = soup.find('div',attrs={'class':'product_info_table'}).find('tbody').find_all('tr')  # 제품의 정보들을 가져옵니다
        for tr_i,each_tr in enumerate(all_tr):
            if '색상' in each_tr.get_text():    # 색상 정보가 있는 부분을 찾습니다
                tr_color = all_tr[tr_i].get_text()  # 색상만 저장합니다
                break
        temp_color = tr_color.strip()[3:]   # '색상' 단어는 제거합니다
        flag1 = 0
        if '상세' not in temp_color:    # 색상 정보가 '상세페이지 참조' 아닌경우 구한 색상을 저장합니다
            color.append(temp_color)
            flag1 = 1   # 색상을 제품 정보 부분에서 바로 구했을 경우 flag1의 값은 1이 됩니다
        else:   # 색상 정보가 '상세페이지 참조'인 경우
            title = content.find('span',attrs={'class':'product_title'}).get_text()
            title_words = title.split() # 제목에서 색상 정보가 있는지 확인합니다
            for word in title_words:    # 제목의 각 단어마다 색상 단어인지 확인합니다
                new_word = ''
                for ch in word:
                    if ch not in ['(',')','[',']']:     # 불용어를 제거합니다
                        new_word += ch
                for col_word in colors_class:
                    if col_word in new_word.lower():    # 전부 소문자로 바꿔줍니다
                        color.append(new_word.lower())  # 색상 단어가 제품명에 있는 경우 저장해줍니다
                        flag1 = 2   # 색상을 제품명에서 찾았을 경우 flag1의 값은 2이 됩니다
                        break
                if flag1 == 2:  # 제품 색상을 제품명에서 찾았을 경우 그 뒤에 나오는 단어들은 건너뜁니다
                    break
            if flag1 != 2:  # 제품 색상 정보가 상세 정보에도 없고 제품명에도 없는 경우
                print("title_words",title_words)    # 위에서 정의한 colors_class에 들어있지 않은 색이 있는지 제품명을 확인
            if flag1 == 0:  # 제품 색상이 사진에 들어있는 경우 색상 정보를 "상세페이지 참조"로 저장합니다
                color.append(temp_color)
        # 웹 색상 코드 (RGB)
        try:
            image = cv.imread(f'images/musinsa/{search_word_eng}/{search_word_eng}_{ind}.jpg')
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = cv.resize(image, (500,300), interpolation = cv.INTER_AREA)
            clt = KMeans(n_clusters=3)  # n-clusters값을 지정하면 이미지 내의 가장 흔한 색상 몇개를 가져올건지 정할 수 있습니다
            clt.fit(image.reshape(-1,3))
            clt_1 = clt.fit(image.reshape(-1, 3))
            palette_val, pal_dict, pal_rgb = palette_perc(clt_1)
            dict_list = list(pal_dict.values())
            pal_rgb_list = pal_rgb.tolist()
            ind_list = [dict_list.index(x) for x in sorted(dict_list, reverse=True)[:2]]    # 불러온 사진에서 가장 많이 등장하는 두개의 색을 가져옵니다
            img_rgb = pal_rgb_list[ind_list[-1]]    # 주로 가장 많이 등장하는 색이 배경색이고 두번째가 제품 색상이기에 구번째 rgb값을 구합니다
        except:
            img_rgb = []
        rgb.append(img_rgb)
    return type_class,categories,items,length,color,rgb

# csv 파일로 저장
def save_csv(type_class,categories,items,length,color,rgb,all_links,search_word_eng,product_number):
    r = zip(type_class,categories,items,length,color,rgb,all_links)
    header = ['상품 분류','카테고리','아이템','기장','색상','세부 색상','제품 링크']
    with open(f'{search_word_eng}_{product_number}.csv', 'w+', newline ='',encoding='UTF-8') as file:
        w = csv.writer(file)
        dw = csv.DictWriter(file,delimiter=',',fieldnames=header)
        dw.writeheader()
        for row in r:
            w.writerow(row)

all_links = product_links(page_range)   # 제품들의 url을 가져옵니다
type_class, categories, items,length, color,rgb = Musinsa_crawler(all_links)    # 구해온 제품들의 링크를 바탕으로 데이터를 크롤링합니다
save_csv(type_class,categories,items,length,color,rgb,all_links,search_word_eng,product_number) # 구한 데이터를 csv파일로 저장해줍니다
