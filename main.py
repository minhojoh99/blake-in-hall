from multiprocessing import freeze_support
import os

import numpy as np
from flask import render_template, request, jsonify, redirect, url_for
from run import server_run, app
from glob import glob
global category_name
from modules.data_preprocess import db_connect_and_getdata
from modules.inference import inference_ready
from sklearn.metrics.pairwise import euclidean_distances
import torch
from PIL import Image
from torchvision import transforms
import shutil
import json
device = 'cpu'

global big_category_name, mid_category_name, item_name, color_name, result_df, big_categories, mid_categories, items, colors
global efficientnet

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))])

os.makedirs('static/Downloads', exist_ok=True)
@app.route('/', methods=['GET', 'POST'])
def main():
	return render_template('main.html')

@app.route('/db_connect', methods=['GET', 'POST'])
def db_connect():
	global result_df, big_categories, mid_categories, items, colors
	result_df, big_categories, mid_categories, items = db_connect_and_getdata()
	colors = result_df['색상'].value_counts()[result_df['색상'].value_counts() > 25].keys()
	return render_template('main.html', big_categories=big_categories, mid_categories=mid_categories, items=items, colors=colors)

@app.route('/model_ready', methods=['GET', 'POST'])
def model_ready():
	global efficientnet
	efficientnet = inference_ready()
	return redirect(url_for('main'))

@app.route('/inference1', methods=['GET', 'POST'])
def inference1():
	global result_df, efficientnet
	try:
		for file in glob(os.path.join("static/Downloads", '*')):
			os.remove(file)
	except Exception as e:
		print(e)

	image_path = f"static/Downloads/{request.files['file'].filename}"
	request.files['file'].save(image_path)
	upload_img = f"Downloads/{request.files['file'].filename}"
	pil_image = Image.open(image_path).convert("RGB")
	batch = torch.stack([trans(pil_image)]).to(device)
	batch_result = efficientnet(batch).detach().numpy()

	select_df = result_df[(result_df['상품분류'] == big_category_name) & (result_df['카테고리'] == mid_category_name) & (result_df['아이템'] == item_name)].reset_index(drop=True)
	images = []
	index = []
	if len(select_df['이미지경로']) == 0:
		return view()
	for idx, path in enumerate(select_df['이미지경로']):
		try:
			images.append(trans(Image.open(path).convert("RGB")))
			index.append(idx)
		except:
			continue
	try:
		select_df = select_df.iloc[index].reset_index(drop=True)
		print(select_df)
		candidates_batch = torch.stack(images).to(device)
		candidates_result = efficientnet(candidates_batch).detach().numpy()

		ucsims = [euclidean_distances(batch_result[0].reshape(1, -1), x.reshape(1, -1)) for x in candidates_result]
		arglists = np.array(ucsims)[:, 0, 0].argsort()

		select_images = select_df.loc[arglists[:10], '이미지경로']
		for select_image in select_images:
			shutil.copy(select_image, f"static/Downloads/{os.path.basename(select_image)}")

		data = [(1/x, y) for x, y in zip(np.array(ucsims)[:, 0, 0][arglists[:10]], select_images)]

	except Exception as e:
		print("Exception " + e)

	return view(data, select_df.iloc[arglists[:10]].reset_index(drop=True), upload_img)

@app.route('/catUpdate', methods=['GET', 'POST'])
def catUpdate():
	global big_categories, mid_categories, items
	# big_category_name = request.form['category1'].replace('\"', '')
	# mid_category_name = request.form['category2'].replace('\"', '')
	# item_name = request.form['category3'].replace('\"', '')
	return render_template('main.html', big_categories=big_categories, mid_categories=mid_categories, items=items, colors=colors)
	# return redirect(url_for('main'))

@app.route('/catUpdate1', methods=['GET', 'POST'])
def catUpdate1():
	global big_category_name, big_categories, mid_categories, items, result_df
	data = request.get_json()
	print(f"data : {data}, data['text'] : {data['text']}")
	big_category_name = data['text'].replace('\"', '').replace(' ', '')
	big_categories = [big_category_name]
	print(f"big_category_name : {big_category_name}")
	result_df = result_df[result_df['상품분류'] == big_category_name].reset_index(drop=True)
	mid_categories = list(result_df['카테고리'].unique())
	print(f"mid_categories : {mid_categories}")
	return jsonify(result="success", mid_categories=mid_categories)

@app.route('/catUpdate2', methods=['GET', 'POST'])
def catUpdate2():
	global big_category_name, mid_category_name, big_categories, mid_categories, items, result_df
	data = request.get_json()
	print(f"data : {data}, data['text'] : {data['text']}")
	mid_category_name = data['text'].replace('\"', '').replace(' ', '')
	mid_categories = [mid_category_name]
	print(f"mid_category_name : {mid_category_name}")
	result_df = result_df[result_df['카테고리'] == mid_category_name].reset_index(drop=True)
	items = list(result_df['아이템'].unique())
	print(f"items : {items}")
	return jsonify(result="success", items=items)


@app.route('/catUpdate3', methods=['GET', 'POST'])
def catUpdate3():
	global big_category_name, mid_category_name, item_name, big_categories, mid_categories, colors, items, result_df
	data = request.get_json()
	print(f"data : {data}, data['text'] : {data['text']}")
	item_name = data['text'].replace('\"', '').replace(' ', '')
	items = [item_name]
	print(f"item_name : {item_name}")
	result_df = result_df[result_df['아이템'] == item_name].reset_index(drop=True)
	colors = list(result_df['색상'].unique())
	print(f"colors : {colors}")
	return jsonify(result="success", colors=colors)

@app.route('/catUpdate4', methods=['GET', 'POST'])
def catUpdate4():
	global big_category_name, mid_category_name, item_name, color_name, big_categories, mid_categories, colors, items, result_df
	data = request.get_json()
	print(f"data : {data}, data['text'] : {data['text']}")
	color_name = data['text'].replace('\"', '').replace(' ', '')
	colors = [color_name]
	print(f"color_name : {color_name}")
	result_df = result_df[result_df['색상'] == color_name].reset_index(drop=True)
	return jsonify(result="success")

@app.route('/view', methods=['GET', 'POST'])
def view(datas, select_df, upload_img):
	data_list = []

	for idx, (sim, obj) in enumerate(datas):  # 튜플 안의 데이터를 하나씩 조회해서
		data_dic = {  # 딕셔너리 형태로
			# 요소들을 하나씩 넣음
			'name': f'Downloads/{os.path.basename(obj)}',
			'similarity' : f"{sim:.4f}",
			'color': list(map(round, eval(select_df.loc[idx, "세부색상"]))),
			'link' : select_df.loc[idx, "제품링크"],
			'path' : select_df.loc[idx, '이미지경로']
		}
		print(data_dic)
		data_list.append(data_dic)  # 완성된 딕셔너리를 list에 넣음

	return render_template('view.html', data_list=data_list, upload_img=upload_img)  # html을 렌더하며 DB에서 받아온 값들을 넘김

if __name__ == "__main__":
	freeze_support()
	# 플라스크 서버를 실행시킨다.
	server_run()