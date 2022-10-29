import streamlit as st
import requests
from PIL import Image
import io
import os
import os.path
#import cv2
import numpy as np
import pathlib

def load_image(image_file):
	img = Image.open(image_file)
	return img

def load_label(label_file, load_img):
    lab = Image.open(label_file)
    load_img = np.array(load_img)
    lab = np.array(lab)
    lab_color = load_img.copy()
    dims = lab.shape
    label_map = {'0':[0,  0,  0],'1':[0,  0,  0],'2':[0,  0,  0],'3':[0,  0,  0],
'4':[0,  0,  0],'5':[111, 74,  0],'6':[81,  0, 81],'7':[128, 64,128],'8':[244, 35,232],
'9':[250,170,160],'10':[230,150,140],'11':[70, 70, 70],'12':[102,102,156],'13':[190,153,153],
'14':[180,165,180],'15':[150,100,100],'16':[150,120, 90],'17':[153,153,153],'18':[153,153,153],
'19':[250,170, 30],'20':[220,220,  0],'21':[107,142, 35],'22':[152,251,152],'23':[70,130,180],
'24':[220, 20, 60],'25':[255,  0,  0],'26':[0,  0,142],'27':[0,  0, 70],'28':[0, 60,100],
'29':[0,  0, 90],'30':[0,  0,110],'31':[0, 80,100],'32':[0,  0,230],'33':[119, 11, 32],
'-1':[0,  0,142]}
    for i in range(dims[0]):
        for j in range(dims[1]):
            lab_color[i, j] = label_map[str(lab[i, j])]
    #lab = cv2.cvtColor(lab,cv2.COLOR_GRAY2RGB)
    return lab_color

st.title("Segmentation d'image CityScapes")



def request_prediction(model_uri, data):
    headers = {"Content-Type": "image/png"}

    img = {'file': data}

    response = requests.post(model_uri, files=img)

    if response.status_code != 200:
                raise Exception("Request failed with status \
                    {}, {}\
                        ".format(response.status_code, 
                        response.text))

    return response

def file_selector(folder_path, target="file"):
    filenames = [f for f in os.listdir(folder_path) if
                 not f[0] == "."]  # get file names from dir excluding hidden files
    if filenames ==[]:
        selected_filename = 0
        result = 0
    else: 
        selected_filename = st.selectbox(f'Select a {target}', filenames)
        abs_path = os.path.join(folder_path, selected_filename)
        if os.path.isdir(abs_path):
            return file_selector(abs_path, target)
        result = os.path.join(folder_path, selected_filename)
    return result, selected_filename


#file_path = "/Users/amaur/Documents/04 Openclassroom/Projet 8/Script/Split_data"
#/Users/amaur/Documents/04 Openclassroom/Projet 8/Script/P8_Data/Labels



def main():
    APP_URI = 'http://fastapi:8000/images'
    #APP_URI = 'http://127.0.0.1:8000/images'

    parent_path = pathlib.Path(__file__).parent.parent.resolve()           
    img_path = os.path.join(parent_path, "tmp_img")
    lab_path = os.path.join(parent_path, "tmp_lab")

    img_p, img = file_selector(img_path)

    if img_p == 0:
        st.write("Merci de télécharger une image")
    else :
        st.write(img)
        st.write(img_p)
        st.write(f'You selected {img[:-16]}')
        load_img = load_image(img_p)
        st.image(load_img,use_column_width='auto',channels='BGR')

    labelnames = [f for f in os.listdir(lab_path) if not f[0] == "."]
    if labelnames == []:
        st.write("Merci de télécharger des fichiers contenant des labels")
    elif os.path.exists(os.path.join(lab_path, img[:-16]+"_gtFine_labelIds.png")):
        lab = os.path.join(lab_path, img[:-16]+"_gtFine_labelIds.png")
        st.write(f'You selected {lab}')
        load_lab = load_label(lab, load_img)
        st.image(load_lab,use_column_width='auto')
    else:
        st.write("Merci de télécharger le fichier de Label correspondant à votre image")

    
    predict_btn = st.button('Prédire')
    if predict_btn:
                
                #im_io = BytesIO()
                data = load_image(img_p)
                with io.BytesIO() as buf:
                    data.save(buf, 'png')
                    image_bytes = buf.getvalue()


                pred = request_prediction(APP_URI, image_bytes)
                segmented_image = Image.open(io.BytesIO(pred.content))
                st.image([segmented_image],use_column_width='auto')


if __name__ == '__main__':
    main()

# Pour lancer les app streamlit:
# streamlit run app.py