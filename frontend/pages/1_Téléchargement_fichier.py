import streamlit as st
import os.path
import pathlib
from PIL import Image

def upload_img():
    if uploaded_img is None:
        st.session_state["Etat_téléchargement"] = "Merci de télécharger une image"
    else:
        for i in range(len(uploaded_img)):
            parent_path = pathlib.Path(__file__).parent.parent.resolve()           
            save_path = os.path.join(parent_path, "tmp_img")
            complete_name = os.path.join(save_path, uploaded_img[i].name)
            with open(complete_name, "wb") as f:
                f.write(uploaded_img[i].getbuffer())
                f.close()
                st.success(f"le fichier {uploaded_img[i].name} a bien été téléchargé!")

def upload_lab():
    if uploaded_lab is None:
        st.session_state["Etat_téléchargement"] = "Merci de télécharger un fichier contenant les labels"
    else:
        for i in range(len(uploaded_lab)):
            parent_path = pathlib.Path(__file__).parent.parent.resolve()           
            save_path = os.path.join(parent_path, "tmp_lab")
            complete_name = os.path.join(save_path, uploaded_lab[i].name)
            with open(complete_name, "wb") as f:
                f.write(uploaded_lab[i].getbuffer())
                f.close()
                st.success(f"le fichier {uploaded_lab[i].name} a bien été téléchargé!")

st.title("Téléchargement des fichiers")

pages_names = ["Images", "Labels"]

page = st.radio("Choix des fichiers à télécharger", pages_names)

if page == "Images":
    st.subheader("Vous voulez aller télécharger ici les images à segmenter")

    uploaded_img = st.file_uploader("Choisir les images à télécharger",
    type='png', 
    accept_multiple_files=True
    )
    st.button("Téléchargement des images sur le serveur", on_click=upload_img)
else :
    st.subheader("Vous voulez aller télécharger ici les fichiers de labels")

    uploaded_lab = st.file_uploader("Choisir les fichiers de labels à télécharger",
    type='png', 
    accept_multiple_files=True
    )
    st.button("Téléchargement des fichiers de labels sur le serveur", on_click=upload_lab)
