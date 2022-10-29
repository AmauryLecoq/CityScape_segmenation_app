
import streamlit as st


st.title("Application de Segmentation d'image CityScapes")

st.write("""Cette application Streamlit utilise un service FastAPI service comme backend.
         Rendez-vous à l'URL suivante `:8000/docs` pour la documentation FastAPI."""
         ) # description and instructions


# Pour lancer les app streamlit:
# streamlit run app.py