#!/bin/bash
# frontent/Dockerfile

FROM python:3.10

COPY requirements.txt app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 80
#EXPOSE 8501

#ENTRYPOINT ["streamlit","run"]
CMD ["streamlit", "run", "app.py", "--server.port", "80"] 
#CMD ["streamlit", "run", "app.py", "--server.port", "8501"] 