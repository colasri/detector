FROM python:3.10
RUN pip install --upgrade pip
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
# RUN pip install git+https://github.com/colasri/detectron2.git
EXPOSE 8501
CMD ["streamlit", "run", "home_page.py"]
