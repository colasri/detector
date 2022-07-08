[![Codacy Badge](https://app.codacy.com/project/badge/Grade/d07dcb3a023c406880e15652d4b2f256)](https://www.codacy.com/gl/colasri/detector/dashboard?utm_source=gitlab.com&amp;utm_medium=referral&amp;utm_content=colasri/detector&amp;utm_campaign=Badge_Grade)

# Detect

Object detection running on a webpage

## Run flask

The flask version: `app.py`.

```shell
flask run
```

For debug mode (only local, do not run on a public facing server, unsafe):

```shell
export FLASK_ENV=development
flask run
```

## Run Streamlit

The streamlit version: `sapp.py`.

```shell
streamlit run sapp.py
```

Setting up environment (for detection, segmentation, web apps, jupyter):

```shell
# conda deactivate
# conda env remove -n stream
conda create -n stream python=3 -y
conda activate stream
pip install -r requirements.txt
pip install git+https://github.com/cocodataset/panopticapi.git git+https://github.com/facebookresearch/detectron2.git
```

## Docker

```shell
time docker build -t ddock .
docker run -p 8501:8501 ddock
docker ps
docker stop $CID
```
