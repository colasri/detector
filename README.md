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
