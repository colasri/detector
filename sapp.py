import streamlit as st
from detect import detector
from cv2 import imwrite
from PIL import Image
import requests
import numpy as np


st.title('Object detection using DETR - DEtection TRansformer')
url = st.text_input('Enter an image URL', 'https://i.redd.it/9eh6phbzw2t31.jpg')

with st.spinner('Downloading image...'):
    image = np.asarray(Image.open(requests.get(url, stream=True).raw))
    # st.image(image)
with st.spinner('Detecting objects...'):
    result = detector(image)
    st.image(result['image'], caption=f'Tagged version of {url}.')
    st.write(result['counts'].to_frame('counts'))
