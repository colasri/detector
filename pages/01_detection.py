import streamlit as st
from src import detector, default_url
from PIL import Image
import requests
import numpy as np


st.title('Object detection using DETR - DEtection TRansformer')
if 'url' in st.session_state:
    url =  st.session_state['url']
else:
    url = default_url
url = st.text_input('Enter an image URL', url)
st.session_state['url'] = url

with st.spinner('Downloading image...'):
    image = np.asarray(Image.open(requests.get(url, stream=True).raw))
    # st.image(image)
with st.spinner('Detecting objects...'):
    result = detector(image)
    st.image(result['image'], caption=f'Tagged version of {url}.')
    st.write(result['counts'].to_frame('counts'))
