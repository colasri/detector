import streamlit as st
from src import segmentor, default_url
from PIL import Image
import requests


st.title('Image segmentation using DETR - DEtection TRansformer')
if 'url' in st.session_state:
    url =  st.session_state['url']
else:
    url = default_url
url = st.text_input('Enter an image URL', url)
st.session_state['url'] = url

with st.spinner('Downloading image...'):
    image = Image.open(requests.get(url, stream=True).raw)
    # st.image(image)
with st.spinner('Segmenting image...'):
    result = segmentor(image)
    for key, fig in result.items():
        print(f'Figure {key}')
        st.write(key)
        st.pyplot(fig)