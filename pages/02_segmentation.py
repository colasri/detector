import streamlit as st
from segment import segmentor
from PIL import Image
import requests


st.title('Image segmentation using DETR - DEtection TRansformer')
url = st.text_input('Enter an image URL', 'https://i.redd.it/9eh6phbzw2t31.jpg')

with st.spinner('Downloading image...'):
    image = Image.open(requests.get(url, stream=True).raw)
    # st.image(image)
with st.spinner('Segmenting image...'):
    result = segmentor(image)
    for key, fig in result.items():
        print(f'Figure {key}')
        st.write(key)
        st.pyplot(fig)
