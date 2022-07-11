import streamlit as st
from src import segmentor, default_url
from PIL import Image
import requests
import sys

st.title('Image segmentation using DETR - DEtection TRansformer')

if sys.platform != 'darwin':
    st.write(
        "<font color='red'>"
        'This web app was developped on local computer with 16 GB of RAM. '
        'The present host has a lower memory limit preventing it from running. '
        'Below is a static snapshot of the app result instead.'
        '</font>',
        unsafe_allow_html=True,
    )
    st.image(Image.open('pages/0.jpg'), caption='Input')
    st.image(Image.open('pages/1.jpg'), caption='Heatmaps')
    st.image(Image.open('pages/2.jpg'), caption='Result')

else:
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
