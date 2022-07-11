import streamlit as st
from src import detector, draw_detection, default_url
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

# If the slider changes, update the query param
def update_threshold():
    threshold = st.session_state["threshold"]
    st.experimental_set_query_params(threshold=threshold)

threshold = st.slider(
    "Select detection threshold", 0., 1., value=0.8, key="threshold", on_change=update_threshold
)

with st.spinner('Detecting objects...'):
    detection = detector(image)
with st.spinner('Plotting result...'):
    result = draw_detection(image, *detection, threshold=threshold)
    st.image(result['image'], caption=f'Tagged version of {url}.')
    st.write(result['counts'].to_frame('counts').astype('object'))
