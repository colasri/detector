import streamlit as st

st.title('DETR playground - DEtection TRansformer')
st.sidebar.markdown("# Home page")

st.markdown(
'**TLDNR**: Using computer vision to detect objects and segment image.\n\n'
'Detail:\n'
'- Fetch an image from provided URL.\n'
'- Object detection using DETR - DEtection TRansformer (facebook/detr-resnet-50) via huggingface and pytorch.\n'
'- Web page generated with streamlit.\n'
'- **Web page hosted in a container, where???**\n'
'- Source code: my gitlab.'
)
