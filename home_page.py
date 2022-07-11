import streamlit as st

st.set_page_config(layout='wide', page_title='DETR', page_icon=':eyes:')

st.title('DETR playground - DEtection TRansformer')
st.sidebar.markdown("# Home page")

st.markdown(
'**TLDNR**: Using computer vision to detect objects and segment image.\n\n'
'Detail:\n'
'- Fetch an image from provided URL.\n'
"- Computer vision using using[ Facebook's DETR - DEtection TRansformer](https://github.com/facebookresearch/detr) (detr-resnet-50 for object detection, detr_resnet101_panoptic for image segmentation) via [HuggingFace](https://huggingface.co) and [PyTorch](https://pytorch.org).\n"
'- Web page generated with [Streamlit](https://streamlit.io) and hosted on Streamlit Cloud.\n'
'- Optionally packaged with [Docker](https://www.docker.com), self-hosted.\n'
'- Source code: [my GitLab](https://gitlab.com/colasri/detector).'
)
