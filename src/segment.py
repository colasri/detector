# Inspired from
#     https://github.com/facebookresearch/detr
#     https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/DETR_panoptic.ipynb#scrollTo=0Ys8lZhFCwXe

USE_DETECTRON2 = True

from PIL import Image
import requests
import io
import math
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
torch.set_grad_enabled(False);
import numpy as np

if USE_DETECTRON2:
    try:
        from detectron2.utils.visualizer import Visualizer
    except ModuleNotFoundError:
        '''Dirty workaround to install detectron2 on streamlit server, since detectron2 requires pytorch
        and it does not work in a single "pip install -r requirements.txt" call. And I can't call
        another "pip install" while setting up the app, this is automated.
        '''
        import subprocess
        import sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/colasri/detectron2.git'])
        from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog

from panopticapi.utils import rgb2id

import itertools
import seaborn as sns

from copy import deepcopy


def segmentor(image, threshold=0.85):
    figures = {}
    figures['input'] = plt.figure(figsize=(15,15))
    plt.imshow(image)
    plt.axis('off')

    model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250)
    model.eval()

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # mean-std normalize the input image (batch-size: 1)
    img = transform(image).unsqueeze(0)
    out = model(img)

    # compute the scores, excluding the "no-object" class (the last one)
    scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
    # threshold the confidence
    keep = scores > threshold

    # Plot all the remaining masks
    ncols = 5
    nrows = max(2, math.ceil(keep.sum().item() / ncols))
    figures['heatmaps'], axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(18, 10))
    for line in axs:
        for a in line:
            a.axis('off')
    for i, mask in enumerate(out["pred_masks"][keep]):
        ax = axs[i // ncols, i % ncols]
        ax.imshow(mask.detach(), cmap="cividis")
        ax.axis('off')
    figures['heatmaps'].tight_layout()


    # the post-processor expects as input the target size of the predictions (which we set here to the image size)
    result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]

    palette = itertools.cycle(sns.color_palette())

    # The segmentation is stored in a special-format png
    panoptic_seg = Image.open(io.BytesIO(result['png_string']))
    panoptic_seg = np.array(panoptic_seg, dtype=np.uint8).copy()
    # We retrieve the ids corresponding to each mask
    panoptic_seg_id = rgb2id(panoptic_seg)

    # Finally we color each mask individually
    panoptic_seg[:, :, :] = 0
    for psid in range(panoptic_seg_id.max() + 1):
        panoptic_seg[panoptic_seg_id == psid] = np.asarray(next(palette)) * 255
    figures['segment 1'] = plt.figure(figsize=(15,15))
    plt.imshow(panoptic_seg)
    plt.axis('off')


    if USE_DETECTRON2:
        # We extract the segments info and the panoptic result from DETR's prediction
        segments_info = deepcopy(result["segments_info"])
        # Panoptic predictions are stored in a special format png
        panoptic_seg = Image.open(io.BytesIO(result['png_string']))
        final_w, final_h = panoptic_seg.size
        # We convert the png into an segment id map
        panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
        panoptic_seg = torch.from_numpy(rgb2id(panoptic_seg))

        # Detectron2 uses a different numbering of coco classes, here we convert the class ids accordingly
        meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
        for sInfo in segments_info:
            c = sInfo["category_id"]
            sInfo["category_id"] = meta.thing_dataset_id_to_contiguous_id[c] if sInfo["isthing"] else meta.stuff_dataset_id_to_contiguous_id[c]

        # Finally we visualize the prediction
        v = Visualizer(np.array(image.copy().resize((final_w, final_h)))[:, :, ::-1], meta, scale=1.0)
        v._default_font_size = 15
        v = v.draw_panoptic_seg(panoptic_seg, segments_info, area_threshold=0)
        figures['segment 2'] = plt.figure(figsize=(15,15))
        plt.imshow(v.get_image())
        plt.axis('off')

    return figures

if __name__ == "__main__":
    # url = 'https://i.imgur.com/XYmCOL5.jpg'
    url = 'https://i.redd.it/9eh6phbzw2t31.jpg'
    # url = 'https://upload.wikimedia.org/wikipedia/commons/4/41/Siberischer_tiger_de_edit02.jpg'
    image =  Image.open(requests.get(url, stream=True).raw)
    result = segmentor(image, threshold=0.85)

    # import cv2
    # import requests
    # import numpy as np

    # # Fetch JPEG data
    # d = requests.get('http://images.cocodataset.org/train2017/000000000086.jpg')

    # # Decode in-memory, compressed JPEG into Numpy array
    # im = cv2.imdecode(np.frombuffer(d.content,np.uint8), cv2.IMREAD_COLOR)

    for key, fig in result.items():
        print(f'Figure {key}')
        fig.show()
    plt.show()
