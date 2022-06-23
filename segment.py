# Inspired from
#     https://github.com/facebookresearch/detr
#     https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/DETR_panoptic.ipynb#scrollTo=0Ys8lZhFCwXe

from PIL import Image
import requests
import io
import math
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
torch.set_grad_enabled(False);
import numpy as np

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from panopticapi.utils import rgb2id

import itertools
import seaborn as sns

from copy import deepcopy


# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250)
model.eval();

url = "http://images.cocodataset.org/val2017/000000281759.jpg" # women & humbrellas
url = 'https://i.redd.it/5sbuolzhpd691.jpg' # hawc & coyote
url = 'https://i.redd.it/9eh6phbzw2t31.jpg' # bottles & glasses
im = Image.open(requests.get(url, stream=True).raw)
plt.figure(figsize=(15,15))
plt.imshow(im)
plt.axis('off')

# mean-std normalize the input image (batch-size: 1)
img = transform(im).unsqueeze(0)
out = model(img)

# compute the scores, excluding the "no-object" class (the last one)
scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
# threshold the confidence
keep = scores > 0.85

# Plot all the remaining masks
ncols = 5
fig, axs = plt.subplots(ncols=ncols, nrows=math.ceil(keep.sum().item() / ncols), figsize=(18, 10))
for line in axs:
    for a in line:
        a.axis('off')
for i, mask in enumerate(out["pred_masks"][keep]):
    ax = axs[i // ncols, i % ncols]
    ax.imshow(mask, cmap="cividis")
    ax.axis('off')
fig.tight_layout()

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
plt.figure(figsize=(15,15))
plt.imshow(panoptic_seg)
plt.axis('off')

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
v = Visualizer(np.array(im.copy().resize((final_w, final_h)))[:, :, ::-1], meta, scale=1.0)
v._default_font_size = 15
v = v.draw_panoptic_seg_predictions(panoptic_seg, segments_info, area_threshold=0)
plt.figure(figsize=(15,15))
plt.imshow(v.get_image())
plt.axis('off')
plt.show()
