import cv2
import numpy as np
from transformers import DetrFeatureExtractor, DetrForObjectDetection
import pandas as pd
import torch
import streamlit as st

classes = None
np.random.seed(2)
COLORS = np.random.uniform(0, 255, size=(80, 3))

def populate_class_labels():
    f = open('./coco2017.txt', 'r')
    classes = [line.strip() for line in f.readlines()]
    return classes

def draw_bbox(img, bboxes, labels, confidences, colors=None, write_conf=False):
    """A method to apply a box to the image
    Inspired from github.com/arunponnusamy/cvlib
    Args:
        img: An image in the form of a numPy array
        bbox: An array of bounding boxes
        labels: An array of labels
        colors: An array of colours the length of the number of targets(80)
        write_conf: An option to write the confidences to the image
    """
    global COLORS
    global classes

    if classes is None:
        classes = populate_class_labels()
    if colors is None:
        colors = COLORS

    for label, bbox, confidence in zip(labels, bboxes, confidences):

        classIndex = classes.index(label)
        color = colors[classIndex]

        if write_conf:
            label += f' {confidence:.2f}'

        bbox = bbox.detach().cpu().numpy().astype(int)
        cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color, 2)
        cv2.putText(img, label, (bbox[0],bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

@st.cache(
    hash_funcs={torch.nn.parameter.Parameter: lambda parameter: parameter.data.numpy()},
    allow_output_mutation=True,
    show_spinner=False)
def detector(image):
    # apply object detection
    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    return (feature_extractor, model, outputs)

def draw_detection(image, extractor, model, detection_result, threshold=0.8):
    # keep only predictions of queries with enough confidence (excluding no-object class)
    probas = detection_result.logits.softmax(-1)[0, :, :-1]
    # keep only the 'person' class
    labels = np.array([model.config.id2label[p.argmax().item()] for p in probas])
    keep = (probas.max(-1).values > threshold)
    # keep = (probas.max(-1).values > threshold) & (labels == 'person')
    keep = np.array(keep, dtype='bool')

    # Count all object passing threshold, 'person' or not
    counts = pd.Series([l for l,p in zip(labels, probas.max(-1).values) if p>threshold], dtype='str').value_counts()

    # rescale bounding boxes
    target_sizes = torch.tensor([image.shape[0], image.shape[1]]).unsqueeze(0)
    print(f'Target sizes {target_sizes}')
    postprocessed_outputs = extractor.post_process(detection_result, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes']

    confidences = torch.amax(probas[keep], dim=1)
    out = draw_bbox(image, bboxes_scaled[keep], labels[keep], confidences, colors=None, write_conf=True)

    print(counts)
    return {'image': out, 'counts': counts}


if __name__ == "__main__":
    from PIL import Image
    import requests
    url = 'https://i.imgur.com/XYmCOL5.jpg'
    image =  np.asarray(Image.open(requests.get(url, stream=True).raw))
    detection = detector(image)
    result = draw_detection(image, *detection, threshold=0.5)
    print(result['counts'])

    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,15))
    plt.imshow(result['image'])
    plt.axis('off')
    plt.show()
