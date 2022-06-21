import urllib.request
import cv2
import numpy as np
from transformers import DetrFeatureExtractor, DetrForObjectDetection
import pandas as pd
import torch

classes = None
np.random.seed(1)
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

def detector(image, threshold=0.8):
    # apply object detection
    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # keep only predictions of queries with enough confidence (excluding no-object class)
    probas = outputs.logits.softmax(-1)[0, :, :-1]
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
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes']

    confidences = torch.amax(probas[keep], dim=1)
    out = draw_bbox(image, bboxes_scaled[keep], labels[keep], confidences, colors=None, write_conf=True)
    # release resources
    cv2.destroyAllWindows()

    print(counts)
    return {'image': out, 'counts': counts}

# Next couple functions credit: https://stackoverflow.com/a/55360543/6352677
def get_opencv_img_from_buffer(buffer, flags=cv2.IMREAD_COLOR):
    bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
    return cv2.imdecode(bytes_as_np_array, flags)

def get_opencv_img_from_url(url, flags=cv2.IMREAD_COLOR):
    req = urllib.request.Request(url)
    return get_opencv_img_from_buffer(urllib.request.urlopen(req), flags)


if __name__ == "__main__":
    url = 'https://i.imgur.com/XYmCOL5.jpg'
    image = get_opencv_img_from_url(url)
    result = detector(image, threshold=0.5)
    print(result['counts'])
    cv2.imwrite('output.jpg',  result['image'])
