from flask import Flask, escape, request, render_template

from detect import detector
from cv2 import imwrite
from PIL import Image
import requests
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():

    url = None
    error = None
    counts = None
    if request.method == 'POST':
        try:
            url = escape(request.form['url'])
            image =  np.asarray(Image.open(requests.get(url, stream=True).raw))
            result = detector(image)
            counts = result['counts'].to_frame('counts').to_html(justify='center', classes='mystyle')
            imwrite('./static/out.jpg', result['image'])
        except ValueError:
            error = 'Invalid URL: please enter a valid image URL.'
    print('BBB')

    return render_template('template.html', url=url, error=error, counts=counts)
