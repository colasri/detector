from flask import Flask, escape, request, render_template

from detect import detector, get_opencv_img_from_url
from cv2 import imwrite

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():

    url = None
    error = None
    counts = None
    if request.method == 'POST':
        try:
            url = escape(request.form['url'])
            image = get_opencv_img_from_url(url)
            result = detector(image)
            counts = result['counts'].to_frame('counts').to_html(justify='center', classes='mystyle')
            imwrite('./static/out.jpg', result['image'])
        except ValueError:
            error = 'Invalid URL: please enter a valid image URL.'
    print('BBB')

    return render_template('template.html', url=url, error=error, counts=counts)
