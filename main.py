import os
import cv2 as cv
from flask import Flask, render_template, request, send_from_directory
from flask import redirect, url_for
from flask import request, url_for
from flask import render_template
from werkzeug.utils import secure_filename
# from .img_processing import embossing,cartoon,pencilGray,pencilColor,oilPainting,detailEnhance

def create_app():
    app = Flask(__name__, static_url_path='')

    app.secret_key = os.urandom(24)
    app.config['RESULT_FOLDER'] = 'result_videos'
    app.config['UPLOAD_FOLDER'] = 'uploads'

    @app.route('/upload_img/<filename>')
    def upload_img(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    @app.route('/result_img/<filename>')
    def result_img(filename):
        return send_from_directory(app.config['RESULT_FOLDER'], filename)

    @app.route('/img_result', methods=['GET', 'POST'])
    def img_result():
        if request.method == 'POST':
            f = request.files['file']

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            print(file_path)
            f.save(file_path)
            file_name = os.path.basename(file_path)

            # reading the uploaded image
            img = cv.imread(file_path)

            # # processing
            # style = request.form.get('style')
            # if style == 'Embossing' :
            #     #embossing
            #     output = embossing(img)
            #     # Write the result to ./result_images
            #     result_fname = os.path.splitext(file_name)[0] + "_emboss.jpg"
            #     result_path = os.path.join(basepath, 'result_images', secure_filename(result_fname))
            #     fname = os.path.basename(result_path)
            #     cv.imwrite(result_path, output)
            #
            #     return render_template('img_result.html', file_name=file_name, result_file=fname)
        return ''

    @app.route('/analyze_video')
    def analyze_video():
        return render_template('out.html')


    @app.route('/about')
    def about():
        return render_template('about.html')


    @app.route('/img_processing/', methods=['Get'])
    def img_processing():
        return render_template('img_processing.html')

    @app.route('/')
    def index():
        return render_template('index.html')

    return app
# git push check
