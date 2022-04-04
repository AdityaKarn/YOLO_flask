from flask import Flask, request, jsonify
from PIL import Image
import os , io , sys
from werkzeug.utils import secure_filename
from flask_cors import CORS
from yolo import process
from datetime import datetime
from random import randint
import shutil
import requests
import cv2
import base64


app = Flask(__name__)
CORS(app)
uploads_dir = os.path.join(app.instance_path, 'uploads')
output_dir = os.path.join(app.instance_path, 'output')

image_path = 'image.jpg'

@app.route('/upload/', methods=['GET'])
def upload_image():
    try:
        os.mkdir(uploads_dir)
        os.mkdir(output_dir)
    except:
        pass

    url = request.values.get('url')
    print(url)
    # alt = request.values.get('token')
    # print(alt)
    r = requests.get(url, allow_redirects = True)

    with open(image_path, 'wb') as img:
        img.write(r.content)
    img.close()




    # file = request.files['file']
    # if not file:
    #     return {'error': 'Missing file'}, 400
    
    now = datetime.now()
    filename = now.strftime("%Y%m%d_%H%M%S") + "_" + str(randint(000, 999))
    src_dir= image_path
    dst_dir= os.path.join(uploads_dir, (filename + '.jpg'))
    shutil.copy(src_dir,dst_dir)
    # file.save(os.path.join(uploads_dir, secure_filename(filename + '.jpg')))
    objects_count, objects_confidence = process(uploads_dir, output_dir, filename)
    
    response = {
        'objects_count': objects_count, 
        'objects_confidence': objects_confidence, 
        'filename': filename + '.jpg'
    }
    img = cv2.imread(dst_dir)
    img = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())

    
    final_prediction = {}
    final_prediction['data'] = response
    final_prediction['image'] = str(img_base64)
    # os.remove(os.path.join(uploads_dir, (filename + '.jpg')))
    # os.remove(os.path.join(output_dir, (filename + '.jpg')))

    return jsonify(final_prediction), 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)