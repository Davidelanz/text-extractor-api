import json

import numpy as np
from cv2 import cv2
from flasgger import Swagger
from flask import Flask, jsonify, request, send_file
from PIL import Image

from extractor import TextExtractor

model = TextExtractor(languages=[
    'en', 'french', 'german', 'latin',
])

app = Flask(__name__)
app.config['SWAGGER'] = {
    'title': 'Text Extractor API',
    'openapi': "3.0.0",
    "specs_route": "/"
}
swagger = Swagger(app, template_file='text-extractor.yaml')

BATCH_SIZE = 1
CHECK_INTERVAL = 0.1


def is_json(myjson):
    try:
        json.loads(myjson)
    except TypeError as e:
        return False
    return True


@app.route('/health')
def health():
    return "ok"


@app.route('/extract', methods=['POST'])
def run_python():

    print(f"Handling request {request}")

    lang = request.form.get('lang')

    preview_mode = request.form.get('preview')
    preview_mode = True if preview_mode is not None and preview_mode == 'true' else False

    filestr = request.files['input'].read()
    npimg = np.fromstring(filestr, np.uint8)
    pil_image = Image.fromarray(cv2.imdecode(
        npimg, cv2.IMREAD_UNCHANGED)).convert("RGB")

    if pil_image is None:
        print("[Returning 400] Invalid file")
        return jsonify({'message': 'invalid file'}), 400

    print(f"Input image shape: {pil_image.size}")
    width, height = pil_image.size

    if height * width >= 6250000:
        print("[Returning 400] Too big size image")
        return jsonify({'message': 'too big size image'}), 400

    ret = {}
    ret["text"] = model.full_pipeline(pil_image, lang)

    if preview_mode:
        dump = json.dumps(ret)
        if len(dump) > 1000:
            print("[Returning 200] Preview JSON dump (1000 chars limit)")
            return jsonify(dump[0:1000]), 200

    print("[Returning 200] Full JSON dump")
    return jsonify(ret), 200


if __name__ == "__main__":
    app.run(debug=False, port=80, host='0.0.0.0')
