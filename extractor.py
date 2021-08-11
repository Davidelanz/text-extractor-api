import numpy as np
from modelplace_api import Device
from openvino_text_spotting_detector import \
    InferenceModel  # import the AI Model
from paddleocr import PaddleOCR
from PIL import Image


class TextExtractor:
    def __init__(self, languages: "list[str]"):
        self.languages = languages
        self.ocr = {}
        for lang in self.languages:
            print(f'Initializing "{lang}" PaddleOCR')
            self.ocr[lang] = PaddleOCR(use_angle_cls=True, lang=lang)

    @staticmethod
    def run_detection(img: Image):
        img = img.convert("RGB")
        model = InferenceModel()  # Initialize a model
        model.model_load(Device.cpu)  # Loading a model weights
        boxes = model.process_sample(img)  # Processing an image
        return boxes

    @staticmethod
    def boxes2crops(img: Image, boxes) -> "list[Image]":
        crops = []
        for box in boxes:
            a, b, c, d = box.points
            w, h = img.size
            x_pad = int(.01 * w)
            y_pad = int(.001 * h)
            crop = img.crop((a.x - x_pad, a.y - y_pad,
                            c.x + x_pad, c.y + y_pad))
            crops.append(crop)
        return crops

    def run_recognition(self, img: Image, lang="latin") -> tuple:
        result = self.ocr[lang].ocr(
            np.array(img), det=False, rec=True, cls=True)
        return result

    def full_pipeline(self, img: Image, lang="latin") -> "list[tuple]":
        img = img.convert("RGB")
        boxes = self.run_detection(img)
        crops = self.boxes2crops(img, boxes)
        texts = [self.run_recognition(c, lang)[0] for c in crops]
        texts = [
            {"word": word, "score": float(score)}
            for word, score in texts
            if ~np.isnan(score) and len(word) > 0
        ]
        return texts
