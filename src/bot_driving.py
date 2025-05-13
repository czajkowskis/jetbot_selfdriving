import cv2
import onnxruntime as rt

from pathlib import Path
import yaml
import numpy as np

from PUTDriver import PUTDriver, gstreamer_pipeline


class AI:
    def __init__(self, config: dict):
        self.path = config['model']['path']

        self.sess = rt.InferenceSession(self.path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
 
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize the image
        img = cv2.resize(img, (224, 224))
        # Convert to float32 and normalize
        img = img.astype(np.float32) / 255.0
        # Normalize using the mean and std of ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        # Transpose to match the input shape (1, 3, 224, 224)
        img = np.transpose(img, (2, 0, 1))
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, detections: np.ndarray) -> np.ndarray:
        # Apply any necessary postprocessing steps here
        outputs = np.clip(detections, -1.0, 1.0)
        return outputs

    def predict(self, img: np.ndarray) -> np.ndarray:
        inputs = self.preprocess(img)

        assert inputs.dtype == np.float32
        assert inputs.shape == (1, 3, 224, 224)
        
        detections = self.sess.run([self.output_name], {self.input_name: inputs})[0]
        outputs = self.postprocess(detections)

        assert outputs.dtype == np.float32
        assert outputs.shape == (2,)
        assert outputs.max() <= 1.0
        assert outputs.min() >= -1.0

        return outputs


def main():
    with open("config.yml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    driver = PUTDriver(config=config)
    ai = AI(config=config)

    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0, display_width=224, display_height=224), cv2.CAP_GSTREAMER)

    # model warm-up
    ret, image = video_capture.read()
    if not ret:
        print('No camera')
        return
    
    _ = ai.predict(image)

    input('Robot is ready to ride. Press Enter to start...')

    forward, left = 0.0, 0.0
    while True:
        print(f'Forward: {forward:.4f}\tLeft: {left:.4f}')
        driver.update(forward, left)

        ret, image = video_capture.read()
        if not ret:
            print('No camera')
            break
        forward, left = ai.predict(image)


if __name__ == '__main__':
    main()
