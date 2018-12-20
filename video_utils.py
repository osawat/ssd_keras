import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
import tensorflow as tf

from ssd import SSD300
from ssd_utils import BBoxUtility

import json


class VideoPlayer:
    def __init__(self, source_path=None, verbose=0):
        """
        Video Player with Object Detection

        Parameters
        ----------
        source_path: video file path
          if nothing specified, 0 is set and WebCam is selected if available.
        verbose: to be used for karas prediction (0,1)

        Configuration
        ----------
        configuration should be written in video_config.json file
          regarding weights file path and classes.

        How to use
        ----------
        To interrupt video, push escape key.
        ex1 video)
        video = VideoPlayer(source_pathpath) # for video
        video.play(10,10,1000)  # view from 10 frame to 1000 frame step 10
        ex2 webcam)
        video = VideoPlayer()  # for webcam
        video.play()
        """
        self.config = json.load(open('video_config.json', 'r'))
        self.source_path = 0 if source_path is None else source_path
        self.cap = cv2.VideoCapture(self.source_path)
        self.verbose = verbose
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.voc_classes = self.config["classes"]
        self.num_classes = len(self.voc_classes) + 1
        self.input_shape = (300, 300, 3)
        self.model = SSD300(self.input_shape, num_classes=self.num_classes)
        self.model.load_weights(self.config["ssd_weights_path"], by_name=True)
        self.bbox_util = BBoxUtility(self.num_classes)

        self.image = None
        self.frame = None
        self.label = None
        self.label_name = None
        self.score = None
        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None

        print('width: {}, height: {}'.format(self.width, self.height))
        print('Total frames: {}'.format(self.frame_count))
        print('fps: {}'.format(self.fps))

    def _detect(self):

        img = cv2.resize(self.image, (300, 300))
        pred = self.model.predict(np.array([img]), batch_size=1,
                                  verbose=self.verbose)
        results = self.bbox_util.detection_out(pred)

        # hook method
        self._before_detection()

        det_label = results[0][:, 0]
        det_conf = results[0][:, 1]
        det_xmin = results[0][:, 2]
        det_ymin = results[0][:, 3]
        det_xmax = results[0][:, 4]
        det_ymax = results[0][:, 5]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        # colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        font = cv2.FONT_HERSHEY_DUPLEX
        font_size = 0.4

        if self.frame_count != -1:
            display_frm = 'Frame: {}'.format(self.frame)
            cv2.putText(self.image, display_frm, (5, 13),
                        font, font_size, (255, 255, 255), 1)

        for i in range(top_conf.shape[0]):
            self.xmin = int(round(top_xmin[i] * self.image.shape[1]))
            self.ymin = int(round(top_ymin[i] * self.image.shape[0]))
            self.xmax = int(round(top_xmax[i] * self.image.shape[1]))
            self.ymax = int(round(top_ymax[i] * self.image.shape[0]))
            self.score = top_conf[i]
            self.label = int(top_label_indices[i])
            self.label_name = self.voc_classes[self.label - 1]
            display_txt = '{:0.2f}, {}'.format(self.score, self.label_name)

            cv2.rectangle(self.image, (self.xmin, self.ymin),
                          (self.xmax, self.ymax), (0, 255, 0), 2)
            cv2.putText(self.image, display_txt, (self.xmin+3, self.ymin+12),
                        font, font_size, (0, 0, 255), 1)

            # hook method
            self._during_detection()

        # hook method
        self._after_detection()

    def play(self, start=0, step=100, last=None):
        """
        Play Video

        Parameters
        ----------
        these parameters is available only for video
        start: int
            the frame to start you want
        step: int
            interval of time between showing
        last: int
            the frame to last you want
        """

        if self.frame_count <= 0:
            while True:
                ret, self.image = self.cap.read()
                if not ret:
                    print("Video Reader does not work well.")
                    break
                self._detect()
                cv2.imshow('img', self.image)
                if cv2.waitKey(1) == 27:  # escape
                    print("Breaked.")
                    break
        else:
            if last is None:
                last = self.frame_count

            for self.frame in range(start, last, step):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame)
                ret, self.image = self.cap.read()
                if not ret:
                    print("Video Reader does not work well.")
                    break
                self._detect()
                cv2.imshow('img', self.image)
                if cv2.waitKey(1) == 27:  # escape
                    print("Breaked.")
                    break
        self.cap.release()
        cv2.destroyAllWindows()
        print("Finished.")

    def _before_detection(self):
        pass

    def _during_detection(self):
        pass

    def _after_detection(self):
        pass


class VideoRecorder(VideoPlayer):
    """
    Record Video with Object Detection
    How to use
    ----------
    To interrupt video, push escape key.
    ex1 video)
    recorder = VideoRecorder(source_path, verbose)
    recorder.record(target_path, start, step, last)
    ex2 WebCam)
    recorder = VideoRecorder(0, verbose)
    recorder.record(target_path)
    """

    def record(self, target_path='out.mp4', start=0, step=100, last=None):
        """
        Record Video with Object Detection
        Parameters
        ----------
        target_path: str
            path to be record video
        these parameters is available only for video
        start: int
            the frame to start you want
        step: int
            interval of time between showing
        last: int
            the frame to last you want
        """

        if self.fps is 0:
            self.fps = 10
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(target_path, fourcc, self.fps,
                              (self.width, self.height))

        if self.frame_count <= 0:  # WebCam
            while True:
                ret, self.image = self.cap.read()
                if not ret:
                    print("Video Reader does not work well.")
                    break
                self._detect()
                cv2.imshow('img', self.image)
                out.write(self.image)
                if cv2.waitKey(1) == 27:  # escape
                    print("Breaked.")
                    break
        else:  # Video
            if last is None:
                last = self.frame_count

            for self.frame in range(start, last, step):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame)
                ret, self.image = self.cap.read()
                if not ret:
                    print("Video Reader does not work well.")
                    break
                self._detect()
                out.write(self.image)
                if cv2.waitKey(1) == 27:  # escape
                    print("Breaked.")
                    break
        self.cap.release()
        cv2.destroyAllWindows()
        print("Finished.")
