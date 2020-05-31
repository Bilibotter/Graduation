import cv2
import sys
sys.path.append('./LFFDApply')
import predict
import mxnet as mx
from config_farm import configuration_10_320_20L_5scales_v2 as cfg


class LFFDDetector(object):
    def __init__(self, symbol_file_path=None, model_file_path=None):
        ctx = mx.gpu(0)
        self.symbol_file_path = r'./weights/symbol_10_320_20L_5scales_v2_deploy.json'
        self.model_file_path = r'./weights/train_10_320_20L_5scales_v2_iter_1000000.params'
        self.symbol_file_path = symbol_file_path if symbol_file_path else self.symbol_file_path
        self.model_file_path = model_file_path if model_file_path else self.model_file_path
        self.face_predictor = predict.Predict(mxnet=mx,
                                             symbol_file_path=self.symbol_file_path,
                                             model_file_path=self.model_file_path,
                                             ctx=ctx,
                                             receptive_field_list=cfg.param_receptive_field_list,
                                             receptive_field_stride=cfg.param_receptive_field_stride,
                                             bbox_small_list=cfg.param_bbox_small_list,
                                             bbox_large_list=cfg.param_bbox_large_list,
                                             receptive_field_center_start=cfg.param_receptive_field_center_start,
                                             num_output_scales=cfg.param_num_output_scales)

    def get_boxes(self, image):
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_COLOR)
        boxes, infer_time = self.face_predictor.predict(image, resize_scale=1, score_threshold=0.8, top_k=10000, \
                                                                    NMS_threshold=0.4, NMS_flag=True, skip_scale_branch_list=[])

        return boxes

    def draw_in_image(self, path, display=True, save_path=None):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        boxes = self.get_boxes(image)
        for box in boxes:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        if save_path:
            cv2.imwrite(save_path, image)
        if display:
            cv2.imshow('test', image)
            k = cv2.waitKey(0)
            # 按下空格退出
            if k == ord(' '):
                cv2.destroyAllWindows()

    def get_pure_faces(self, path):
        if isinstance(path, str):
            image = cv2.imread(path, cv2.IMREAD_COLOR)
        else:
            image = path
        boxes = self.get_boxes(image)
        faces = []
        for i, box in enumerate(boxes):
            box = tuple(map(int, box))
            face = image[box[1]:box[3], box[0]:box[2], :]
            faces.append(face)

        return faces

    def realtime_detect(self):
        video_capture = cv2.VideoCapture(0)
        try:
            while video_capture.isOpened():
                ret, image = video_capture.read()
                if not ret:
                    print('摄像头有问题')
                    break
                boxes = self.get_boxes(image)
                for box in boxes:
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                cv2.imshow('Online', image)
                k = cv2.waitKey(1)
                if k == ord(' '):
                    break
                if k == ord('s'):
                    cv2.imwrite('realtine.jpg', image)
        finally:
            video_capture.release()
            cv2.destroyAllWindows()


def run_time_test():
    import time
    detector = LFFDDetector()
    cap = cv2.VideoCapture(0)
    cost_times = 0
    delay_times = 200
    test_times = 1000
    multiply = test_times
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        delay_times -= 1
        start = time.time()
        boxes = detector.get_boxes(img)
        cost = time.time() - start
        if delay_times <= 0:
            cost_times += cost
            test_times -= 1
            print(test_times)
        if test_times <= 0:
            avg_time = cost_times / multiply
            print(f'Cost time is {avg_time} second.')
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    # run_time_test()
    # sys.exit(0)
    detector = LFFDDetector()
    detector.realtime_detect()
    sys.exit(0)
    # LFFD的感受野有限，人像不能过大
    path = r'F:/obama1.jpg'
    detector = LFFDDetector()
    detector.draw_in_image(path, save_path=path.replace('.jpg', 'LFFD.jpg'))
    import math
    import matplotlib.pyplot as plt

    dense_face_path = 'F:/1140.jpg'
    detector.draw_in_image(dense_face_path)
    faces = detector.get_pure_faces(dense_face_path)
    nums = len(faces)
    print(f'Got {nums} faces in image.')
    sub_width = math.ceil(math.sqrt(nums))
    for i, face in enumerate(faces):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        plt.subplot(sub_width, sub_width, i + 1)
        plt.axis('off')
        plt.imshow(face)
    plt.show()
    _ = input('按下Enter键关闭')
    plt.close()
