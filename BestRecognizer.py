import dlib
import keras
import pickle
import cupy as cp
from seetaface.api import *
current_path = os.path.abspath(__file__)
ABS_PATH_UTILS = os.path.dirname(current_path)


class BaseFaceRecognizer(object):
    def __init__(self, model):
        self.size = 256
        self.tolerance = 0.2
        self.threshold = 0.62
        if model == 'precise':
            from DSFD import DSFD_Detector
            self.face_detector = DSFD_Detector()
        else:
            from LFFD import LFFDDetector
            self.face_detector = LFFDDetector()
        model_save_path = r"weights\shape_predictor_5_face_landmarks.dat"
        self.landmarks_5_detector = dlib.shape_predictor(model_save_path)
        init_mask = FACERECOGNITION
        self.face_recognizer = SeetaFace(init_mask)

    def verification(self, unknown_feature, features):
        max_error_times = len(features) * self.tolerance
        current_error_times = 0
        for feature in features:
            similarity = self.get_similarity_(feature, unknown_feature)
            current_error_times += similarity < self.threshold
            if current_error_times > max_error_times:
                return False
        return True

    def get_similarity(self, img1, img2):
        feature1 = self.get_features(img1)[0]
        feature2 = self.get_features(img2)[0]
        return self.get_similarity_(feature1, feature2)

    def get_similarity_(self, feature1, feature2):
        dot = cp.sum(cp.multiply(feature1, feature2))
        norm = cp.linalg.norm(feature1) * cp.linalg.norm(feature2)
        similarity = dot / norm
        return similarity

    def get_features(self, img):
        crop_faces = self.get_crop_faces(img)
        return self.get_features_(crop_faces)

    def get_features_(self, crop_faces):
        features = []
        for crop_face in crop_faces:
            feature = self.face_recognizer.ExtractCroppedFace(crop_face)
            features.append(cp.array(feature))
        return features

    def get_crop_faces(self, img):
        boxes_set = self.get_boxes_set(img)
        return self.get_crop_faces_(img, boxes_set)

    def get_crop_faces_(self, img, boxes_set):
        crop_faces = []
        for boxes in boxes_set:
            rect = dlib.rectangle(boxes[0], boxes[1], boxes[2], boxes[3])
            full_object_detection = self.landmarks_5_detector(img, rect)
            crop_faces.append(dlib.get_face_chip(img, full_object_detection))
        return crop_faces

    def get_boxes_set(self, img):
        return self.face_detector.get_boxes(img)


class DataSets(object):
    def __init__(self, model, save_path=None):
        self.name_2_features = {}
        self.utils = BaseFaceRecognizer(model=model)
        default_name = 'OnlineDataSet' if model == 'fast' else 'OfflineDataset'
        default_save_path = os.path.join(ABS_PATH_UTILS, default_name)
        self.save_path = save_path if save_path else default_save_path
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        else:
            self.load()

    def __call__(self, *args, **kwargs):
        return self.name_2_features

    def get_person(self, name):
        return self.name_2_features[name]

    def build_with_feature(self, features):
        pass

    def build(self, images_path):
        names = os.listdir(images_path)
        for name in names:
            features = []
            images_dir = os.path.join(images_path, name)
            faces = os.listdir(images_dir)
            for face in faces:
                img = os.path.join(images_dir, face)
                feature = self.utils.get_features(img)[0]
                features.append(feature)
            save_path = os.path.join(self.save_path, f'{name}.pkl')
            with open(save_path, 'wb')as f:
                pickle.dump(features, f)
            print(f'Dump {name} successful.')

    def load(self):
        pkls = os.listdir(self.save_path)
        for pkl in pkls:
            path = os.path.join(self.save_path, pkl)
            with open(path, 'rb')as f:
                features = pickle.load(f)
            name = pkl.split('.')[0]
            self.name_2_features[name] = features


class OnlineRecognizer(object):
    def __init__(self):
        self.recognizer = BaseFaceRecognizer(model='fast')
        path = os.path.join(ABS_PATH_UTILS, r'weights\fas.h5')
        # 活体检测
        self.fas = keras.models.load_model(path)
        self.dataset = DataSets(model='fast')
        self.dataset.load()

    def get_living_score(self, crop_face):
        temp = crop_face.copy()
        temp = (cv2.resize(temp, (224, 224)) - 127.5) / 127.5
        score = self.fas.predict(np.array([temp]))
        return score[0][0]

    def get_name(self, crop_face):
        feature = self.recognizer.get_features_([crop_face])[0]
        name_2_features = self.dataset()
        for name in name_2_features:
            if self.recognizer.verification(feature, name_2_features[name]):
                return name
        return 'unknown'

    def get_draw_info(self, img):
        infos = []
        boxes_set = self.recognizer.get_boxes_set(img)
        crop_faces = self.recognizer.get_crop_faces_(img, boxes_set)
        for boxes, crop_face in zip(boxes_set, crop_faces):
            name = self.get_name(crop_face)
            living_score = self.get_living_score(crop_face)
            infos.append((boxes, name, living_score))
        return infos


class OfflineRecognizer(object):
    def __init__(self):
        self.recognizer = BaseFaceRecognizer(model='precise')
        self.dataset = DataSets(model='precise')
        self.dataset.load()

    def get_name(self, feature):
        name_2_features = self.dataset()
        for name in name_2_features:
            if self.recognizer.verification(feature, name_2_features[name]):
                return name
        return 'unknown'

    def get_present_and_absent(self, photo):
        name_2_features = self.dataset()
        all_students = set(name_2_features.keys())
        present_students = set()
        features = self.recognizer.get_features(photo)
        for feature in features:
            name = self.get_name(feature)
            present_students.add(name)
        if 'unknown' in present_students:
            all_students.add('unknown')
        absent_students = all_students - present_students
        return present_students, absent_students


def real_time_test(online_recognizer):
    import cv2
    online_recognizer.dataset.load()
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_DUPLEX
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        for boxes, name, living_score in online_recognizer.get_draw_info(image):
            color = [0, 255, 0] if living_score > 0.6 and name != 'unknown' else [0, 0, 255]
            cv2.rectangle(image, (boxes[0], boxes[1]), (boxes[2], boxes[3]),
                          color=color, thickness=2)
            cv2.putText(image, name, (int(boxes[0]) + 2, int(boxes[1] - 6)), font,
                        1.0, color, 1)
            cv2.putText(image, str(living_score)[:4], (int(boxes[0]) + 2, int(boxes[1] - 26)), font,
                        1.0, color, 1)
        cv2.imshow('realtime', image)
        k = cv2.waitKey(1)
        if k == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
        elif k == ord(' '):
            cv2.imwrite('realtimes.jpg', image)


def offline_test():
    import cv2
    photo = cv2.imread('F:/classmates.jpg')
    offline_recognizer = OfflineRecognizer()
    features = offline_recognizer.recognizer.get_features(photo)
    for i, feature in enumerate(features):
        offline_recognizer.dataset.name_2_features[i] = [feature]
    present_students, absent_students = offline_recognizer.get_present_and_absent(photo)
    print('present students', present_students)
    print('absent students', absent_students)
    for name1 in offline_recognizer.dataset.name_2_features:
        for name2 in offline_recognizer.dataset.name_2_features:
            f1 = offline_recognizer.dataset.name_2_features[name1][0]
            f2 = offline_recognizer.dataset.name_2_features[name2][0]
            if offline_recognizer.recognizer.get_similarity_(f1, f2) >= 0.62:
                print(f'Input id {name1}, predict id {name2}')
                break


if __name__ == '__main__':
    offline_test()
    online_recognizer = OnlineRecognizer()
    real_time_test(online_recognizer)
