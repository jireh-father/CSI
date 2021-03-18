from skin_recognizer import SkinRecognizer
import cv2
import time

"""
Pytorch 모델 테스트
"""
# 모델 경로
model_path = '/home/irelin/resource/afp/skin_anomaly_detection/pretrained_models/100.model'

image_file = '/home/irelin/resource/afp/skin_anomaly_detection/real_test_images/111.png'

# gpu 사용X
use_cuda = False

# score threshold값
score_thres = 0.860

n_classes = 3
is_multi_class = True

# 피부 모델의 모듈 로딩
sr = SkinRecognizer(model_path, use_cuda=use_cuda, score_thres=score_thres, is_multi_class=is_multi_class,
                    n_classes=n_classes)

img = cv2.imread(image_file, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 속도 정확히 측정하기 위한 테스트(첫 inference는 느려서)
sr.is_skin_and_what_class(img)

test_cnt = 10
# 피부 여부 확인 함수
total_exec_time = 0.
for i in range(test_cnt):
    start = time.time()
    class_name, is_skin, score = sr.is_skin_and_what_class(img)
    total_exec_time += time.time() - start
print("pytorch model inference time", total_exec_time / test_cnt)
print('class', class_name, ', is skin', is_skin)

"""
onnx 모델 테스트
"""
# onnx 모델 경로
model_path = '/home/irelin/resource/afp/skin_anomaly_detection/pretrained_models/multi_class_skin.onnx'

image_file = '/home/irelin/resource/afp/skin_anomaly_detection/real_test_images/111.png'

# gpu 사용X
use_cuda = False

# score threshold값
score_thres = 0.860

n_classes = 3
is_multi_class = True
use_onnx = True
# 피부 모델의 모듈 로딩
sr = SkinRecognizer(model_path, use_cuda=use_cuda, score_thres=score_thres, is_multi_class=is_multi_class,
                    n_classes=n_classes, use_onnx=use_onnx)

img = cv2.imread(image_file, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 속도 정확히 측정하기 위한 테스트(첫 inference는 느려서)
sr.is_skin_and_what_class(img)

# 피부 여부 확인 함수
total_exec_time = 0.
for i in range(test_cnt):
    start = time.time()
    class_name, is_skin, score = sr.is_skin_and_what_class(img)
    total_exec_time += time.time() - start
print("onnx model inference time", total_exec_time / test_cnt)
print('class', class_name, ', is skin', is_skin)
