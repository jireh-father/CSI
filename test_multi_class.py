from skin_recognizer import SkinRecognizer
import cv2

# 모델 경로
model_path = './logs/skin_total_resnet18_imagenet_sup_CSI_linear_bak/100.model'

image_file = '/home/ubuntu/source/CSI/20210228_172821_lzjz_EAR.png'

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
print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 피부 여부 확인 함수
class_name, is_skin = sr.is_skin_and_what_class(img)
print(class_name, 'is skin', is_skin)
