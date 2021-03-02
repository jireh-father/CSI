from skin_recognizer import SkinRecognizer
import cv2
import pickle


# 모델 경로
model_path = 'last.model'
# 통계정보 경로
axis_path = 'axis.pth'


image_file = 'image.jpg'

weight_sim_path = 'feats_1_resize_fix_0.54_weight_sim.pth'
weight_shi_path = 'feats_1_resize_fix_0.54_weight_shi.pth'
weight_shi = pickle.load(open(weight_sim_path, "rb"))
weight_sim = pickle.load(open(weight_shi_path, "rb"))

# gpu 사용X
use_cuda = False

# score threshold값
score_thres = 0.860

# 피부 모델의 모듈 로딩
sr = SkinRecognizer(model_path, axis_path, use_cuda=use_cuda, score_thres=score_thres, weight_shi=weight_shi, weight_sim=weight_sim)

img = cv2.imread(image_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 피부 여부 확인 함수
is_skin = sr.is_skin(img)
print('is skin', is_skin)

from PIL import Image
import glob
import os

dir1 = "/home/irelin/source/android-demo-app/HelloWorldApp/app/src/main/assets/skin"
dir2 = "/home/irelin/source/android-demo-app/HelloWorldApp/app/src/main/assets/noskin"
for image_file in glob.glob(os.path.join(dir1, "*")):
    Image.open(image_file).convert("RGB").resize((224, 224)).save(image_file + ".jpg", format="JPEG")

for image_file in glob.glob(os.path.join(dir2, "*")):
    Image.open(image_file).convert("RGB").resize((224, 224)).save(image_file + ".jpg", format="JPEG")
