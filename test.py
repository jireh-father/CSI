from skin_recognizer import SkinRecognizer
import cv2

model_path = 'last.model'
axis_path = 'axis.pth'
image_file = 'image.jpg'
use_cuda = False
score_thres = 0.86
sr = SkinRecognizer(model_path, axis_path, use_cuda=use_cuda, score_thres=score_thres)

img = cv2.imread(image_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
is_skin = sr.is_skin(img)
print('is skin', is_skin)
