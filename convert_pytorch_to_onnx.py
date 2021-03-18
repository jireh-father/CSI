import torch

from skin_recognizer import SkinRecognizer

# 모델 경로
model_path = './logs/skin_total_resnet18_imagenet_sup_CSI_linear_bak/100.model'

# gpu 사용X
use_cuda = False

# score threshold값
score_thres = 0.860

n_classes = 3
is_multi_class = True

# 피부 모델의 모듈 로딩
sr = SkinRecognizer(model_path, use_cuda=use_cuda, score_thres=score_thres, is_multi_class=is_multi_class,
                    n_classes=n_classes)

map_location = 'cpu'

# checkpoint_dict = torch.load(model_path, map_location=map_location)
# pretrained_dict = checkpoint_dict['state_dict']

# model.load_state_dict(pretrained_dict)
# model = .module
model = sr.model.to('cpu')
# print(model)

model.eval()
batch_size = 1
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True).to('cpu')
torch_out = model(x)

# 모델 변환
torch.onnx.export(model,  # 실행될 모델
                  x,  # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  # "/home/irelin/source/font_recognition/model/font_classification/fr_full_lbatch_epoch_17666_c4079.onnx",  # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  "multi_class_skin.onnx",  # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,  # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  # opset_version=11,          # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적하시 상수폴딩을 사용할지의 여부
                  input_names=['input'],  # 모델의 입력값을 가리키는 이름
                  output_names=['output'],  # 모델의 출력값을 가리키는 이름
                  # )
                  # dynamic_axes={'input': {2: 'height', 3: 'width'}}
                  )
