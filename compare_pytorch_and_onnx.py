# import onnx
import torch
import numpy as np
import time

batch_size = 1
# onnx_model = onnx.load("super_resolution.onnx")
# onnx.checker.check_model(onnx_model)

import onnxruntime

ort_session = onnxruntime.InferenceSession("multi_class_skin.onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# ONNX 런타임에서 계산된 결과값
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True).to('cpu')

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)
# ort_outs = ort_session.run(None, ort_inputs)

test_cnt = 10
total = 0.
for i in range(test_cnt):
    start = time.time()
    logits = ort_session.run(None, ort_inputs)
    total += time.time() - start
print("onnx model inference time", total / test_cnt)

# ONNX 런타임과 PyTorch에서 연산된 결과값 비교

model_path = './logs/skin_total_resnet18_imagenet_sup_CSI_linear_bak/100.model'
# gpu 사용X
use_cuda = False

# score threshold값
score_thres = 0.860

n_classes = 3
is_multi_class = True
use_onnx = True

# 피부 모델의 모듈 로딩
from skin_recognizer import SkinRecognizer

sr = SkinRecognizer(model_path, use_cuda=use_cuda, score_thres=score_thres, is_multi_class=is_multi_class,
                    n_classes=n_classes, use_onnx=use_onnx)

torch_out = sr.model(x)

total = 0.
for i in range(test_cnt):
    start = time.time()
    torch_out = sr.model(x)
    print(torch_out.shape)
    total += time.time() - start
print("pytorch model inference time", total / test_cnt)
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
