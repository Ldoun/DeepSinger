import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#1차 senetence level 분할 모델 학습
#2차 sentence level 분할 모델을 바탕으로 전이학습하듯 phoneme level 분할 학습을 진행함
# output 모델 구조는 동일하나 모델 파일 2개

class alignment_model(nn.Module):
    def __init__(self):
        super().__init__()