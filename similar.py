
from __future__ import absolute_import, division, print_function

import click
import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from libs.models import *
from libs.utils import DenseCRF

from glob import glob
from PIL import Image
import os
import argparse
def simi():
    # 명령행 인자 파서 생성
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='C:/cap/deeplab-pytorch/configs/cocostuff164k.yaml', help='설정 파일 경로')
    parser.add_argument('--model-path', type=str, default='C:/cap/min/flask-app/data/models/coco/deeplabv1_resnet101/deeplabv2_resnet101_msc-cocostuff164k-100000.pth', help='모델 파일 경로')
    parser.add_argument('--image-folder', type=str, default='C:/cap/min/flask-app/static/save', help='이미지 폴더 경로')
    parser.add_argument('--crf', action='store_true', help='CRF 적용 여부')
    parser.add_argument('--cpu', action='store_true', help='CPU 모드 사용 여부')

    # 명령행 인자 파싱
    args = parser.parse_args()


    def multi(config_path, model_path, image_folder, cuda, crf):
        """
        Inference from a single image
        """
        def colorize(labelmap):
            # Assign a unique color to each label
            labelmap = labelmap.astype(np.float32) / CONFIG.DATASET.N_CLASSES
            colormap = cm.jet_r(labelmap)[..., :-1] * 255.0
            return np.uint8(colormap)
        # Setup

        # Setup
        CONFIG = OmegaConf.load(config_path)
        device = get_device(cuda)
        torch.set_grad_enabled(False)

        classes = get_classtable(CONFIG)
        postprocessor = setup_postprocessor(CONFIG) if crf else None

        model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        print("Model:", CONFIG.MODEL.NAME)
        
        
        # 
        image_paths = glob(os.path.join(image_folder, '*')) # 이미지 파일의 경로들.
        image_names = [im_path.split('/')[-1].split('.')[0] for im_path in image_paths] # 이미지 파일의 이름들.

        # 저장 폴더
        mask_folder='./static/mask'
        if not os.path.exists(mask_folder):
            os.mkdir(mask_folder)
        
        num_image=len(os.listdir(image_folder))
        num_label=len(os.listdir(mask_folder))
        
        if num_label==num_image:
            print('이미지와 라벨의 개수가 이미 같습니다')
            return
        
        # Inference
        for idx, image_path in enumerate(image_paths):
            if idx < num_label :
                print('pass. ... ', image_path)
                continue
            
            print('process ... ', image_path)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            
            image, raw_image = preprocessing(image, device, CONFIG)
            labelmap = inference(model, image, raw_image, postprocessor)
            
            labelmap=labelmap.astype(np.uint8)
            
            labelmap = colorize(labelmap)

            # save
            save_path = os.path.join(mask_folder, *image_names[idx].split('\\')[1:])
            save_path = f"{save_path}.png"
            
            Image.fromarray(labelmap).save(save_path) 
            
    def get_device(cuda):
        cuda = cuda and torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        if cuda:
            current_device = torch.cuda.current_device()
            print("Device:", torch.cuda.get_device_name(current_device))
        else:
            print("Device: CPU")
        return device


    def get_classtable(CONFIG):
        with open(CONFIG.DATASET.LABELS) as f:
            classes = {}
            for label in f:
                label = label.rstrip().split("\t")
                classes[int(label[0])] = label[1].split(",")[0]
        return classes


    def setup_postprocessor(CONFIG):
        # CRF post-processor
        postprocessor = DenseCRF(
            iter_max=CONFIG.CRF.ITER_MAX,
            pos_xy_std=CONFIG.CRF.POS_XY_STD,
            pos_w=CONFIG.CRF.POS_W,
            bi_xy_std=CONFIG.CRF.BI_XY_STD,
            bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
            bi_w=CONFIG.CRF.BI_W,
        )
        return postprocessor


    def preprocessing(image, device, CONFIG):
        # Resize
        scale = CONFIG.IMAGE.SIZE.TEST / max(image.shape[:2])
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
        raw_image = image.astype(np.uint8)

        # Subtract mean values
        image = image.astype(np.float32)
        image -= np.array(
            [
                float(CONFIG.IMAGE.MEAN.B),
                float(CONFIG.IMAGE.MEAN.G),
                float(CONFIG.IMAGE.MEAN.R),
            ]
        )

        # Convert to torch.Tensor and add "batch" axis
        image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        image = image.to(device)

        return image, raw_image


    def inference(model, image, raw_image=None, postprocessor=None):
        _, _, H, W = image.shape

        # Image -> Probability map
        logits = model(image)
        logits = F.interpolate(logits, size=(
            H, W), mode="bilinear", align_corners=False)
        probs = F.softmax(logits, dim=1)[0]
        probs = probs.cpu().numpy()

        # Refine the prob map with CRF
        if postprocessor and raw_image is not None:
            probs = postprocessor(raw_image, probs)

        labelmap = np.argmax(probs, axis=0)

        return labelmap

    multi(args.config_path, args.model_path, args.image_folder, cuda=args.cpu, crf=args.crf)
    imgs = []

    file = "./static/mask"
    img_list = os.listdir(file)

    for i in range(2):
        path = os.path.join(file,img_list[i])
        imgs.append(path)
        
    hists = []
    for img in imgs:
        #BGR 이미지를 HSV 이미지로 변환
        read_img = cv2.imread(img)
        hsv = cv2.cvtColor(read_img, cv2.COLOR_BGR2HSV)
        #히스토그램 연산(파라미터 순서 : 이미지, 채널, Mask, 크기, 범위)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        #정규화(파라미터 순서 : 정규화 전 데이터, 정규화 후 데이터, 시작 범위, 끝 범위, 정규화 알고리즘)
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        #hists 리스트에 저장
        hists.append(hist)

    #1번째 이미지를 원본으로 지정
    query = hists[0]

    #비교 알고리즘의 이름들을 리스트에 저장
    methods = ['BHATTACHARYYA']
    #1. cv2.HISTCMP_CORREL : 상관관계
    #1: 완전 일치, -1: 완전 불일치, 0: 무관계 -> 빠르지만 부정확

    #3. cv2.HISTCMP_BHATTACHARYYA : 바타차야 거리
    #0: 완전 일치, 1: 완전 불일치 -> 느리지만 가장 정확
    score = 0
    for index, name in enumerate(methods):
        
        for i, histogram in enumerate(hists):
            ret = cv2.compareHist(query, histogram, index) 
            if(i+1 == 2):    
                score += ret
                
    return round(score*100,2)
    

