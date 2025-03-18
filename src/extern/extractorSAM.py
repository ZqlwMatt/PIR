import os
from PIL import Image
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import torch.nn.functional as F

from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)

if __name__ == '__main__':
    
    parser = ArgumentParser(description="SAM feature extracting params")
    
    # parser.add_argument("--image_root", default='./data/360_v2/garden/', type=str)
    parser.add_argument("--sam_checkpoint_path", default="./sam_ckpt/sam_vit_b_01ec64.pth", type=str)
    parser.add_argument("--sam_arch", default="vit_b", type=str)

    args = parser.parse_args()
    
    print("Initializing SAM...")
    model_type = args.sam_arch
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)
    
    IMAGE_DIR = './resource/sam_img'
    assert os.path.exists(IMAGE_DIR) and "Please specify a valid image root"
    OUTPUT_DIR = './resource/features'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    import numpy as np
    import cv2
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    
    print("Extracting features...")
    for path in tqdm(os.listdir(IMAGE_DIR)):
        name = path.split('.')[0]
        img = cv2.imread(os.path.join(IMAGE_DIR, path))
        # B, C, H, W = img.shape
        img = cv2.resize(img,dsize=(1024,1024),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
        print(img.shape)
        predictor.set_image(img)
        features = predictor.features
        print(f"name = {name}, features_shape = {features.shape}")
        torch.save(features, os.path.join(OUTPUT_DIR, name+'.pt'))
        
        dimension = 256
        features = F.interpolate(features, size=(512, 512)) # [1, c, h, w]
        features = features.permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()
        component = pca.fit_transform(features.reshape(-1, dimension))
        component = component.reshape([512, 512, 3])
        component = ((component - component.min()) / (component.max() - component.min())).astype(np.float32) # normalize to [0,1]
        component *= 255.
        component = component.astype(np.uint8)
        
        cv2.imwrite(os.path.join(OUTPUT_DIR, name+'_sam.png'), component.squeeze())