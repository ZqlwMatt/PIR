import torch
import torch.nn as nn
import clip
from PIL import Image
from pathlib import Path
import random, glob, os
from tqdm import tqdm
import configargparse
import numpy as np

parser = configargparse.ArgumentParser()
parser.add_argument("--images_path", type=str)


class CLIPExtractor(nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        self.device = device
        
    def forward(self, image):
        image = Image.fromarray(image)
        down_sample = 8
        image = image.resize((image.width//down_sample, image.height//down_sample))
        
        patch_size = 10
        stride = patch_size//4
        patches = []
        idxes = []
        image_feature = None
        # loop to get all the patches
        h, w, = image.height, image.width
        for x_idx in range((h-patch_size)//stride + 1 + int((h-patch_size)%stride>0)):
            start_x = x_idx * stride
            for y_idx in range((w-patch_size)//stride + 1 + int((w-patch_size)%stride>0)):
                start_y = y_idx * stride
                # add randomness
                (left, upper, right, lower) = (
                    max(start_y-random.randint(0, patch_size//4), 0), 
                    max(start_x-random.randint(0, patch_size//4), 0), 
                    min(start_y+patch_size+random.randint(0, patch_size//4), w),
                    min(start_x+patch_size+random.randint(0, patch_size//4), h)
                    )
                patches.append(self.preprocess(image.crop((left, upper, right, lower))))
                # patches.append(self.preprocess(image[upper:lower, left:right]))
                idxes.append((left, upper, right, lower))
            
        # get clip embedding 
        count = torch.zeros((1, 1, h, w)).to(self.device)
        sum_feature = torch.zeros((1, 512, h, w)).to(self.device)

        with torch.no_grad():
            chunk_size = 8
            for chunk_idx in range(len(patches)//chunk_size + int(len(patches)%chunk_size>0)):
                patch_chunk = torch.stack(patches[chunk_idx*chunk_size : (chunk_idx+1)*chunk_size]).to(self.device)
                patch_chunk_feature = self.model.encode_image(patch_chunk)
                for i in range(chunk_size):
                    patch_idx = chunk_idx*chunk_size + i
                    if patch_idx >= len(idxes): break

                    sum_feature[:, :, idxes[patch_idx][1]:idxes[patch_idx][3], idxes[patch_idx][0]:idxes[patch_idx][2]] += \
                        patch_chunk_feature[i:i+1, :, None, None]
                    count[:, :, idxes[patch_idx][1]:idxes[patch_idx][3], idxes[patch_idx][0]:idxes[patch_idx][2]] += 1

            image_feature = sum_feature / count
        return np.array(image_feature.cpu()).squeeze(0).transpose(1, 2, 0)

if __name__ == '__main__':
    
    args = parser.parse_args()

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

    clip_extractor = CLIPExtractor("ViT-B/16", device=device)

    image_paths = sorted(glob.glob(f'{args.images_path}/*'))
    
    # Command: python CLIP_extractor.py --images_path "./resource/sam_img"
    # print(image_paths)
    for image_path in image_paths:
        image = Image.open(image_path)
        image = np.array(image)
        clip_feature = clip_extractor(image)
        print(clip_feature.shape)
    
    # save_path = args.save_path
    # os.makedirs(save_path, exist_ok=True)

    # for image_path in tqdm(image_paths):

    #     down_sample = 8
    #     image_path = Path(image_path)
    #     image = Image.open(image_path)
    #     image = image.resize((image.width//down_sample, image.height//down_sample))

    #     patch_sizes = [min(image.size)//5, min(image.size)//7, min(image.size)//10]

    #     image_feature = []

    #     # loop over all the scale
    #     for patch_size in patch_sizes:
    #         stride = patch_size//4
    #         patches = []
    #         idxes = []
    #         # loop to get all the patches
    #         for x_idx in range((image.height-patch_size)//stride + 1 + int((image.height-patch_size)%stride>0)):
    #             start_x = x_idx * stride
    #             for y_idx in range((image.width-patch_size)//stride + 1 + int((image.width-patch_size)%stride>0)):
    #                 start_y = y_idx * stride
    #                 # add randomness
    #                 (left, upper, right, lower) = (
    #                     max(start_y-random.randint(0, patch_size//4), 0), 
    #                     max(start_x-random.randint(0, patch_size//4), 0), 
    #                     min(start_y+patch_size+random.randint(0, patch_size//4), image.width),
    #                     min(start_x+patch_size+random.randint(0, patch_size//4), image.height)
    #                     )
    #                 patches.append(preprocess(image.crop((left, upper, right, lower))))
    #                 idxes.append((left, upper, right, lower))
                
    #         # get clip embedding 
    #         count = torch.zeros((1, 1, image.height, image.width)).to(device)
    #         sum_feature = torch.zeros((1, 512, image.height, image.width)).to(device)

    #         with torch.no_grad():
    #             chunk_size = 8
    #             for chunk_idx in range(len(patches)//chunk_size + int(len(patches)%chunk_size>0)):
    #                 patch_chunk = torch.stack(patches[chunk_idx*chunk_size : (chunk_idx+1)*chunk_size]).to(device)
    #                 patch_chunk_feature = model.encode_image(patch_chunk)
    #                 for i in range(chunk_size):
    #                     patch_idx = chunk_idx*chunk_size + i
    #                     if patch_idx >= len(idxes): break

    #                     sum_feature[:, :, idxes[patch_idx][1]:idxes[patch_idx][3], idxes[patch_idx][0]:idxes[patch_idx][2]] += \
    #                         patch_chunk_feature[i:i+1, :, None, None]
    #                     count[:, :, idxes[patch_idx][1]:idxes[patch_idx][3], idxes[patch_idx][0]:idxes[patch_idx][2]] += 1

    #             image_feature.append(sum_feature / count)

    #     image_feature = torch.cat(image_feature).detach().cpu() # [scale, D, height, width]
        
        # save the extracted feature
        # torch.save(image_feature, f'{save_path}/{image_path.stem}.pth')