import os
import cv2
from tqdm import tqdm
from glob import glob
from albumentations import CenterCrop,RandomRotate90, Sharpen,AdvancedBlur,RandomGamma ,VerticalFlip,GaussNoise,CLAHE,RandomBrightnessContrast
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import os
import cv2
from tqdm import tqdm
from glob import glob
import numpy as np
import random

def load_data(path):
    images = sorted(glob(os.path.join(path,"train/*")))
    masks = sorted(glob(os.path.join(path, "train_annot/*")))

    return images,masks

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

import random
def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)

def add_noise(xb):
    sigma = random.uniform(5, 10)
    h, w = xb.shape
    gauss = np.random.normal(0, sigma, (h, w))
    gauss = gauss.reshape(h, w)
    noisy = xb + gauss
    noisy = np.clip(noisy, 0, 255).astype('uint8')
    return noisy

def elastic_transform(image, label, alpha=10, sigma=2, alpha_affine=2, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    # pts1: 仿射变换前的点(3个点)
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size,
                        center_square[1] - square_size],
                       center_square - square_size])
    # pts2: 仿射变换后的点
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                       size=pts1.shape).astype(np.float32)
    # 仿射变换矩阵
    M = cv2.getAffineTransform(pts1, pts2)
    # 对image进行仿射变换.
    imageB = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    imageA = cv2.warpAffine(label, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # generate random displacement fields
    # random_state.rand(*shape)会产生一个和shape一样打的服从[0,1]均匀分布的矩阵
    # *2-1是为了将分布平移到[-1, 1]的区间, alpha是控制变形强度的变形因子
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    # generate meshgrid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # x+dx,y+dy
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    # bilinear interpolation
    xb = map_coordinates(imageB, indices, order=1, mode='constant').reshape(shape)
    yb = map_coordinates(imageA, indices, order=1, mode='constant').reshape(shape)
    return xb, yb

def augment_data(images,masks,save_path,augment=True):
    H= 384
    W= 384

    for x,y in tqdm(zip(images,masks), total=len(images)):
        name = x.split("/")[-1].split(".")
        image_name= name[0]
        image_extn = name[1]

        name = y.split("/")[-1].split(".")
        mask_name= name[0]
        mask_extn = name[1]

        x = cv2.imread(x,cv2.IMREAD_GRAYSCALE)
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        if augment ==True:

            aug = RandomRotate90(p=1.0)
            augmented = aug(image=x, mask=y)
            x0 = augmented["image"]
            y0 = augmented["mask"]

            aug = RandomGamma( always_apply=True, p=1.0)
            augmented = aug(image=x0, mask=y0)
            x1 = augmented["image"]
            y1 = augmented["mask"] #원래는 블러,로테 2개했음

            aug = CLAHE( always_apply=True, p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = GaussNoise( always_apply=True, p=1.0)
            augmented = aug(image=x0, mask=y0)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            aug = RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented["image"]
            y4 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x0, mask=y0)
            x5_ = augmented["image"]
            y5_ = augmented["mask"]
            x5, y5 = elastic_transform(x5_, y5_)

            aug = Sharpen(p=1.0)
            augmented = aug(image=x, mask=y)
            x6 = augmented["image"]
            y6 = augmented["mask"]



            # save_images = [x1, x2, x4, x5]
            # save_masks = [y1, y2, y4, y5]

            save_images = [x1,x2]
            save_masks = [y1,y2]

            #mal 에서 1,2,4, 다른건 1,2,3

        else:
            save_images = [x]
            save_masks = [y]

        idx=0
        for i, m in zip(save_images,save_masks):
            i=cv2.resize(i,(W,H))
            m = cv2.resize(m, (W, H))

            tmp_img_name= f"augmented_{image_name}_{idx}.{image_extn}"
            tmp_msk_name = f"augmented_{mask_name}_{idx}.{mask_extn}"

            image_path = os.path.join(save_path,"train",tmp_img_name)
            mask_path = os.path.join(save_path, "train_annot", tmp_msk_name)

            cv2.imwrite(image_path,i)
            cv2.imwrite(mask_path,m)

            idx+=1




if __name__ == "__main__":
    path = ""
    # path = "./Thyroid Dataset/DDTI dataset/CV_all/4/"
    images,masks = load_data(path)
    print(f"Original images: {len(images)} - Original masks:{len(masks)}")
    augment_data(images,masks,path,augment=True)

    images,masks = load_data(path)
    print(f"Augmented images: {len(images)} - Original masks:{len(masks)}")
