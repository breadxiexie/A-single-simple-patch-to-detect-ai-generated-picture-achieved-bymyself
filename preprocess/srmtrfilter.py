import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

class SRMConv2d(nn.Module):
    def __init__(self, inc, outc, filter_num, learnable=False):
        super(SRMConv2d, self).__init__()
        self.outc = outc  # 将outc的赋值提前
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc, filter_num)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)

    def forward(self, x):
        out = F.conv2d(x, self.kernel, stride=1, padding=2, groups=1)
        out = self.truc(out)
        return out

    def _build_kernel(self, inc, filter_num):
        q = [4.0, 12.0, 2.0]
        filters_all = [
            [[0, 0, 0, 0, 0],
             [0, -1, 2, -1, 0],
             [0, 2, -4, 2, 0],
             [0, -1, 2, -1, 0],
             [0, 0, 0, 0, 0]],
            [[-1, 2, -2, 2, -1],
             [2, -6, 8, -6, 2],
             [-2, 8, -12, 8, -2],
             [2, -6, 8, -6, 2],
             [-1, 2, -2, 2, -1]],
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 1, -2, 1, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]
        ]
        selected_filter = np.asarray(filters_all[filter_num], dtype=float) / q[filter_num]
        filters = np.repeat(selected_filter[None, None, ...], inc, axis=1)
        filters = np.repeat(filters, self.outc, axis=0)
        return torch.FloatTensor(filters)

# 继续后续函数和主程序

def apply_srm_to_image(image_path, srm_layer, threshold):
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.ToTensor()
    img_t = preprocess(img)
    img_t = img_t.unsqueeze(0).to(device)

    with torch.no_grad():
        processed_img = srm_layer(img_t)
    processed_img = ((processed_img + 3.0) / 6.0).squeeze(0)
    processed_img = processed_img.mean(dim=0, keepdim=True)
    processed_img = (processed_img > threshold).float()
    to_pil_image = transforms.ToPILImage()
    srm_img_bw = to_pil_image(processed_img)

    return srm_img_bw

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化SRM模型，此处指定单个filter
filter_num = 1  # 指定filter编号，0, 1, 或 2
srm_layer = SRMConv2d(inc=3, outc=1, filter_num=filter_num, learnable=False).to(device)
# 二值化阈值，根据需求自行调整
threshold_value = 0.5  # 此处设置为0到1之间的值，对应于-3到3的硬限幅值


top_folder_path = r'path'
result_folder_path = r'path'

if not os.path.exists(result_folder_path):
    os.makedirs(result_folder_path)

for dirpath, dirnames, filenames in os.walk(top_folder_path):
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(dirpath, filename)
            print(f"Processing: {image_path}")

            try:
                srm_img_bw = apply_srm_to_image(image_path, srm_layer, threshold_value)
                new_folder_path = dirpath.replace(top_folder_path, result_folder_path)
                if not os.path.exists(new_folder_path):
                    os.makedirs(new_folder_path)
                new_image_path = os.path.join(new_folder_path, filename)
                srm_img_bw.save(new_image_path)
                #print(f"Saved: {new_image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

