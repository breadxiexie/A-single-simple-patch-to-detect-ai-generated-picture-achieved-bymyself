import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image

# 定义SRM卷积层
class SRMConv2d_all(nn.Module):
    def __init__(self, inc=3, learnable=False):
        super(SRMConv2d_all, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)

    def forward(self, x):
        out = F.conv2d(x, self.kernel, stride=1, padding=2, groups=3)
        out = self.truc(out)
        return out

    def _build_kernel(self, inc):
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
        filters = []
        for i, filt in enumerate(filters_all):
            filt = np.asarray(filt, dtype=float) / q[i]
            filt = np.repeat(filt[None, None, ...], inc, axis=0)
            filters.append(filt)
        filters = np.concatenate(filters, axis=0)  # 将滤波器数组合并为一个大数组
        filters = np.tile(filters, (inc, 1, 1, 1))  # 复制滤波器以匹配in_channels
        filters = torch.from_numpy(filters).float()  # 转换为torch tensor
        return filters

# 应用SRM滤波器并进行二值化处理的函数
def apply_srm_to_image(image_path, srm_layer, threshold_value):
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.ToTensor()
    img_t = preprocess(img)
    img_t = img_t.unsqueeze(0).to(device)

    with torch.no_grad():
        processed_img = srm_layer(img_t)
    processed_img = ((processed_img + 3.0) / 6.0).squeeze(0)
    processed_img = processed_img.mean(dim=0, keepdim=True)
    processed_img = (processed_img > threshold_value).float()
    to_pil_image = transforms.ToPILImage()
    srm_img_bw = to_pil_image(processed_img)

    return srm_img_bw

# 主程序
# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化SRM模型，并应用所有三个filter
srm_layer_all = SRMConv2d_all(inc=3, learnable=False).to(device)

# 二值化阈值，根据需求自行调整
threshold_value = 0.5  # 此处设置为0到1之间的值，对应于-3到3的硬限幅值
# 输入的顶层文件夹路径和输出的结果文件夹路径
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
                srm_img_bw = apply_srm_to_image(image_path, srm_layer_all, threshold_value)
                new_folder_path = dirpath.replace(top_folder_path, result_folder_path)
                if not os.path.exists(new_folder_path):
                    os.makedirs(new_folder_path)
                new_image_path = os.path.join(new_folder_path, filename)
                srm_img_bw.save(new_image_path)
                print(f"Saved: {new_image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
