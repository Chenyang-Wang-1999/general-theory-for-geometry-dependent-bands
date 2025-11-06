'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-06-25
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection


def create_white_to_color_cmap(rgb, name='white_to_color', n_bins=256, gray=1):
    """
    创建从白色到指定RGB颜色的渐变colormap
    参数:
        rgb: 目标颜色的RGB值，元组或列表，范围[0,1]
        name: colormap名称
        n_bins: 颜色分段数
    返回:
        自定义LinearSegmentedColormap对象
    """
    # 确保RGB值在0-1范围内
    r, g, b = [max(0, min(1, val)) for val in rgb]
    
    # 生成从白色(1,1,1)到目标颜色的渐变
    colors = np.linspace(gray, r, n_bins)[:, np.newaxis]  # R通道
    colors = np.hstack((colors, np.linspace(gray, g, n_bins)[:, np.newaxis]))  # G通道
    colors = np.hstack((colors, np.linspace(gray, b, n_bins)[:, np.newaxis]))  # B通道
    
    # 创建自定义colormap
    cmap = LinearSegmentedColormap.from_list(name, colors)
    return cmap
