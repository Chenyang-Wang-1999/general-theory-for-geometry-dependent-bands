'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-05-29
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import shapely
import numpy as np
import matplotlib.pyplot as plt

polygon = np.random.randn(3,2)
polygon_obj = shapely.Polygon(polygon)
pt_list = np.random.randn(1000,2)

fig = plt.figure()
ax = fig.gca()
polygon_loop = np.vstack([polygon, polygon[0,:]])
ax.plot(polygon_loop[:,0], polygon_loop[:,1], '-')
for j in range(1000):
    if(polygon_obj.contains(shapely.Point(pt_list[j,:]))):
        ax.plot(pt_list[j,0], pt_list[j,1],'b.')
    else:
        ax.plot(pt_list[j,0], pt_list[j,1],'r.')

plt.show()