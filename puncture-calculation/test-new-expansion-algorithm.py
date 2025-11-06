'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-05-07
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import numpy as np
import GBZ_manifold as Gm
import partial_GBZ_solver as pGs
import poly_tools as pt
import matplotlib.pyplot as plt



coeffs = pt.CScalarVec([
    1, -1, -1, -1, -1, -0.5, -0.5
])
degs = pt.CLaurentIndexVec([
  # E, beta1, beta2
    1, 0, 0,
    0, 1, 0,
    0, -1, 0,
    0, 1, -1,
    0, -2, 1,
    0, 0, 2,
    0, 0, -2
])
f = pt.CLaurent(3)
f.set_Laurent_by_terms(coeffs, degs)

E = 2
f_partial = f.partial_eval(
    pt.CScalarVec([E]), pt.CIndexVec([0]), pt.CIndexVec([1,2])
)
glue_tol = 0.05
GBZ_data, param_list, aGBZ_data = pGs.solve_GBZ(
    'CP1', f_partial, glue_tol=glue_tol
)

plt.figure(1)
sol_num = np.zeros(len(GBZ_data))
for j in range(len(GBZ_data)):
    sol_num[j] = len(GBZ_data[j])
plt.plot(param_list[:,0], sol_num, '.-')

plt.figure(2)
sol_num = np.zeros(len(aGBZ_data))
for j in range(len(aGBZ_data)):
    sol_num[j] = len(aGBZ_data[j])
plt.plot(param_list[:,0], sol_num, '.-')
plt.show()

GBZ_data, param_list, aGBZ_data = pGs.expand_solution(
    GBZ_data, aGBZ_data, param_list
)

GBZ_segs = pGs.get_GBZ_segments(GBZ_data, param_list, 10*glue_tol, pGs.aGBZEquation1DCP1(f_partial))
GBZ_loops = pGs.GBZ_segments_to_loops(GBZ_segs)
# GBZ_loops = pGs.batch_get_global(GBZ_loops)
# for loop in GBZ_loops:
#     err_code = pGs.glue_loop(loop, pGs.aGBZEquation1DCP1(f_partial), max_dist=glue_tol, glue_tol=1e-4)
#     print(err_code)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
print(len(GBZ_loops))
all_colors = ['r-', 'g-', 'b-', 'm-', 'y-']
for j, loop in enumerate(GBZ_loops): 
    # for seg in [loop]:
    for seg in loop:
        curr_points = np.zeros((len(seg.point_vec), 2), dtype=complex)
        for point_id, point in enumerate(seg.point_vec):
            Gm.CP1_to_chart(point, [0,0])
            curr_points[point_id,:] = point.coords
        ax.plot(curr_points[:,1].real, curr_points[:,1].imag, np.abs(curr_points[:,0]), all_colors[j%len(all_colors)])
        print(len(seg.point_vec))
        if(len(seg.point_vec) == 1):
            ax.plot(curr_points[:,1].real, curr_points[:,1].imag, np.abs(curr_points[:,0]), 'rx')
        print(Gm.get_distance(seg.point_vec[0], seg.point_vec[-1]))


ax.set_xlim([-4,4])
ax.set_ylim([-4,4])
ax.set_zlim([-4,4])
plt.show()