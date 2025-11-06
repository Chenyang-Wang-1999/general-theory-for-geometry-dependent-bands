'''
author:        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2024-05-17
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
sys.path.append('..')
import figure_settings_common as fs
mpl.use("pdf")
plt.style.use("paper_plot.mplstyle")

###### user settings #########
font_size = 10
text_latex = "$\\mathcal{B}_1$ "
##############################

fig = plt.figure(figsize=(5*fs.cm, 5 * fs.cm))
ax = fig.gca()
ax.set_axis_off()
txt = plt.text(0,0,text_latex, fontsize=font_size)
print(txt.get_fontname())
print(txt.get_fontfamily())
print(txt.get_fontsize())
plt.savefig("current-formula.pdf")