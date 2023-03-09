import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.collections import PatchCollection
def plot_color_code(x,folder_path,file_name,num_color):
    cmap = plt.cm.get_cmap('rainbow', num_color+ 1)
    # 绘制长方形带状条
    fig, ax = plt.subplots(figsize=(12, 1))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)
    patches_list = []
    factor = len(x)/2000 * 0.003
    for i in range(len(x)):
        rect = patches.Rectangle((factor * i, -1), factor, 1, linewidth=0, facecolor=cmap(x[i]))
        patches_list.append(rect)
    pc = PatchCollection(patches_list, match_original=True)
    ax.add_collection(pc)
    ax.axis('off')
    plt.savefig('{}{}.png'.format(folder_path,file_name))
    plt.clf()