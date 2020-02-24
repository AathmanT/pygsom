import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors


def show_gsom(output, max_count,index_col,label_col):
    #listed_color_map = _get_color_map(max_count, alpha=0.9)

    fig, ax = plt.subplots()
    for index, node in output.iterrows():
        x=node['x']
        y=node['y']
        if node['hit_count']>0:
            #c='red'
            #label = ", ".join(map(str,i[index_col]))
            count_0 = 0
            count_1 = 0
            label_str = ""
            labels = node[index_col]

            for label in labels:
                if label == '1':
                    count_1 += 1
                if label == '0':
                    count_0 += 1
            if count_1 != 0:
                label_str = label_str + "1(" + str(count_1) + ")"
            if count_0 != 0:
                label_str = label_str + "0(" + str(count_0) + ")"
        else:
            label_str = ""
            #c='yellow'
        ax.plot(x,y, 'o', color=findColor(count_0,count_1),markersize=1)
        # ax.annotate(label_str, (x, y), fontsize=2)
        print("{},{}  ==> {}".format(x, y,label_str))
        plt.text(x, y + 0.1, label_str, fontsize=2)

    ax.set_title("GSOM Map")
    #plt.show()
    plt.savefig("output/gsom_"+datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S")+".png",dpi=1200)

def findColor(count_0, count_1):
    if count_0 > count_1:
        color = 'blue'
    elif count_0 < count_1:
        color = 'red'
    else:
        color = 'orange'
    return color

# def _get_color_map(max_count, alpha=0.5):
#
#     np.random.seed(1)
#
#     cmap = cm.get_cmap('Reds', max_count + 1)  # set how many colors you want in color map
#     # https://matplotlib.org/examples/color/colormaps_reference.html
#
#     color_list = []
#     for ind in range(cmap.N):
#         c = []
#         for x in cmap(ind)[:3]: c.append(x * alpha)
#         color_list.append(tuple(c))
#
#     return colors.ListedColormap(color_list, name='gsom_color_list')


