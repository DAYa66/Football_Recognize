import numpy as np
import plotly.graph_objs as go
import random
import matplotlib.pyplot as plt


def create_colormap(n_colors):
    part_1 = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                (255, 255, 0),(0, 255, 255), (255, 0, 255),
                (100, 100, 0), (100, 0, 100), (0, 100, 100),
                (255, 255, 255), (100, 100, 100)]
    if n_colors > 11:
        second_cmap = plt.cm.get_cmap('viridis', n_colors-11)
        part_2 = []
        for i in range(n_colors-11):
            part_2 += [second_cmap(i)[:3]]
        random.shuffle(part_2)
        return np.concatenate((part_1, part_2), axis=0)
    else: return part_1
    

COLORS_TUPLE = create_colormap(50)
    
def color_fn(i):
    if i==-1:
        return (0, 0, 0)
    if i >= len(COLORS_TUPLE):
        return (random.random()*255, random.random()*255, random.random()*255)
    else: return 255*COLORS_TUPLE[i]


'''
todo: имеет смысл переписать через color_fn(), потому что по сути это color_fn, но не от лейбла, а от массива лейблов
'''
def get_colors_for_labels(labels):
    label_colors = []
    for i in labels:
        label_colors += [color_fn(i)]
    return label_colors



'''
Функция для пересчета векторов в формат, подходящий для отрисовки
'''
def transfer_vectors_to_rgb(vectors):
    vectors = np.asarray(vectors)
    vectors_copy = np.int0(vectors.copy()).clip(0, 255)/255
    return vectors_copy
    # b, g, r = vectors_copy[:, 0], vectors_copy[:, 1], vectors_copy[:, 2]
    # rgb = np.vstack((r, g, b)).T
    # return rgb


'''
Функция для отрисовки векторов цветов в виде точек в 3D пространстве.
vectors: np.array
Если параметр colors не определен, то будет отрисовка в собственном цвете, иначе - в указанных.
'''
def plot_3d_model(vectors, colors=None):
    if colors is None:
        colors = transfer_vectors_to_rgb(vectors)

    Scene = dict(xaxis = dict(title='R'), yaxis = dict(title='G'), zaxis = dict(title='B'))
    
    trace = go.Scatter3d(x=vectors[:, 0], y=vectors[:, 1], z=vectors[:, 2], 
                         mode='markers', 
                         marker=dict(color=colors, size=5, line=dict(color='black', width=5)))
    layout = go.Layout(margin=dict(l=0,r=0), scene = Scene, height = 800, width = 800)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    fig.show()