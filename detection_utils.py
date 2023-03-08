import numpy as np


'''
Функция для получения bbox'а нужной части футболки из keypoints
'''
def get_bbox_coords(rs, ls, rh, lh):
    x_array = [rs[0], ls[0], rh[0], lh[0]]
    y_array = [rs[1], ls[1], rh[1], lh[1]]
    
    x_array = sorted(x_array)
    y_array = sorted(y_array)
    
    x0, x1 = x_array[1:3]
    y0, y1 = y_array[1:3]
    
    return int(x0), int(y0), int(x1), int(y1)

'''
Функция для перевода координат вершин bbox'а из пикселей в относительные
'''
def get_relative_bboxes(bboxes, height, width):
    rel_bboxes = bboxes.copy()

    # x0, y0, x1, y1
    rel_bboxes[:, 0] /= width
    rel_bboxes[:, 2] /= width
    rel_bboxes[:, 1] /= height
    rel_bboxes[:, 3] /= height
    
    return rel_bboxes


'''
Функция, определяющая вектор цвета из полученнного bbox'a
todo: исправить разделение на 3 вектора, оно не имеет смысла
возвращает вектор в формате RGB
'''
def get_shirt_vector(frame, x0, y0, x1, y1):
    shirt_color = frame[y0:y1, x0:x1, :]
    means = shirt_color.mean(axis=1)
         
    if np.any(np.isnan(means)):
        print("Nan in means")
        print(f"x0:{x0}; x1:{x1}; y0:{y0}; y1:{y1}")
        return frame[int((y0+y1)/2), int((x0+x1)/2), :]
    
    delta = means.shape[0] // 3
    vec1 = means[:delta, :]
    vec2 = means[delta:2*delta, :]
    vec3 = means[2*delta:3*delta, :]

    vec1 = vec1.mean(axis=0)
    vec2 = vec2.mean(axis=0)
    vec3 = vec3.mean(axis=0)  
    
    res = np.mean([vec1, vec2, vec3], axis=0)
    b, g, r = res[0], res[1], res[2]
    res_rgb = np.hstack((r, g, b))
    return res_rgb


'''
Функция, которая возвращает векторы цветов, используя текущий кадр и распознанные keypoints
'''
def get_vectors(frame, kps):    
    vectors = []
    for point in kps:
        ls = point[5] # left_shoulder
        rs = point[6] # right_shoulder
        lh = point[11] # left_hip
        rh = point[12] # right_hip
        
        x0, y0, x1, y1 = get_bbox_coords(rs, ls, rh, lh)
        if x0 == 0 or x1 == 0 or y0 == 0 or y1 == 0:
            # print("zeros encountered")
            pass
        if x0 < 0: x0 = 0
        if x1 < 0: x1 = 0
        if y0 < 0: y0 = 0
        if y1 < 0: y1 = 0
        # print(f"x0:{x0}; x1:{x1}; y0:{y0}; y1:{y1}")
        if x1 - x0 <=3: x1 += 3
        if y1 - y0 <=3: y1 += 3
        
        height, width = frame.shape[0:2]
        
        if x1 >= width: 
            x0, x1 = width-3, width
        if y1 >= height:
            y0, y1 = height-3, height
            
        vec = get_shirt_vector(frame, x0, y0, x1, y1)
        vectors += [vec]
    return vectors

