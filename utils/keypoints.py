import numpy as np


JOINTS_PAIR = [[0 , 1], [1 , 2], [2 , 0], [1 , 3], [2 , 4], [3 , 5], [4 , 6], [5 , 6], [5 , 11],
[6 , 12], [11 , 12], [5 , 7], [7 , 9], [6 , 8], [8 , 10], [11 , 13], [13 , 15], [12 , 14], [14 , 16]]


def npget_bbox(keypoints,threshold=0.02):
    '''

    Args:
        keypoints: [N,3] or [N,2] ,[x,y,visible]
        threshold:

    Returns:
        [xmin,ymin,xmax,ymax]

    '''
    assert len(keypoints.shape)==2,f"ERROR kps shape {keypoints.shape}"

    if keypoints.shape[-1]>=3:
        mask = keypoints[:,2]>threshold
        if np.any(mask):
            keypoints = keypoints[mask]
        else:
            return None
    xmin = np.min(keypoints[:,0])
    xmax = np.max(keypoints[:,0])
    ymin = np.min(keypoints[:,1])
    ymax = np.max(keypoints[:,1])
    return np.array([xmin,ymin,xmax,ymax],dtype=np.float32)


