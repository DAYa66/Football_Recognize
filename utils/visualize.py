import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from Football_Recognize.utils.bboxes import npchangexyorder
from Football_Recognize.utils.keypoints import JOINTS_PAIR


colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
_DEFAULT_COLOR_MAP = [255,0,0,0,255,0,0,0,255,255,255,0,165,42,42,0,192,0,250,170,31,250,170,32,196,196,196,190,153,153,180,165,180,90,120,150,250,170,33,250,170,34,128,128,128,
                  250,170,35,102,102,156,128,64,255,140,140,200,170,170,170,250,170,36,250,170,160,250,170,37,96,96,96,230,150,140,128,64,128,
                  110,110,110,110,110,110,244,35,232,128,196,128,150,100,100,70,70,70,150,150,150,150,120,90,220,20,60,220,20,60,
                  255,0,100,255,0,200, 250,170,29,250,170,28,250,170,26,250,170,25,250,170,24,250,170,22,250,170,21,
                  250,170,20, 250,170,19,250,170,18,250,170,12,250,170,11, 250,170,16, 250,170,15, 250,170,15,
                  64,170,64,230,160,50,70,130,180,190,255,255,152,251,152,107,142,35,0,170,30,
                  255,255,128,250,0,30,100,140,180,220,128,128,222,40,40,100,170,30,40,40,40,33,33,33,100,128,160,20,20,255,142,0,0,
                  70,100,150,250,171,30,250,172,30,250,173,30,250,174,30,250,175,30,250,176,30,210,170,100,153,153,153,153,153,153,128,128,128,
                  0,0,80,210,60,60,250,170,30,250,170,30,250,170,30,250,170,30,250,170,30,250,170,30,192,192,192,192,192,192,192,192,192,
                  220,220,0,220,220,0,0,0,196,192,192,192,220,220,0,140,140,20,119,11,32,150,0,255,0,60,100,0,0,142,0,0,90,
                  0,0,230,0,80,100,128,64,64,0,0,110,0,0,70,0,0,142,0,0,192,170,170,170,32,32,32,111,74,0,120,10,10,
                  81,0,81,111,111,0,0,0,0]
colors_tableau_large = np.reshape(np.array(_DEFAULT_COLOR_MAP),[-1,3]).tolist()


def color_fn(label):
    color_nr = len(colors_tableau)
    return colors_tableau[label%color_nr]

def fixed_color_large_fn(label):
    color_nr = len(colors_tableau_large)
    return colors_tableau_large[label%color_nr]

def default_text_fn(label,score):
    return str(label)

def get_text_pos_fn(pmin,pmax,bbox,label,text_size):
    text_width,text_height = text_size
    return (pmin[0] + text_height+5, pmin[1]+5)

def write_fps(frame, fps) -> None:
    cv2.putText(frame, str(round(fps, 1)), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=1)

def draw_bboxes_xy(img, classes=None, scores=None, bboxes=None,
                color_fn=fixed_color_large_fn,
                text_fn=default_text_fn,
                get_text_pos_fn=get_text_pos_fn,
                thickness=2,show_text=True,font_scale=1.2,text_color=(0.,255.,0.),
                is_relative_coordinate=False,
                is_show_text=None,
                fill_bboxes=False):
    if bboxes is not None:
        bboxes = npchangexyorder(bboxes)
    return draw_bboxes(img,classes,scores=scores,bboxes=bboxes,color_fn=color_fn,
                       text_fn=text_fn,get_text_pos_fn=get_text_pos_fn,thickness=thickness,
                       show_text=show_text,font_scale=font_scale,text_color=text_color,
                       is_relative_coordinate=is_relative_coordinate,
                       is_show_text=is_show_text,
                       fill_bboxes=fill_bboxes)

'''
bboxes: [N,4] (y0,x0,y1,x1)
color_fn: tuple(3) (*f)(label)
text_fn: str (*f)(label,score)
get_text_pos_fn: tuple(2) (*f)(lt_corner,br_corner,bboxes,label)
'''
def draw_bboxes(img, classes=None, scores=None, bboxes=None,
                        color_fn=fixed_color_large_fn,
                        text_fn=default_text_fn,
                        get_text_pos_fn=get_text_pos_fn,
                        thickness=2,show_text=True,font_scale=1.2,text_color=(0.,255.,0.),
                        is_relative_coordinate=True,
                        is_show_text=None,
                        fill_bboxes=False):
    if bboxes is None:
        return img

    bboxes = np.array(bboxes)
    if len(bboxes) == 0:
        return img
    if classes is None:
        classes = np.zeros([bboxes.shape[0]],dtype=np.int32)
    if is_relative_coordinate and np.any(bboxes>1.1):
        print(f"Use relative coordinate and max bboxes value is {np.max(bboxes)}")
    elif not is_relative_coordinate and np.all(bboxes<1.1):
        print(f"Use absolute coordinate and max bboxes value is {np.max(bboxes)}")

    bboxes_thickness = thickness if not fill_bboxes else -1
    if is_relative_coordinate:
        shape = img.shape
    else:
        shape = [1.0,1.0]
    if len(img.shape)<2:
        print(f"Error img size {img.shape}.")
        return img
    img = np.array(img)
    if scores is None:
        scores = np.ones_like(classes,dtype=np.float32)
    if not isinstance(bboxes,np.ndarray):
        bboxes = np.array(bboxes)
    for i in range(bboxes.shape[0]):
        try:
            bbox = bboxes[i]
            if color_fn is not None:
                color = color_fn(classes[i])
            else:
                color = (int(random.random()*255), int(random.random()*255), int(random.random()*255))
            p10 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
            p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
            cv2.rectangle(img, p10[::-1], p2[::-1], color, bboxes_thickness)
            if show_text and text_fn is not None:
                f_show_text = True
                if is_show_text is not None:
                    f_show_text = is_show_text(p10,p2)

                if f_show_text:
                    text_thickness = 1
                    s = text_fn(classes[i], scores[i])
                    text_size,_ = cv2.getTextSize(s,cv2.FONT_HERSHEY_DUPLEX,fontScale=font_scale,thickness=text_thickness)
                    p = get_text_pos_fn(p10,p2,bbox,classes[i],text_size)
                    cv2.putText(img, s, p[::-1], cv2.FONT_HERSHEY_DUPLEX,
                                fontScale=font_scale,
                                color=(0., 0., 0.0),
                                thickness=text_thickness+1)
                    cv2.putText(img, s, p[::-1], cv2.FONT_HERSHEY_DUPLEX,
                                fontScale=font_scale,
                                color=(255., 0., 0.0),
                                thickness=text_thickness)
                    
        except Exception as e:
            bbox = bboxes[i]
            p10 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
            p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
            if color_fn is not None:
                color = color_fn(classes[i])
            else:
                color = (random.random()*255, random.random()*255, random.random()*255)
            print("Error:",img.shape,shape,bboxes[i],classes[i],p10,p2,color,thickness,e)
            
    return img


def add_jointsv1(image, joints, color, r=5,line_thickness=2,no_line=False,joints_pair=None,left_node=None):

    def link(a, b, color):
        jointa = joints[a]
        jointb = joints[b]
        cv2.line(
                image,
                (int(jointa[0]), int(jointa[1])),
                (int(jointb[0]), int(jointb[1])),
                color, line_thickness )

    # add link
    if not no_line and joints_pair is not None:
        for pair in joints_pair:
            link(pair[0], pair[1], color)

    # add joints
    node_color = None
    for i, joint in enumerate(joints):
        if left_node is None:
            node_color = colors_tableau[i]
        elif i in left_node:
            node_color = (0,255,0)
        else:
            node_color = (0,0,255)
        cv2.circle(image, (int(joint[0]), int(joint[1])), r, node_color, -1)

    return image


def add_jointsv2(image, joints, color, r=5,line_thickness=2,no_line=False,joints_pair=None,left_node=None):

    def link(a, b, color):
        jointa = joints[a]
        jointb = joints[b]
        if jointa[2] > 0.01 and jointb[2] > 0.01:
            cv2.line(
                image,
                (int(jointa[0]), int(jointa[1])),
                (int(jointb[0]), int(jointb[1])),
                color, line_thickness )

    # add link
    if not no_line and joints_pair is not None:
        for pair in joints_pair:
            link(pair[0], pair[1], color)

    # add joints
    for i, joint in enumerate(joints):
        if joint[2] > 0.05 and joint[0] > 1 and joint[1] > 1:
            if left_node is None:
                node_color = colors_tableau[i]
            elif i in left_node:
                node_color = (0,255,0)
            else:
                node_color = (0,0,255)
            cv2.circle(image, (int(joint[0]), int(joint[1])), r, node_color, -1)

    return image


def draw_keypoints(image, joints, color=[0,255,0],no_line=False,joints_pair=JOINTS_PAIR,left_node=list(range(1,17,2)),r=5,line_thickness=2):
    '''

    Args:
        image: [H,W,3]
        joints: [N,kps_nr,2] or [kps_nr,2]
        color:
        no_line:
        joints_pair: [[first idx,second idx],...]
    Returns:

    '''
    image = np.ascontiguousarray(image)
    joints = np.array(joints)
    if color is None:
        use_random_color=True
    else:
        use_random_color = False
    if len(joints.shape)==2:
        joints = [joints]
    else:
        assert len(joints.shape)==3,"keypoints need to be 3-dimensional."

    for person in joints:
        if use_random_color:
            color = np.random.randint(0, 255, size=3)
            color = [int(i) for i in color]

        if person.shape[-1] == 3:
            add_jointsv2(image, person, color=color,no_line=no_line,joints_pair=joints_pair,left_node=left_node,r=r,
                         line_thickness=line_thickness)
        else:
            add_jointsv1(image, person, color=color,no_line=no_line,joints_pair=joints_pair,left_node=left_node,r=r,
                         line_thickness=line_thickness)

    return np.array(image)


def get_text_rect(img,pos,text,font,margin0=7,margin1=7):
    img = Image.fromarray(img)


    painter = ImageDraw.Draw(img)
    text_w = 1
    text_h = 1
    for t in text:
        tw,th= painter.textsize(t,font=font)
        text_h = th
        text_w = max(text_w,tw)

    rect = [pos[0]-margin0,pos[1]-margin0,pos[0]+text_w+margin0,pos[1]+text_h+(text_h+margin1)*(len(text)-1)+margin0]
    rect = [max(x,0) for x in rect]
    return rect,text_w,text_h

def __get_img_rect(img,rect):
    return img[rect[1]:rect[3],rect[0]:rect[2]]

def draw_text(img,pos,text,background_color=(0,0,255),text_color=(255,255,255),font_size=17,
              font='/home/wj/ai/file/font/simhei.ttf',margin0=7,margin1=7,alpha=0.4):
    '''

    Args:
        img: [H,W,3] rgb image
        pos: (x,y) draw position
        text: a str a list of str text do draw
        background_color: rgb rect background color or None
        text_color: rgb text color
        font_size:
        font: ImageFont.trutype or font path
        margin0: text top left right bottom margin
        margin1: margin between text line
        alpha: background alpha

    Returns:

    '''
    if isinstance(text,str):
        if len(text)==0:
            return img
        text = [text]
    if isinstance(font,str):
        font = ImageFont.truetype(font, font_size)
    rect,text_w,text_h = get_text_rect(img,pos, text,
                                       font,
                                       margin0=margin0,
                                       margin1=margin1)
    if background_color is not None:
        target_img = __get_img_rect(img,rect)
        background = np.array(list(background_color)).reshape([1,1,3])*alpha
        target_img = (target_img*(1-alpha)+background).astype(np.uint8)
        img[rect[1]:rect[3],rect[0]:rect[2]] = target_img

    img = Image.fromarray(img)
    painter = ImageDraw.Draw(img)
    y = pos[1]
    for t in text:
        painter.text((pos[0],y),t,
                     font=font,
                     fill=text_color)
        y += text_h+margin1

    return np.array(img)
