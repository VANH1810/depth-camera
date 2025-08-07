import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RED = (0,0,255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
margin = 2
font = ImageFont.truetype('libs/source/epkgobld.ttf', 15)
LINE_COLOR = [(0, 215, 255),
              (0, 255, 204),
              (0, 134, 255),
              (0, 255, 50),
              (77, 255, 222),
              (77, 196, 255),
              (77, 135, 255),
              (191, 255, 77),
              (77, 255, 77),
              (77, 222, 255),
              (255, 156, 127),
              (0, 127, 255),
              (255, 127, 77),
              (0, 77, 255),
              (255, 77, 36),
              (73, 165, 130),
              (130, 220, 175),
              (37,  12, 155),
              (119, 106, 153),
              (83, 118,  41),
              (228, 193, 188),
              (255, 15, 255),
              (255, 168, 77),
              (0, 255, 255),
              (255, 0, 36),
              (73, 0, 255),
              (130, 255, 175),
              (0,  255, 12),
              (119, 206, 153),
              (83, 245,  41)
            ]

L_PAIR = [[0,0,"NO", 'NOSE'],[0,1,"LE","LEFT_EYE"], [1,3,"LE","LEFT_EAR"],[0,2,"RE","RIGHT_EYE"],[2,4,"RE","RIGHT_EAR"],
          [5,7,"LS","LEFT_SHOULDER"],[7,9,"LE",'LEFT_ELBOW'], [6,8,'RS','RIGHT_SHOULDER'], [8,10,"RE","RIGHT_ELBOW"],
          [5,11,"LH","LEFT_HIP"],[6,12,"RH","RIGHT_HIP"], [5,6,"",""], [11,12,"",""],[11,13,"LK","LEFT_KNEE"],[13,15,"LA",'LEFT_ANKLE'],
          [12,14,"RK","RIGHT_KNEE"],[14,16,"RA","RIGHT_ANKLE"]]


p_color = [(0, 255, 255), (0, 191, 255),(0, 255, 102),(0, 77, 255), (0, 255, 0),
            (77,255,255), (77, 255, 204), (77,204,255), (191, 255, 77), (77,191,255), (191, 255, 77), 
            (204,77,255), (77,255,204), (191,77,255), (77,255,191), (127,77,255), (77,255,127), (255,255,70), (255,180,20), (20,180,255), (155,155,155)] 
line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), 
            (77,255,222), (77,196,255), (77,135,255), (191,255,77), 
            (77,255,77), (77,222,255), (255,156,127), (0,127,255), 
            (255,127,77), (0,77,255), (127,127,255), (255,0,127), 
            (0,127,0), (255,255,128), (0,0 ,50), (0,150 ,50), (255,180,20), (20,180,255)]

def _draw_limbs(current_pose, img, thickness=2):
    for i, pair in enumerate(L_PAIR):
        c_a, c_b, tag = pair[0], pair[1], pair[2]
        x0 = int(current_pose[c_a][0])
        y0 = int(current_pose[c_a][1])
        x1 = int(current_pose[c_b][0])
        y1 = int(current_pose[c_b][1])
        if (x0 > 0 and x1 > 0 and y0 > 0 and y1 > 0):
            if (x1- x0) > 0.3*img.shape[1] or (y0-y1) > 0.3*img.shape[0]:
                pass
                #_get_abnormal(img, LINE_COLOR[i], current_pose)
                #print((x0,y0), (x1,y1))
            img = cv2.line(img, (x0, y0), (x1, y1), LINE_COLOR[i], thickness)
            img = cv2.putText(img, tag, (x1, y1), cv2.FONT_HERSHEY_PLAIN,
                        max(0.5, thickness/5), LINE_COLOR[i], 1)
            img = cv2.circle(img, (x0, y0), 1, LINE_COLOR[i], thickness)
            img = cv2.circle(img, (x1, y1), 1, LINE_COLOR[i], thickness)
    return img
def get_bbox(pose):
    x_list = pose[:,0]
    y_list = pose[:,1]
    x_nonzeros = x_list[x_list > 0]
    y_nonzeros = y_list[y_list > 0]
    if len(x_nonzeros)*len(y_nonzeros) == 0:
        return [0,0,0,0]
    xmin = min(x_nonzeros)
    ymin = min(y_nonzeros)
    xmax = max(x_nonzeros)
    ymax = max(y_nonzeros)
    box = [xmin, ymin, xmax, ymax]
    return box

def _draw_pid(img, box, identity=None, thickness=1, offset=(0, 0), use_mct=False):
    x1, y1, x2, y2 = [int(i+offset[idx % 2]) for idx, i in enumerate(box)]
    x_mean = int((x1 + x2) / 2)
    y_mean = int((y1 + y2 ) / 2)
    if not use_mct:
        label = f'{identity}'
        height = 0.65 + thickness/5 * 0.25
        thickness = max(2, min(thickness, 5))
        if x_mean > 0 and x_mean < 640 and y_mean > 0 and y_mean < 360:
            cv2.putText(img, label, (x_mean, y_mean), cv2.FONT_HERSHEY_SIMPLEX, height, [200, 0, 200], thickness)
    else:
        label = '{}'.format(identity).lstrip('|')
        im = Image.fromarray(img)
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype('./source/fonts/Helios.ttf', 32)
        text_size = font.getsize(label)
        draw.rectangle(((x_mean, y_mean), (x_mean + text_size[0], y_mean + text_size[1])), fill="black")
        draw.text((x_mean,y_mean), label, font=font, fill=RED)
        img[:,:,:] = np.array(im)[:,:,:]
    return img


def _draw_action( overlay, box, cls_name, txtcolor, bcolor=(0, 0, 255)):
    x1, y1, _, y2 = box
    im = Image.fromarray(overlay)
    draw = ImageDraw.Draw(im)
    # text_width, text_height = draw.textsize(cls_name, font=font)
    _, _, text_width, text_height = draw.textbbox((0, 0), text=cls_name, font=font)
    x_label = max(0, x1)
    y_label = min(y1+20, y2)
    top_left = (int(x_label - margin), int(y_label - margin))
    bottom_right = (int(x_label + text_width + margin),
                    int(y_label + text_height + margin)
                    )
    cv2.rectangle(overlay, top_left, bottom_right, bcolor, -1)

    overlay = _draw_label(overlay, cls_name, (x_label, y_label), txtcolor)
    return overlay

def _draw_label(img, label, pos, textColor=(255, 255, 255)):
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    draw.text(pos, label, font=font, fill=textColor)
    frame = np.array(im)
    return frame
