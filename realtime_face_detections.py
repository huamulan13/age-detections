import os, argparse, cv2
from typing import Union, List
import numpy as np
from pustaka import helper
from PIL import Image, ImageDraw

def builder(args):
    face_detector = helper.get_dlib_face_detector()
    cap = cv2.VideoCapture(0)
    colors = ["red","green", "blue", "pink", "orange", "brown"]
    while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            o, image = cap.read()
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            landmarks = face_detector(image)
            draw = ImageDraw.Draw(image)

            for i,landmark in enumerate(landmarks):
                linep = helper.crop_face(image, landmark, expand=.70)
                linep = linep[0]
                color_pic = i%len(colors)
                draw.line([(linep[0],linep[1]),(linep[2],linep[1])], fill=colors[color_pic], width=5)
                draw.line([(linep[2],linep[3]),(linep[2],linep[1])], fill=colors[color_pic], width=5)
                draw.line([(linep[0],linep[1]),(linep[0],linep[3])], fill=colors[color_pic], width=5)
                draw.line([(linep[0],linep[3]),(linep[2],linep[3])], fill=colors[color_pic], width=5)
            open_cv_image = np.array(image)
            # Convert RGB to BGR 
            open_cv_image = open_cv_image[:, :, ::-1].copy() 
            cv2.imshow("XMOD-FeceNET-face", open_cv_image)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='./input',
    )
    parser.add_argument(
        '--database_file', 
        type=str, 
        default='./iqbal.xmod',
    )
    args = parser.parse_args()
    builder(args)
