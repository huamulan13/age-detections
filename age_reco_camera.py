import argparse, cv2
from datetime import datetime
from typing import Union, List
import numpy as np
from pustaka import helper,Age
from PIL import Image, ImageFont, ImageDraw

age_model = Age.loadModel()

def builder(args):
    face_detector = helper.get_dlib_face_detector()
    cap = cv2.VideoCapture(args.camera)
    colors = ["blueviolet","brown","pink", "orange","green", "blue"]
    font = ImageFont.truetype("./FiraCode-Regular.ttf", 35)
    while True:
            o, image = cap.read()
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            landmarks = face_detector(image)
            draw = ImageDraw.Draw(image)
            for i,landmark in enumerate(landmarks):
                linep = helper.crop_face(image, landmark, expand=.65)
                face = helper.align_and_crop_face(image, linep, i, 224, show_face=args.show_face)
                linep = linep[0]
                color_pic = i%len(colors)

                face = np.expand_dims(face, axis=0)
                signature = age_model.predict(face,verbose = 0)
                apparent_age = Age.findApparentAge(signature)
                
                #Draw Rectangle
                draw.line([(linep[0],linep[1]),(linep[2],linep[1])], fill=colors[color_pic], width=5)
                draw.line([(linep[2],linep[3]),(linep[2],linep[1])], fill=colors[color_pic], width=5)
                draw.line([(linep[0],linep[1]),(linep[0],linep[3])], fill=colors[color_pic], width=5)
                draw.line([(linep[0],linep[3]),(linep[2],linep[3])], fill=colors[color_pic], width=5)
                
                #Draw Age
                draw.text((linep[0],linep[3]), str(int(apparent_age)),colors[color_pic],font=font)
            open_cv_image =  helper.convert_rgb2bgr(image)
            cv2.imshow("Age_Reco", open_cv_image)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                print("Stop!")
                helper.clear_shel()
                break

if __name__ == '__main__':
    helper.clear_shel()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--camera',
        type=int, 
        default=0,
        help="field to select camera number"
    )

    parser.add_argument(
        '-f','--show_face', 
        type=bool, 
        default=False,
        help="Displays the cropped face in a separate window",
    )
    args = parser.parse_args()
    #Membersihkan keluaran Terminal Sebelumnya
    print("Start")
    builder(args)