# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:19:42 2021

@author: DenizS
"""


import cv2 
import numpy as np
import pyautogui as p
import time as t
import tensorflow as tf
import load as l
from tkinter import *
from PIL import Image,ImageTk

def exit(event):
    root.destroy()

root=Tk()
root.geometry("900x700")
root.title("Video Player Controller")
root.configure(bg="black")

imgDur = ImageTk.PhotoImage(Image.open("handSign//dur.png"))
panel1 = Label(root, image = imgDur)
panel1.place(x=0,y=545)
#panel1.pack(side="left")
imgIleri = ImageTk.PhotoImage(Image.open("handSign//ileri.png"))
panel2 = Label(root, image = imgIleri)
panel2.place(x=0,y=0)
#panel2.pack(side="left")
imgGeri = ImageTk.PhotoImage(Image.open("handSign//geri.png"))
panel3 = Label(root, image = imgGeri)
panel3.place(x=0,y=140)
imgYukari = ImageTk.PhotoImage(Image.open("handSign//ses_yukselt.png"))
panel4 = Label(root, image = imgYukari)
panel4.place(x=0,y=280)
imgAsagi = ImageTk.PhotoImage(Image.open("handSign//ses_alcalt.png"))
panel5 = Label(root, image = imgAsagi)
panel5.place(x=0,y=420)
#panel3.pack(side="left")

f1=LabelFrame(root,bg="red")
f1.pack()
L1=Label(f1,bg="red")
L1.pack(side="right")



category_index = l.label_map_util.create_category_index_from_labelmap(l.k.ANNOTATION_PATH+'/label_map.pbtxt')

# Setup capture
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True: 
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = l.detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    
    

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    
    
    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    l.viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=1,
                min_score_thresh=.9,
                agnostic_mode=False)
    image_np_with_detections=cv2.cvtColor(image_np_with_detections,cv2.COLOR_BGR2RGB)
    image_np_with_detections=ImageTk.PhotoImage(Image.fromarray(image_np_with_detections))
    L1['image']=image_np_with_detections
    root.update()
    #cv2.imshow('Video Player Controller',  cv2.resize(image_np_with_detections, (width, height)))

    if detections['detection_scores'][0]>=0.99:
        if detections['detection_classes'][0]==0:
            print('İleri')
            p.press('right')
            panel2.config(bg="green")
            t.sleep(0.15)
            
        elif detections['detection_classes'][0]==1:
            print('Geri')
            p.press('left')
            panel3.config(bg="green")
            t.sleep(0.15)

        elif detections['detection_classes'][0]==2:
            print('Ses Yükselt')
            p.press('up')
            panel4.config(bg="green")
            t.sleep(0.15)

        elif detections['detection_classes'][0]==3:
            print('Ses Alçalt')
            p.press('down')
            panel5.config(bg="green")
            t.sleep(0.15) 

        elif detections['detection_classes'][0]==4:
            print('Dur')
            p.press('space')
            panel1.config(bg="green")
            t.sleep(0.33) 

        else:
            pass
    else:
        panel1.config(bg="red")
        panel2.config(bg="red")
        panel3.config(bg="red")
        panel4.config(bg="red")
        panel5.config(bg="red")
        pass
    
    root.bind("<Escape>", exit)


root.mainloop()
cap.release()
cv2.destroyAllWindows()