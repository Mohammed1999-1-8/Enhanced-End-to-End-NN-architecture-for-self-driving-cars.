from pynput import keyboard
import time
import os
import numpy as np
from mss import mss
import cv2
import torch
import pandas as pd


path = 'raw_data/train/'

csv_file_name = 'ref.csv'


screen_hight = 320
screen_width = 650
screen_top_padding = 180
screen_left_padding = 40
#################
speedometer_hight = 180
speedometer_width = 180
speedometer_top_padding = 400
speedometer_left_padding = 600
#################
Image_hight = 128
Image_width = 256

#################
speedometer_image_hight = 64
speedometer_image_width = 64

sct = mss()
monitor={'top':screen_top_padding,'left':screen_left_padding,'width':screen_width,'height':screen_hight}
speedometer_monitor = {'top':speedometer_top_padding,'left':speedometer_left_padding,'width':speedometer_width,'height':speedometer_hight}


forward_img_counter = 0
left_img_counter = 0
reverse_img_counter = 0
right_img_counter = 0
break_img_counter = 0
total_img_counter = 0


def load_status():
    global forward_img_counter
    global left_img_counter
    global reverse_img_counter
    global right_img_counter
    global break_img_counter
    global total_img_counter

    try :
        state = torch.load(path + 'status.s')
        forward_img_counter = state['forword']
        left_img_counter = state['left']
        reverse_img_counter = state['reverse']
        right_img_counter = state['right']
        break_img_counter = state['break']
        total_img_counter = state['total']

        print('state was loaded !!')
    except :
        print('state loading failed !!!')

    if total_img_counter == 0:

        data = {
        'im_path':[],
        'sp_path':[],
        'ref':[]
        }

        df = pd.DataFrame(data)

        df.to_csv(path+csv_file_name)


    return

def save_status():

    state = {
    'forword' : forward_img_counter , 
    'left' : left_img_counter ,
    'reverse' : reverse_img_counter , 
    'right': right_img_counter , 
    'break' : break_img_counter ,
    'total' : total_img_counter

    }

    torch.save(state ,path + 'status.s')
    print("state was saved .")

    return


def show_state():
    print("forward_img_counter : ",forward_img_counter)
    print("left_img_counter : ",left_img_counter)
    print("reverse_img_counter : ",reverse_img_counter)
    print("right_img_counter : ",right_img_counter)
    print("break_img_counter : ",break_img_counter)
    print('total : ' + str(forward_img_counter + left_img_counter + reverse_img_counter + right_img_counter + break_img_counter))
    return





def grab_screen(a_monitor,width,hight):
    screen = np.array(sct.grab(a_monitor))
    screen = cv2.resize(screen, (width, hight))
    return screen



def on_press(key):
    key_list = ['w','a','s','d','l']
    global forward_img_counter
    global left_img_counter
    global reverse_img_counter
    global right_img_counter
    global break_img_counter
    global total_img_counter
    try :
        char = key.char
        if char  in key_list :
            total_img_counter = total_img_counter + 1
            (main_screen , speedometer) = (grab_screen(monitor,Image_width,Image_hight),
                grab_screen(speedometer_monitor,speedometer_image_width,speedometer_image_hight))
            main_img_dir = path + 'image/image_' + str(total_img_counter) + '.jpg'
            PS_img_dir = path + 'speedometer/sp_image_' + str(total_img_counter) + '.jpg'
            cv2.imwrite(main_img_dir, main_screen)
            cv2.imwrite(PS_img_dir,speedometer)
            if char == key_list[0] :
                forward_img_counter = forward_img_counter + 1

                data = {
                'im_path':[main_img_dir],
                'sp_path':[PS_img_dir],
                'ref':[0]
                }
            if char == key_list[1] :
                left_img_counter = left_img_counter + 1

                data = {
                'im_path':[main_img_dir],
                'sp_path':[PS_img_dir],
                'ref':[1]
                }
            if char == key_list[2] :
                reverse_img_counter = reverse_img_counter + 1

                data = {
                'im_path':[main_img_dir],
                'sp_path':[PS_img_dir],
                'ref':[2]
                }
            if char == key_list[3] :
                right_img_counter = right_img_counter + 1

                data = {
                'im_path':[main_img_dir],
                'sp_path':[PS_img_dir],
                'ref':[3]
                }
            if char == key_list[4] :
                break_img_counter = break_img_counter + 1

                data = {
                'im_path':[main_img_dir],
                'sp_path':[PS_img_dir],
                'ref':[4]
                }

            df = pd.DataFrame(data)

            df.to_csv(path+csv_file_name, mode = 'a' , index = False, header = False)



    except AttributeError:
        if (key == keyboard.Key.esc):
            return False

    
   


if __name__ == "__main__":
    
    
    time.sleep(6)
    load_status()

    
    with keyboard.Listener(on_press) as listener:
        listener.join()



    show_state()
    save_status()