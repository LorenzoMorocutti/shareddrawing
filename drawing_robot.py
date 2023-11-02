
import time
# import threading
# import pandas as pd
# from rdp import rdp
# import numpy as np
import argparse
from subprocess import Popen
import ast
import yarp
import matplotlib.pyplot as plt
import tensorflow as tf
import json

from six.moves import xrange

# libraries required for visualisation:
from IPython.display import SVG, display
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import svgwrite

# import our command line tools
from magenta.models.sketch_rnn.sketch_rnn_train import *
#model_predict.py is the modified file getting from model.py of sketch-RNN :
from magenta.models.sketch_rnn.model_predict_bd import *
from magenta.models.sketch_rnn.utils import *
from magenta.models.sketch_rnn.rnn import *

#from pynput.mouse import Button, Controller
from math import cos, sin, pi
import pyautogui
tf.compat.v1.disable_eager_execution()



class robot_drawing():

    def __init__(self):
        self.LEFT_MARGIN = 0.1
        self.screen_height = 1080


    def draw_robot(self, last_x, last_y):
        #mouse = Controller()


        time.sleep(0.1)

        with open('strokes_predict.npy', 'rb') as file:
            #gen_stroke = np.load(file)
            gen_stroke = np.load(file)
            coord = np.zeros(np.shape(gen_stroke))
            print("gen_stroke \n", gen_stroke)
            file.close()

        last_point = [last_x, last_y]
        print("last point :")
        #print(self.last_point)
        print(last_point)


        gen_stroke[0, 0:2] += last_point[0:2]
        print("gen_stroke before adding:", gen_stroke)
        for i in range(len(gen_stroke) - 1):
            #coord[i+1, 0:2] = gen_stroke[i + 1, 0:2] + coord[i, 0:2]
            gen_stroke[i + 1, 0:2] += gen_stroke[i, 0:2]


        print("\n\n\nThese are gen_strokes: ", gen_stroke)

        print(gen_stroke[0][0], gen_stroke[0][1])

        x0, y0 = self.change_base_ref(gen_stroke[0][0], gen_stroke[0][1])
        pyautogui.moveTo(x0, y0)

        time.sleep(0.5)

        #pyautogui.click(x=745, y=692, clicks=2, interval=1)
        pyautogui.mouseDown()

        print("down")



        for n_stroke, pt in enumerate(gen_stroke[0:, :]):
        #for pt in gen_stroke[:, :]:
            if n_stroke > 100:
                print("too many strokes for the robot, let's stop drawing here")
                break
            x = pt[0]
            y = pt[1]
            time.sleep(0.1)
            x0, y0 = self.change_base_ref(x, y)
            print("mouse position: ", x0, y0)

            time.sleep(0.1)

            pyautogui.moveTo(x0, y0)

            #print(mouse.position)

        print("releasing Button left")
        pyautogui.mouseUp()

        return


    def change_base_ref(self, x, y):
        x = x + self.LEFT_MARGIN
        y = self.screen_height - y
        return x,y


# if __name__ == "__main__":
#     screen_height = 1080
#     screen_width = 1920
#
#     rec = "[" + str(int(screen_width / 3)) + "," + str(int(screen_height / 2)) + "]"
#     alpha = 45 * pi / 100
#     SECURITY_MARGIN = 0.1
#     LEFT_MARGIN = 0
#
#     print("in drawing")
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--pt_coord",
#         type=str,
#         default=rec,
#         help="both X and Y coordinates of the last point of the last stroke drawn")
#     parser.add_argument(
#         "--predict_file",
#         type=str,
#         default="strokes_predict.npy",
#         help="name of file on which will be saved the predicting points to follow to draw the next stroke on the screen")
#
#     FLAGS, unparsed = parser.parse_known_args()
#
#     draw_robot()

