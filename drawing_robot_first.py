
import os
os.environ["KIVY_NO_ARGS"] = "1"
import sys
# from kivy.app import App
# from kivy.uix.widget import Widget
# from kivy.graphics import Line, Color
# from kivy.core.window import Window
# #from kivy.uix.button import Button  #####
# from kivy.interactive import InteractiveLauncher
# from kivy.graphics import Color, Ellipse
import time
# from kivy.uix.boxlayout import BoxLayout
# from kivy.lang import Builder
import threading
import pandas as pd
from rdp import rdp
import numpy as np
import argparse
from subprocess import Popen
import ast
import yarp
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import random

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

from pynput.mouse import Button, Controller
from math import cos, sin, pi

data_dir = '/usr/local/src/robot/cognitiveinteraction/container/12_categories/12_categories_NPZ/'
np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

def load_env_compatible(data_dir, model_dir):
  #Loads environment for inference mode, used in jupyter notebook.
  # modified https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/sketch_rnn_train.py
  # to work with depreciated tf.HParams functionality
  model_params = get_default_hparams()
  with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
    data = json.load(f)
  fix_list = ['conditional', 'is_training', 'use_input_dropout', 'use_output_dropout', 'use_recurrent_dropout']
  for fix in fix_list:
    data[fix] = (data[fix] == 1)
  model_params.parse_json(json.dumps(data))
  return load_dataset(data_dir, model_params, inference_mode=True)

def load_model_compatible(model_dir):
  #Loads model for inference mode, used in jupyter notebook.
  # modified https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/sketch_rnn_train.py
  # to work with depreciated tf.HParams functionality
  model_params = sketch_rnn_model.get_default_hparams()
  with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
    data = json.load(f)
  fix_list = ['conditional', 'is_training', 'use_input_dropout', 'use_output_dropout', 'use_recurrent_dropout']
  for fix in fix_list:
    data[fix] = (data[fix] == 1)
  model_params.parse_json(json.dumps(data))

  model_params.batch_size = 1  # only sample one at a time
  eval_model_params = sketch_rnn_model.copy_hparams(model_params)
  eval_model_params.use_input_dropout = 0
  eval_model_params.use_recurrent_dropout = 0
  eval_model_params.use_output_dropout = 0
  eval_model_params.is_training = 0
  sample_model_params = sketch_rnn_model.copy_hparams(eval_model_params)
  sample_model_params.max_seq_len = 1  # sample one point at a time
  return [model_params, eval_model_params, sample_model_params]


def change_base_ref(x, y):
    x = x + LEFT_MARGIN
    y = screen_height - y
    return x,y


def draw_robot():

    mouse = Controller()

    time.sleep(0.1)

    with open('strokes_predict.npy', 'rb') as file:
        #gen_stroke = np.load(file)
        gen_stroke = np.load(file)
        coord = np.zeros(np.shape(gen_stroke))
        print("gen_stroke \n", gen_stroke)
        file.close()
        # print("stroke generated :" )
        # print(gen_stroke)
        # last_point = ast.literal_eval(FLAGS.pt_coord)

    last_point = ast.literal_eval(FLAGS.pt_coord)
    print("last point :")
    #print(self.last_point)
    print(last_point)

    # for i in range(len(gen_stroke) - 1):
    #     #coord[i+1, 0:2] = gen_stroke[i + 1, 0:2] + coord[i, 0:2]
    #     gen_stroke[i, 0:2] = gen_stroke[i+1, 0:2]

    #k = len(gen_stroke)
    #for i in range(k - 1):
    #    gen_stroke[k-1-i, 0:2] = gen_stroke[k-2-i, 0:2]

    # # convert from Deltax, Deltay back to x,y
    #coord[0,0:2] = gen_stroke[0, 0:2] + self.last_point[0:2]

    #gen_stroke[0, 0:2] += self.last_point[0:2]
    gen_stroke[0, 0:2] += last_point[0:2]
    print("gen_stroke before adding:", gen_stroke)
    for i in range(len(gen_stroke) - 1):
        #coord[i+1, 0:2] = gen_stroke[i + 1, 0:2] + coord[i, 0:2]
        gen_stroke[i + 1, 0:2] += gen_stroke[i, 0:2]


    print("\n\n\nThese are gen_strokes: ", gen_stroke)

    print(gen_stroke[0][0], gen_stroke[0][1])

    x0, y0 = change_base_ref(gen_stroke[0][0], gen_stroke[0][1])
    mouse.position = (x0, y0)

    time.sleep(0.5)

    mouse.press(Button.left)

    for n_stroke, pt in enumerate(gen_stroke[0:, :]):
    #for pt in gen_stroke[:, :]:
        if n_stroke > 30:
            print("too many strokes for the robot, let's stop drawing here")
            break
        x = pt[0]
        y = pt[1]
        time.sleep(0.1)
        x0, y0 = change_base_ref(x, y)
        print("mouse position: ", x0, y0)

        time.sleep(0.1)

        mouse.position = (x0, y0)

        print(mouse.position)

    print("releasing Button left")
    mouse.release(Button.left)


    return



if __name__ == "__main__":
    screen_height = 1080
    screen_width = 1920
    carlo_screen_w = 0.294
    carlo_screen_h = 0.165

    rec = "[" + str(int(screen_width / 3)) + "," + str(int(screen_height / 2)) + "]"
    alpha = 45 * pi / 100
    SECURITY_MARGIN = 0.1
    LEFT_MARGIN = 0

    print("in drawing")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single_model_path",
        type=str,
        default="/usr/local/src/robot/cognitiveinteraction/container/12_unc_model/",
        help="where is located the repository storing the every single model trained for prediction")
    parser.add_argument(
        "--class_name",
        type=str,
        default="owl",
        help="name of model to be taken for prediction")
    parser.add_argument(
        "--data_saved_repo_name",
        type=str,
        default="/usr/local/src/robot/cognitiveinteraction/drawing_update/",
        help="repository where is located the overall data of the session")

    FLAGS, unparsed = parser.parse_known_args()

    model_dir_uncond = FLAGS.single_model_path + FLAGS.class_name + '_unc/'

    # Loading Model
    [train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = load_env_compatible(data_dir,
                                                                                                        model_dir_uncond)

    # start session:
    reset_graph()
    model = Model(hps_model, gpu_mode=False)
    eval_model = Model(eval_hps_model, reuse=True)
    sample_model = Model(sample_hps_model, reuse=True)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # loads the weights from checkpoint into our model
    load_checkpoint(sess, model_dir_uncond)

    std = 35
    strokes5 = sample(sess, sample_model, temperature=0.25)
    generated_next_stroke = to_normal_strokes(strokes5)
    # copy of the generated stroke to unnormalize it
    gen_stroke = np.copy(generated_next_stroke)

    gen_stroke[:, 0:1] *= std
    gen_stroke[:, 1:2] *= -std
    file_name_prediction_path = FLAGS.data_saved_repo_name + 'strokes_predict.npy'
    print(file_name_prediction_path)
    with open(file_name_prediction_path, 'w+b') as f:
        np.save(f, gen_stroke.astype(int))

    # with open('all_strokes3.npy', 'w+b') as f:
    #  np.save(f,gen_stroke.astype(int))
    # print(gen_stroke)
    f.close()

    # launch the drawing on the screen
    p = Popen(["python drawing_robot.py --predict_file " + file_name_prediction_path], shell=True)

