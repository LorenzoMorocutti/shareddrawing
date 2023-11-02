#sheet manager class

import os
os.environ["KIVY_NO_ARGS"] = "1"

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Line, Color, Ellipse, Canvas, Translate, Fbo, ClearColor, ClearBuffers, Scale
import kivy.graphics
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.interactive import InteractiveLauncher
from kivy.core.image import Image
import time
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
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

from six.moves import xrange

# libraries required for visualisation:
from IPython.display import SVG, display
import PIL
from PIL import Image, ImageGrab
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

from drawing_robot import robot_drawing


strokes_count = 0
human = True

tf.compat.v1.disable_eager_execution()

class Printer():

    def __init__(self):

        self.last_point = [0.0, 0.0]

        self.x = []
        self.y = []

        self.total_strokes_x = []
        self.total_strokes_y = []

        self.human_strokes_x = []
        self.human_strokes_y = []

        self.robot_strokes_x = []
        self.robot_strokes_y = []


        self.trasp_x = []
        self.trasp_y = []
        self.trasp_xy = []
        self.trasp_rdp = []
        self.trasp_rdp_stroke3 = []

        self.trasp_total_x = []
        self.trasp_total_y = []
        self.trasp_total_xy = []
        self.trasp_total_rdp = []
        self.trasp_total_rdp_stroke3 = []

        self.trasp_human_x = []
        self.trasp_human_y = []
        self.trasp_human_xy = []
        self.trasp_human_rdp = []
        self.trasp_human_rdp_stroke3 = []

        self.trasp_robot_x = []
        self.trasp_robot_y = []
        self.trasp_robot_xy = []
        self.trasp_robot_rdp = []
        self.trasp_robot_rdp_stroke3 = []


        self.rdp_x = []
        self.rdp_y = []

        self.rdp_total_x = []
        self.rdp_total_y = []

        self.rdp_human_x = []
        self.rdp_human_y = []

        self.rdp_robot_x = []
        self.rdp_robot_y = []

        self.copy_strokes_total = np.array([[]])
        self.just_first_stroke = True


        self.screen_height = 1080
        self.screen_width = 1920
        self.carlo_screen_w = 0.294
        self.carlo_screen_h = 0.165

        self.rec = "[" + str(int(self.screen_width/3)) + "," + str(int(self.screen_width/2)) + "]"
        self.alpha = 45*pi/100
        self.SCREEN_LEFT_BOTTOM_CORNER_POS_X_ICUB_REF = -0.3
        self.SCREEN_LEFT_BOTTOM_CORNER_POS_Y_ICUB_REF = -0.1
        self.SCREEN_LEFT_BOTTOM_CORNER_POS_Z_ICUB_REF = -0.0
        self.SECURITY_MARGIN = 0.1
        self.LEFT_MARGIN = 0

        self.human_turn = False
        #self.robotPrinter = MyPaintWidget()

        #self.list = []

    def appending_x(self, touch_x):
        self.x.append(touch_x)
        self.total_strokes_x.append(touch_x)

        if human == True:
            self.human_strokes_x.append(touch_x)
        else:
            self.robot_strokes_x.append(touch_x)

        return

    def appending_y(self, touch_y):
        self.y.append(touch_y)
        self.total_strokes_y.append(touch_y)

        if human == True:
            self.human_strokes_y.append(touch_y)
        else:
            self.robot_strokes_y.append(touch_y)

        return

    def saving(self):
        global human




        #transposed of the lists to be able to use the rdp algorithm

        self.trasp_x = np.array(self.x).reshape(len(self.x), 1)
        self.trasp_y = np.array(self.y).reshape(len(self.y), 1)
        self.trasp_xy = np.hstack((self.trasp_x, self.trasp_y))

        self.trasp_total_x = np.array(self.total_strokes_x).reshape(len(self.total_strokes_x), 1)
        self.trasp_total_y = np.array(self.total_strokes_y).reshape(len(self.total_strokes_y), 1)
        self.trasp_total_xy = np.hstack((self.trasp_total_x, self.trasp_total_y))

        self.trasp_human_x = np.array(self.human_strokes_x).reshape(len(self.human_strokes_x), 1)
        self.trasp_human_y = np.array(self.human_strokes_y).reshape(len(self.human_strokes_y), 1)
        self.trasp_human_xy = np.hstack((self.trasp_human_x, self.trasp_human_y))

        self.trasp_robot_x = np.array(self.robot_strokes_x).reshape(len(self.robot_strokes_x), 1)
        self.trasp_robot_y = np.array(self.robot_strokes_y).reshape(len(self.robot_strokes_y), 1)
        self.trasp_robot_xy = np.hstack((self.trasp_robot_x, self.trasp_robot_y))

        #this reduces the number of points representative of the line drawn
        self.trasp_rdp = rdp(self.trasp_xy, epsilon=2.0)
        self.trasp_total_rdp = rdp(self.trasp_total_xy, epsilon=2.0)
        self.trasp_human_rdp = rdp(self.trasp_human_xy, epsilon=2.0)
        self.trasp_robot_rdp = rdp(self.trasp_robot_xy, epsilon=2.0)

        j, l = np.shape(self.trasp_rdp)
        print("queti sono j e l", j, l)
        self.trasp_rdp_stroke3 = np.zeros((j-1, l+1))
        self.trasp_rdp_stroke3[:, 0:2] = self.trasp_rdp[1:, 0:2] - self.trasp_rdp[:-1, 0:2]
        self.trasp_rdp_stroke3[j-2, l] = 1.0

        self.trasp_total_rdp_stroke3 = self.trasp_total_rdp[1:, 0:2] - self.trasp_total_rdp[:-1, 0:2]
        self.trasp_human_rdp_stroke3 = self.trasp_human_rdp[1:, 0:2] - self.trasp_human_rdp[:-1, 0:2]
        self.trasp_robot_rdp_stroke3 = self.trasp_robot_rdp[1:, 0:2] - self.trasp_robot_rdp[:-1, 0:2]

        self.rdp_x = list(self.trasp_rdp[:,0])
        self.rdp_y = list(self.trasp_rdp[:,1])

        self.rdp_total_x = list(self.trasp_total_rdp[:, 0])
        self.rdp_total_y = list(self.trasp_total_rdp[:, 1])

        self.rdp_human_x = list(self.trasp_human_rdp[:, 0])
        self.rdp_human_y = list(self.trasp_human_rdp[:, 1])

        self.rdp_robot_x = list(self.trasp_robot_rdp[:, 0])
        self.rdp_robot_y = list(self.trasp_robot_rdp[:, 1])

        k = len(self.trasp_rdp)
        self.last_point[0] = int(self.rdp_x[k - 1])
        self.last_point[1] = int(self.rdp_y[k - 1])

        print("try to classify")

        #printer.classify_last_stroke()

        print("first_h: ",human)

        if self.human_turn == False:
            self.human_turn = True

            printer.classify_all_strokes()

            if printer.predict():
                print("I predicted a new stroke and I can go on with the drawing")
                #printer.draw_robot()
                print(self.last_point[0])
                print(self.last_point[1])

                # my_text = '[' + str(self.last_point[0]) + ',' + str(self.last_point[1]) + ']'
                # print(my_text)
                # argument = 'drawing_robot.py --pt_coord=' + my_text + " --predict_file " + '/usr/local/src/robot/cognitiveinteraction/drawing_update/' + "strokes_predict.npy"
                #
                # p = Popen(["python " + argument], shell=True)

                draw_class = robot_drawing()
                draw_class.draw_robot(self.last_point[0], self.last_point[1])

        return


    def classify_last_stroke(self):

        my_text = str([self.rdp_x, self.rdp_y])
        my_text = '\"[' + my_text + ']\"'

        argument = '/usr/local/src/robot/cognitiveinteraction/container/IRCN-IIT/Classifier/classify-all_classes.py --steps 150000 --model_dir ' + FLAGS.class_model_path + ' --classes_file ' + FLAGS.TFRecord_path + 'training.tfrecord.classes --training_data ' + FLAGS.TFRecord_path + 'training.tfrecord-?????-of-????? --eval_data ' + FLAGS.TFRecord_path + 'eval.tfrecord-?????-of-????? --predict_for_data ' + my_text + ' --classify_output_file ' + str(
            '/usr/local/src/robot/cognitiveinteraction/drawing_update/' + FLAGS.classify_output_file)

        # one space after python cmd is necessary
        p = Popen(["python " + argument], shell=True)
        p.wait()
        print("!!!The Classifier has given its interpretation!!!")

        self.x.clear()
        self.y.clear()

        return

    def classify_all_strokes(self):

        my_text = str([self.rdp_total_x, self.rdp_total_y])
        my_text = '\"[' + my_text + ']\"'
        print(my_text)

        argument = '/usr/local/src/robot/cognitiveinteraction/container/IRCN-IIT/Classifier/classify-all_classes.py --steps 150000 --model_dir ' + FLAGS.class_model_path + ' --classes_file ' + FLAGS.TFRecord_path + 'training.tfrecord.classes --training_data ' + FLAGS.TFRecord_path + 'training.tfrecord-?????-of-????? --eval_data ' + FLAGS.TFRecord_path + 'eval.tfrecord-?????-of-????? --predict_for_data ' + my_text + ' --classify_output_file ' + str(
            '/usr/local/src/robot/cognitiveinteraction/drawing_update/' + FLAGS.classify_output_file)

        # one space after python cmd is necessary
        p = Popen(["python " + argument], shell=True)
        p.wait()
        print("!!!The Classifier has given its interpretation!!!")

        self.x.clear()
        self.y.clear()

        return


    def classify_human_strokes(self):

        my_text = str([self.rdp_human_x, self.rdp_human_y])
        my_text = '\"[' + my_text + ']\"'

        argument = '/usr/local/src/robot/cognitiveinteraction/container/IRCN-IIT/Classifier/classify-all_classes.py --steps 150000 --model_dir ' + FLAGS.class_model_path + ' --classes_file ' + FLAGS.TFRecord_path + 'training.tfrecord.classes --training_data ' + FLAGS.TFRecord_path + 'training.tfrecord-?????-of-????? --eval_data ' + FLAGS.TFRecord_path + 'eval.tfrecord-?????-of-????? --predict_for_data ' + my_text + ' --classify_output_file ' + str(
            '/usr/local/src/robot/cognitiveinteraction/drawing_update/' + FLAGS.classify_output_file)

        # one space after python cmd is necessary
        p = Popen(["python " + argument], shell=True)
        p.wait()
        print("!!!The Classifier has given its interpretation!!!")

        self.x.clear()
        self.y.clear()

        return


    def classify_robot_strokes(self):

        my_text = str([self.rdp_robot_x, self.rdp_robot_y])
        my_text = '\"[' + my_text + ']\"'

        argument = '/usr/local/src/robot/cognitiveinteraction/container/IRCN-IIT/Classifier/classify-all_classes.py --steps 150000 --model_dir ' + FLAGS.class_model_path + ' --classes_file ' + FLAGS.TFRecord_path + 'training.tfrecord.classes --training_data ' + FLAGS.TFRecord_path + 'training.tfrecord-?????-of-????? --eval_data ' + FLAGS.TFRecord_path + 'eval.tfrecord-?????-of-????? --predict_for_data ' + my_text + ' --classify_output_file ' + str(
            '/usr/local/src/robot/cognitiveinteraction/drawing_update/' + FLAGS.classify_output_file)

        # one space after python cmd is necessary
        p = Popen(["python " + argument], shell=True)
        p.wait()
        print("!!!The Classifier has given its interpretation!!!")

        self.x.clear()
        self.y.clear()

        return

    ######################################################
    #           PREDICTION SECTION
    ######################################################



    def predict(self):

        userDictionary, choice, value = printer.bar_chart()
        #def predict(model_dir_uncond, repo_saved_data, who):

        model_dir_uncond = FLAGS.single_model_path + str(choice) + '_unc/'
        print("model_dir_uncond: ", model_dir_uncond)
        repo_saved_data = '/usr/local/src/robot/cognitiveinteraction/drawing_update/'
        data_dir = '/usr/local/src/robot/cognitiveinteraction/container/12_categories/12_categories_NPZ/'
        models_root_dir = '/tmp/sketch_rnn/models'
        rows_start = []


        [train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = printer.load_env_compatible(data_dir, model_dir_uncond)

        # start session:
        reset_graph()
        model = Model(hps_model, gpu_mode=False)
        eval_model = Model(eval_hps_model, reuse=True)
        sample_model = Model(sample_hps_model, reuse=True)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        # loads the weights from checkpoint into our model
        load_checkpoint(sess, model_dir_uncond)

        # copy_strokes = self.trasp_rdp.copy()
        # previous_strokes = self.trasp_rdp.copy()

        # dimensions = np.shape(self.trasp_rdp_stroke3)
        # rows_start.append(dimensions[0])
        # copy_strokes = np.zeros((dimensions[0], dimensions[1] + 1))
        # copy_strokes[:, :-1] = self.trasp_rdp_stroke3
        # copy_strokes[dimensions[0] - 1][dimensions[1]] = 1.0


        if self.just_first_stroke:
            self.copy_strokes_total = self.trasp_rdp_stroke3
            self.just_first_stroke = False
            print("non dovrei tornarci")
        else:
            self.copy_strokes_total = np.concatenate((self.copy_strokes_total, self.trasp_rdp_stroke3), axis=0)

        std = np.std(self.trasp_rdp_stroke3[0:, 0:2])
        print("standard deviation is " + str(std))

        print("matrice trasp_rdp_Stroke3: \n", self.trasp_rdp_stroke3)

        self.trasp_rdp_stroke3[:, 0:1] /= std
        self.trasp_rdp_stroke3[:, 1:2] /= -std

        #print("copy stokes are: ")
        print("last stroke \n", self.trasp_rdp_stroke3)
        print("questo è quello che mi dà la matrice rdp: \n", self.trasp_rdp)

        print("all strokes \n", self.copy_strokes_total)

        #dict_model = printer.dict_number_strokes_start_stop_from_numpy_list_of_points(self.trasp_rdp_stroke3)
        dict_model = printer.dict_number_strokes_start_stop_from_numpy_list_of_points(self.copy_strokes_total)
        nb_tot_strokes = dict_model['nb_strokes']
        print("nb_tot_strokes: ", nb_tot_strokes)

        if nb_tot_strokes == 0:
            print("printing sample0.svg")
            printer.draw_strokes(self.trasp_rdp_stroke3, name_index=0)

        #list of strokes to compound the future next drawing step by step
        strokes_lists = []

        # Take the first stroke
        for i in range(nb_tot_strokes):
            print("for nb_tot_strokes")
            #aggregated_strokes = printer.slice_of_numpy_points(dict_model, self.trasp_rdp_stroke3, 0, i)
            aggregated_strokes = printer.slice_of_numpy_points(dict_model, self.copy_strokes_total, 0, i)
            strokes_lists.append([aggregated_strokes, [0, i]])
            where_am_i = i

        previous_pt, previous_state = remember_state(sess, sample_model, aggregated_strokes)
        generated_next_stroke, is_finished = printer.decode(sess, sample_model, previous_pt, previous_state,
                                                    temperature=0.25, draw_mode=False)

        if is_finished == 0:

            gen_stroke = np.copy(generated_next_stroke)

            gen_stroke[:, 0:1] *= std
            gen_stroke[:, 1:2] *= -std

            #previous_strokes = np.concatenate((previous_strokes, gen_stroke.astype(int)), axis=0)
            copy_strokes = np.concatenate((self.trasp_rdp_stroke3, gen_stroke.astype(int)), axis=0)

            # save prediction stroke
            with open(repo_saved_data + '/strokes_test.npy', 'wb') as f:
                #np.save(f, previous_strokes)
                np.save(f, copy_strokes)
            f.close()
            with open(repo_saved_data + '/strokes_predict.npy', 'wb') as f:
                np.save(f, gen_stroke.astype(int))
            print("generated_next_stroke\n", gen_stroke.astype(int), "\n", len(gen_stroke.astype(int)))

            f.close()
            strokes_lists.append([generated_next_stroke, [0, where_am_i + 1]])
            aggregated_strokes = np.concatenate((aggregated_strokes, generated_next_stroke), axis=0)
            strokes_lists.append([aggregated_strokes, [0, where_am_i + 1]])

            print("dovrei aver pensato a qualcosa")
            return 1

        else:
            print("I DON'T THINK OF ANYTHING TO ADD NOW, IN MY POINT OF VIEW THE DRAWING LOOKS COMPLETE")
            return 0

    def bar_chart(self):

        # print("Read the file containing the estimation of the strokes")
        # read one line from a file as a dictionary format
        file = open('/usr/local/src/robot/cognitiveinteraction/drawing_update/estimation.txt', "r")
        contents = file.read()
        Dictionary = ast.literal_eval(contents)
        file.close()
        # first mention of choice as value
        max_probability_class = max(Dictionary, key=Dictionary.get)
        print(max_probability_class)
        value_max_class = Dictionary[max_probability_class]

        plt.bar(Dictionary.keys(), Dictionary.values(), color='g')
        name_of_graph = str('/usr/local/src/robot/cognitiveinteraction/drawing_update/' + "graph_bar.png")
        plt.savefig(name_of_graph, dpi=300, bbox_inches='tight')
        # plt.show(block=False)
        # plt.pause(5)
        plt.close()
        return Dictionary, max_probability_class, value_max_class

    def load_env_compatible(self, data_dir, model_dir):
        # Loads environment for inference mode, used in jupyter notebook.
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

    def dict_number_strokes_start_stop_from_numpy_list_of_points(self, numpy_list_of_points):
        strokes = {
            "nb_strokes": 0,
            "nb_pts_by_stroke": [],
            "start": [],
            "stop": [],
        }

        for i in range(len(numpy_list_of_points)):
            if i == 0:
                strokes["start"].append(0)
            if numpy_list_of_points[i][2] == 1.0:
                strokes["nb_strokes"] += 1
                strokes["stop"].append(i)
                if i != (len(numpy_list_of_points) - 1):
                    strokes["start"].append(i + 1)

        for i in range(strokes["nb_strokes"]):
            strokes["nb_pts_by_stroke"].append(strokes["stop"][i] - strokes["start"][i] + 1)

        return strokes


    def decode(self, sess_b, sample_model, previous_pt, previous_state, z_input=None, draw_mode=True, temperature=0.25,
               factor=0.01):
        z = None
        end_of_strokes = 0
        if z_input is not None:
            z = [z_input]
        sample_strokes, end_of_strokes = next_stroke_prediction(sess_b, sample_model, previous_pt, previous_state,
                                                                seq_len=100, temperature=temperature, z=z)
        if end_of_strokes == 1:
            print("IT IS THE LAST STROKE IN MY POINT OF VIEW")
            return sample_strokes, end_of_strokes
        else:
            strokes = to_normal_strokes(sample_strokes)
            if draw_mode:
                printer.draw_strokes(strokes, factor)
            return strokes, end_of_strokes

    def draw_strokes(self, data, factor=0.2, svg_filename='/tmp/sketch_rnn/svg/sample', name_index=0):
        print("inside draw_strokes")
        svg_filename = svg_filename + str(name_index) + '.svg'
        # print(svg_filename)
        tf.gfile.MakeDirs(os.path.dirname(svg_filename))
        min_x, max_x, min_y, max_y = get_bounds(data, factor)
        dims = (50 + max_x - min_x, 50 + max_y - min_y)
        dwg = svgwrite.Drawing(svg_filename, size=dims)
        dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
        lift_pen = 1
        abs_x = 25 - min_x
        abs_y = 25 - min_y
        p = "M%s,%s " % (abs_x, abs_y)
        command = "m"
        for i in xrange(len(data)):
            if (lift_pen == 1):
                command = "m"
            elif (command != "l"):
                command = "l"
            else:
                command = ""
            x = float(data[i, 0]) / factor
            y = float(data[i, 1]) / factor
            lift_pen = data[i, 2]
            p += command + str(x) + "," + str(y) + " "
        the_color = "black"
        stroke_width = 1
        dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
        dwg.save()
        display(SVG(dwg.tostring()))

        return

    def slice_of_numpy_points(self, dict_np_list_points, np_list_points, start, stop):
        # please DO NOT TOUCH INDICE +1 here below it is necessary
        return np_list_points[dict_np_list_points["start"][start]:dict_np_list_points["stop"][stop] + 1]



class MyPaintWidget(Widget):

    Window.clearcolor = (1, 1, 1, 1)

    def on_touch_down(self, touch):
        global human

        with self.canvas:
            if human == True:
                Color(0, 0, 0)
            else:
                Color(1, 0, 0)
            touch.ud["line"] = Line(points=(touch.x, touch.y), width=1.5, cap='none', joint='round')

            #return super(MyPaintWidget, self).on_touch_down(touch)
            return

    def on_touch_move(self, touch):
        global human

        with self.canvas:
            if human == True:
                Color(0, 0, 0)
            else:
                Color(1, 0, 0)

            touch.ud["line"].points += (touch.x, touch.y)

            printer.appending_x(touch.x)
            printer.appending_y(touch.y)
            return

    def on_touch_up(self, touch):
        global strokes_count, human

        with self.canvas:
            if human == True:
                Color(0, 0, 0)
            else:
                Color(1, 0, 0)

            touch.ud["line"].points += (touch.x, touch.y)

            text = "strokes_" + str(strokes_count) + ".png"
            print(text)

            # self.export_to_png(text)

            #ImageGrab.grab().crop((65, 65, 1920, 1015)).save("/usr/local/src/robot/cognitiveinteraction/drawing_update/" + text)
            ImageGrab.grab().crop((65, 65, 2560, 1375)).save("/usr/local/src/robot/cognitiveinteraction/drawing_update/" + text)

            print(strokes_count)
            strokes_count += 1
            human = not human

            if human == False:
                printer.human_turn = False
                printer.saving()
            if human == True:
                printer.saving()
            #printer.classify()

            return



class MainApp(App):

    def build(self):

        Window.maximize()
        paint = MyPaintWidget()

        global printer, human
        #human = True
        printer = Printer()
        return paint

    def destroy(self):
        Window.close()

    def initiate(self, x):
        print(x)


def read_App():
    while True:
        print("a")
        #x_list = printer.printing()
        #print(x_list)




if __name__ == "__main__":

     # global printer, human
     # printer = Printer()
     #
     # # example of a threaded class to do something (in this case, printing the x coordinate)
     # t2 = threading.Thread(target=read_App, daemon=True)
     # t2.start()
     #
     # human = False

     parser = argparse.ArgumentParser()
     parser.register("type", "bool", lambda v: v.lower() == "false")

     # Classifier information
     parser.add_argument(
         "--TFRecord_path",
         type=str,
         default="/usr/local/src/robot/cognitiveinteraction/container/IRCN-IIT/Classifier/X6-new/TFRecord_simplified_6_new/",
         help="where is located the repository storing the TFRecords (batches of datas from each class used to train the Classifier)")
     parser.add_argument(
         "--class_model_path",
         type=str,
         default="/usr/local/src/robot/cognitiveinteraction/container/IRCN-IIT/Classifier/X6-new/model_simplified_6_new/",
         help="where is located the repository storing the trained model of the classifier for X classes")
     parser.add_argument(
         "--classify_output_file",
         type=str,
         default=str("estimation.txt"),
         help="name of file on which will be saved the resulting dictionnary ouput of the classifer all_classes and respective percentage estimation")


     # prediction information
     parser.add_argument(
         "--single_model_path",
         type=str,
         default="/usr/local/src/robot/cognitiveinteraction/container/12_unc_model/",
         help="where is located the repository storing every single trained model for prediction")
     parser.add_argument(
         "--predict_file",
         type=str,
         default="strokes_predict.npy",
         help="name of file on which will be saved the predicting points to follow to draw the next stroke on the screen")
     parser.add_argument(
         "--class_name",
         type=str,
         default="NONE",
         help="name of the class to predict first,next,last stroke")



     FLAGS, unparsed = parser.parse_known_args()

     drawing = MainApp().run()
