import kivy
kivy.require("1.10.0") 
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.gridlayout import GridLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.pagelayout import PageLayout
from kivy.base import runTouchApp
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.properties import NumericProperty
import json
from urllib.request import urlopen
import urllib.request
# import serial
import numpy  
import matplotlib.pyplot as plt 
from preprocessing import take_photos , prepare_image_train, prepare_image_test, prepare_image_val
from train_model_v2 import train_model, test_acc
from img_model_v1 import img_model
# from drawnow import *

# arduinoData = serial.Serial('com10', 9600)

class ScreenOne(Screen):
    pass
    
    	
class ScreenTwo(Screen):

    def create_dataset(self):
        take_photos()

            
    
class ScreenThree(Screen):
   
    def preprocessing_img(self):
        self.x_train, self.y_train = prepare_image_train()
        print(self.x_train.shape)
        print(self.y_train.shape)
        print("$$$$$$$$$$$$$$$$$$$")
        self.x_val, self.y_val = prepare_image_val()
        print(self.x_val.shape)
        print(self.y_val.shape)
        print("$$$$$$$$$$$$$$$$$$$")
        self.x_test, self.y_test = prepare_image_test()
        print(self.x_test.shape)
        print(self.y_test.shape)
        print("$$$$$$$$$$$$$$$$$$$")


   
class ScreenFour(Screen):
    def Train_Model(self):
        train_model()
    def Test_Acc(self):
        test_acc()
class ScreenFive(Screen):
    def output(self):
        img_model()

        
        
#     def s1(self):
#         arduinoData.write('a'.encode('ascii'))
#         self.ss.text="scenario 1 en cours.."
#         self.sss.text=" "
#         self.ssss.text=" "
#     def s2(self):
#         arduinoData.write('b'.encode('ascii'))
#         self.sss.text="scenario 2 en cours.."
#         self.ss.text=" "
#         self.ssss.text=" "
#     def s3(self):
#         arduinoData.write('c'.encode('ascii'))
#         self.ssss.text="scenario 3 en cours.."
#         self.sss.text=" "
#         self.ss.text=" "
class ScreenManagement(ScreenManager):
    pass

# my_frame = Builder.load_file("main.kv")
my_frame = Builder.load_file("test.kv")

class mainApp(App):
    def build(self):
        return my_frame
 
mainApp().run()