import numpy as np
import matplotlib.pyplot as plt
from os import makedirs
from os.path import join, exists, expanduser
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input,decode_predictions

import sys
from sklearn import svm
from keras.layers import Flatten, MaxPooling2D, Input, AveragePooling2D
from keras.models import Model
import os
import imageio as imgio

# To run python path_to_training_from_directory_in_which_this_script_is_into path_to_test_from_directory_in_which_this_script_is_into

# Main class, with initialisation
class TransfearLearning:


    def __init__(self, train_path = '/training/'  , test_path = '/testing/' ):
        self.model = Resnet() #Load the model into memory
        # set up where the training and testing data is located, by default look into currentpath + /training for
        # training data and + /testing for testing data. 
        self.global_path = os.getcwd()
        self.train_path = train_path
        self.test_path = test_path
        self.save_type = '.jpg'
        self.categories = {}
        # initialising the generator, only augmenting with a horizontal flip, shear, rotation and zoom
        self.train_datagen = image.ImageDataGenerator(horizontal_flip=True , shear_range=0.3, rotation_range=10, zoom_range=0.3 ,  dtype='uint8')
        self.classes = []

    # function used to turn a number to its string equivalant
    def to_name(self,number):
            return self.categories[number]

    # after class has been created start the program with initialise
    def initialise(self):

        print ('Generating feature vectors')
        self.max_item_counter = ((n_images/15) -1)
        # initialise a flow from directory generator, it will automaticaly get the classes as folders
        self.train_generator = self.train_datagen.flow_from_directory(
        self.global_path + self.train_path,
        target_size=(224, 224),
        batch_size=1,
        shuffle=True,
        color_mode='rgb')

        pre_categories = self.train_generator.class_indices

        self.categories = {}
        for i,j in pre_categories.items():
            self.categories[j] = i

        print (self.categories)
        # parse 3000 images throu the network, capture the feature vectors for future use
        lable_preds = self.model.resnet.predict_generator(self.generator_train_c(), steps=20000, workers=0)

        classes_temp = np.zeros(len(self.classes))
        for i in range(len(self.classes)):
            classes_temp[i] = self.classes[i]
        self.classes = classes_temp

        print ('All the classes',self.classes)

        print ('Finding support vectors')

        

        # use the featue vectors to train the svm classifier
        clf = self.svm_implementation(lable_preds)

        # At this point the model has been built and classifier trained, load the test samples and generate
        # the feature vectors just like the trianing case
        pred_test = self.model.resnet.predict_generator(self.generator_test_c(), steps=2985)
        # use the feture vectors to predict which claass they belong too.
        prediction_svc_test = clf.predict(pred_test)

        # to save labels in a text file for hand in
        self.save_lables(prediction_svc_test, 0, 2985, save = False)


    # svm training, takes the feture vectors and outputs a svm model
    def svm_implementation(self, feature_vectors):
        x = feature_vectors
        y = self.classes
        print ('x shape:', x.shape)
        print ('y shape:',y.shape)
        clf = svm.SVC( decision_function_shape='ovo', kernel='rbf')
        clf.fit(x, y) 
        return clf 

    # Training generator, it automaticaly chooses at random one image from the training dataset
    # and returns it to train the model
    def generator_train_c(self):
        while True:
            x, y = self.train_generator.next()   
            print ('the shape of us',x.shape, y.shape, self.to_name(np.argmax(y)))
            self.classes.append(np.argmax(y))
            yield x, y
    # Test generator, it opens the testing folder and outputs all images one by one for testing
    # as it iterates for 2985 times and as some pictures missing the loop breaks,
    # has to skip an iteration on those numbers 
    def generator_test_c(self):
        item_counter = 0
        while True: 
            if item_counter == 1314 or item_counter == 2938 or item_counter == 2962:
                item_counter +=1

            load_str = self.global_path + self.test_path + str(item_counter)+ self.save_type
            img = image.load_img(load_str, target_size = (224,224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            item_counter +=1

            yield img,None

    def save_lables(self,prediction_svc_test, start, to, save= False):

        if save:
            f = open('Lables.txt', 'w')
            count = 0
            for i in range(start, to):
                if i == 1314 or i == 2938 or i == 2962:
                    count +=1
                intuition = str(count) +self.save_type + ' ' + self.to_name(prediction_svc_test[i]) + '\n'
                count +=1
                f.write(intuition)
        else:
            print ('enter True to save lables')


class Resnet:
    def __init__(self):
        self.create_network()
        #load the resnet50 network, remove the last layer, load the imagenet weights, averagepool the 
        # last layer to reduce the dimensionality and flatten it to recive a 2048 size feature vector
    def create_network(self):
        resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))
        x = resnet.output
        x = AveragePooling2D((7, 7))(x)
        x = Flatten()(x)
        self.resnet = Model(resnet.input, x)

        
if __name__ == "__main__":
    TransfearLearning(sys.argv[1], sys.argv[2])
    TransfearLearning.initialise()

