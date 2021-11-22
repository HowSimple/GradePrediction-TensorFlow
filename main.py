import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers

class GradePredictionNetwork:
    def __init__(self, inputFile, gaussians, epochs):
        np.set_printoptions(precision=3, suppress=True)
        np.set_printoptions(threshold=np.inf)


        self.load_input(inputFile)
        self.prepare_datasets()
        #self.model()

        #self.train( epochs)
    def load_input(self, inputPath):
        self.student_data = pd.read_csv(inputPath)

        self.column_names = ["gender"
            , "race/ethnicity", "parental level of education", "lunch"
            , "test preparation course", "math score", "reading score"
            , "writing score"]
        self.gradeLabels = ["math score", "reading score"
            , "writing score"]




    def prepare_datasets(self):

        feature_columns = ["gender"
            , "race/ethnicity", "parental level of education", "lunch"
            , "test preparation course"]
        #self.student_data = pd.get_dummies(self.student_data, columns=feature_columns, prefix='', prefix_sep='')
        self.train_dataset, self.test_dataset, self.valid_dataset = np.split(self.student_data.sample(frac=1),
                                           [int(0.8 * len(self.student_data)), int(0.9 * len(self.student_data))])

        features = self.student_data.drop(columns=self.gradeLabels, inplace=False)
        # one hot encoding
        features = pd.get_dummies(features, columns=feature_columns, prefix='', prefix_sep='')
        grades = self.student_data.drop(columns=feature_columns, inplace=False)

        #print(features.head())
        #print(grades.head())

        #self.train_dataset = self.student_data.sample(frac=0.7, random_state=0)
        #self.train_dataset = pd.get_dummies(self.train_dataset, columns=feature_columns, prefix='', prefix_sep='')
        #self.train_grades = self.train_dataset.drop(columns=feature_columns, inplace=False)
        #self.train_labels = self.train_dataset['writing grade']
        #self.train_features = pd.get_dummies(features, columns=feature_columns, prefix='', prefix_sep='')
        #self.test_dataset = self.train_dataset.drop(self.train_dataset.index)
        #self.test_dataset = self.test_dataset.sample(frac=0.5, random_state=0)
       #self.valid_dataset = self.test_dataset.drop(self.test_dataset.index)

        print("Train:",self.train_dataset.head())
        print("Test:",self.test_dataset.head())
    #    print("Valid:", self.valid_dataset.head())
        #self.train_labels = self.train_dataset.drop(columns=,axis=1,inplace=False)

        self.train_dataset = self.train_dataset.drop(columns=['math score','reading score','writing score'])


        #TODO: add other grade cols


      #  self.test_dataset = self.student_data.drop(self.train_dataset.index)

    def model(self):
        print(self.train_dataset.shape[1:])
        print(self.train_dataset.head())
        self.model = tf.keras.models.Sequential([
    #tf.keras.layers.Dense(80, activation="relu", input_shape=tf.self.train_dat.shape[1:]),
    tf.keras.layers.Dense(80, activation="relu", input_shape=np.array(self.train_dataset.shape[1:])),
    tf.keras.layers.Dense(1)
])




    def model2(self):
        grades = np.array(self.test_dataset['writing score'])
        features = self.test_dataset.drop(columns=['math score','reading score','writing score'])
        normalizer = layers.Normalization(input_shape=[17, ], axis=None)
        normalized_dataset = normalizer.adapt(np.array(features))
        input_layer = tf.keras.Input(shape=(normalized_dataset.shape[1],))
        dense_layer_1 = tf.Dense(17, activation='relu')(input_layer)
        output_layer = tf.Dense(grades.shape[1])(dense_layer_1)

        model = tf.Model(inputs=input_layer, outputs=output_layer)
        self.model = model



        #model = tf.keras.Sequential()
        #model.add(normalizer, layers.Dense(units=1))
        #self.model = tf.keras.Sequential([
        #    normalizer, layers.Dense(units=1)
        #])
        #self.model = model
        print(self.model.summary())

    def predict(self):
        self.writing_grades = np.array(self.test_dataset['writing score'])
        print(self.model.predict(self.writing_grades[:10]))

    def train(self, epochs):
        self.model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])


        history = self.model.fit(self.train_dataset, self.train_labels, epochs=epochs,
                            )


    def train2(self, target_col, epochs):

        self.model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])


        pd.set_option('display.max_rows', len(self.test_dataset))
        print(self.test_dataset.head())
        print(self.model.trainable_variables)
        history = self.model.fit(
            np.array(self.test_dataset['writing score']),
            self.test_dataset.pop('writing score'),
            epochs=epochs,
            #Suppress logging.
            verbose=0,
            # Calculate validation results on 20% of training data
            validation_split= 0.2
        )
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        print(hist.tail())
        #print(history)
        #return history





    #TODO:
    def plot_error(self, history, label):
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.ylim([0, 10])
        plt.xlabel('Epoch')
        plt.ylabel('Error [%a]'.format(label))
        plt.legend()
        plt.grid(True)



if __name__ == '__main__':
    print('Starting...')

    GradePredictionNetwork("StudentsPerformance.csv", 1,100)
