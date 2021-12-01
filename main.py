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

        self.train( epochs)
        self.evaluate()
        #self.predict()
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
        label_columns = ['math score','reading score','writing score']

        self.student_data = pd.get_dummies(self.student_data, columns=feature_columns, prefix='', prefix_sep='')
        self.train_features, self.test_features, self.valid_features = np.split(self.student_data.sample(frac=1),
                                           [int(0.8 * len(self.student_data)), int(0.9 * len(self.student_data))])

        self.train_labels = self.train_features[label_columns]
        #self.train_labels = self.train_features['math score']
        self.train_features = self.train_features.drop(columns=label_columns)

        #print("Train features:",self.train_features.head())
        #print("Train labels:", self.train_labels.head())
        self.test_labels = self.test_features[label_columns]
        #self.test_labels = self.test_features["math score"]

        self.test_features = self.test_features.drop(columns=label_columns)
        #print("Test features:", self.test_features.head())
        #print("Test labels:", self.test_labels.head())
        self.valid_labels = self.valid_features[label_columns]
        #self.valid_labels = self.valid_features["math score"]

        self.valid_features = self.valid_features.drop(columns=label_columns)
        #print("Valid features:", self.valid_features.head())
        #print("Valid labels:", self.valid_labels.head())









    def train(self, epochs):

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(20, activation="relu"),
            tf.keras.layers.Dense(15, activation="relu"),
            # tf.keras.layers.Dense(10, activation="relu"),
            # tf.keras.layers.Dense(5, activation="relu"),

            tf.keras.layers.Dense(self.train_labels.shape[1], activation="relu")

        ])
        self.model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

        self.history = self.model.fit(self.train_features,
                                 self.train_labels,

                                 epochs=epochs,
                                      batch_size=64,
                                 # Suppress logging.
                                 verbose=1,
                                 validation_data= (self.test_features, self.test_labels)
                            )


    def evaluate(self):
        test_loss, test_acc = self.model.evaluate(self.test_features, self.test_labels, verbose=2)
        valid_loss, valid_acc = self.model.evaluate(self.valid_features, self.valid_labels, verbose=2)




        print('\nTest set metric: {}, loss: {}'.format( test_acc, test_loss))

        print('\nValid set metric: {}, loss: {}'.format( valid_acc, valid_loss))

        #print('\nValid set error:', valid_acc)




        plt.figure(figsize=(10, 8))
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')

        plt.legend(['loss', 'val_loss'])
        plt.xlabel("")
        plt.ylabel("Mean Squared Error", fontsize=16)
        #plt.show()







if __name__ == '__main__':
    print('Starting...')

    GradePredictionNetwork("StudentsPerformance.csv", 1,500)
