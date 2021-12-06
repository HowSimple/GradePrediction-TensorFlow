import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

class GradePredictionNetwork:
    def __init__(self, inputFile, epochs, input_layer_size):
        np.set_printoptions(precision=3, suppress=True)
        np.set_printoptions(threshold=np.inf)
        self.load_input(inputFile)
        self.prepare_datasets()

        math_model = self.train(epochs, input_layer_size, "math score")
        writing_model = self.train(epochs, input_layer_size, "writing score")
        reading_model = self.train(epochs, input_layer_size, "reading score")

        self.evaluate_models([math_model, writing_model,reading_model])



    def load_input(self, inputPath):
        self.student_data = pd.read_csv(inputPath)


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
        self.train_features = self.train_features.drop(columns=label_columns)

        self.test_labels = self.test_features[label_columns]
        self.test_features = self.test_features.drop(columns=label_columns)
        self.valid_labels = self.valid_features[label_columns]
        self.valid_features = self.valid_features.drop(columns=label_columns)




    def train(self, epochs, input_layer_size, target):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(input_layer_size,  activation="relu"),
            tf.keras.layers.Dense(10, activation="relu"),

            tf.keras.layers.Dense(1, activation="relu")

        ])
        model._name = target.replace(" ", "_")
        model.compile(optimizer='adam', loss='mae', metrics=['mse'])
        history = model.fit(self.train_features,
                                 self.train_labels[target],

                                 epochs=epochs,
                                      batch_size=64,
                                 # Suppress logging.
                                 verbose=0,
                                # callbacks=a
                                 validation_data= (self.test_features, self.test_labels)
                            )
        return model, history


    def evaluate_models(self, models):
        plt.figure(figsize=(10, 8))
        #figures,axis= plt.subplots(1,3)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy", fontsize=16)
        optimizer = models[0][1].model.optimizer._name
        layers = len(models[0][0].layers) - 2
        loss = models[0][1].model.loss.upper()
        metrics = models[0][1].model.metrics

        title = "Hidden layers: {}, target variables: {}   loss fn: {} optimizer: {}  ".format(layers,self.train_labels.shape[1]-2, loss,optimizer, )

        for model in models:
            target_column = model[0].name.replace("_"," ")
            print(model[0].evaluate(self.test_features, self.test_labels[target_column], verbose=2))
            test_loss, test_metric  = model[0].evaluate(self.test_features, self.test_labels[target_column], verbose=2)
            valid_loss, valid_metric = model[0].evaluate(self.valid_features, self.valid_labels[target_column], verbose=2)

            plt.plot(model[1].history['loss'], label='train: {} '.format(target_column))
            plt.plot(model[1].history['val_loss'],  label='valid: {} '.format(target_column))

            valid_results = '\n{}   {}: {} {}:{}'.format(target_column, loss, "{:.2f}".format(valid_loss),metrics[1]._name, "{:.2f}".format(valid_metric))



            title = title + valid_results

        plt.title(title)
        plt.legend(loc="upper right")
        plt.show()


if __name__ == '__main__':
    print('Starting...')

    GradePredictionNetwork("StudentsPerformance.csv",100,17)
