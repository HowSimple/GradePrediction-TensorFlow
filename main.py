import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers



class GradePredictionNetwork:


    def __init__(self, inputFile, gaussian):
        self.inputFile = inputFile
        print("TensorFlow version:", tf.__version__)
        self.prepare_datasets()

    def prepare_datasets(self):
        column_names = ["gender"
            , "race/ethnicity", "parental level of education", "lunch"
            , "test preparation course", "math score", "reading score"
            , "writing score"]
        feature_columns = ["gender"
            , "race/ethnicity", "parental level of education", "lunch"
            , "test preparation course"]
        target_columns = ["math score", "reading score"
            , "writing score"]

        np.set_printoptions(precision=3, suppress=True)


        student_data = pd.read_csv(self.inputFile)
        print("Initial input data:")
        print(student_data.head())

        student_data = student_data.drop(columns=target_columns)
        print("Drop grade scores:")
        print(student_data.head())


        student_data = pd.get_dummies(student_data, columns=feature_columns, prefix='', prefix_sep='')
        print("One hot encoded:")
        print(student_data.head())


        train_dataset = student_data.sample(frac=0.8, random_state=0)
        test_dataset = student_data.drop(train_dataset.index)
        print("Training data", train_dataset)
        print("Test data", test_dataset)

        #split datasets
        # train, validation, test = np.split(student_data.sample(frac=1) , [int(0.8 * len(student_data)), int(0.9 * len(student_data))])


        #self.train = self.df_to_dataset(train, batch_size=5)



    #    train_ds = self.df_to_dataset(train, batch_size=5)

     #   [(train_features, label_batch)] = train_ds.take(1)
       # print('Every feature:', list(train_features.keys()))
       # print('A batch of ages:', train_features['Age'])
       # print('A batch of targets:', label_batch)
        # grades_features = grades_train.copy()
        # self.grades_features = np.array(grades_features)





if __name__ == '__main__':
    print('Starting...')

    GradePredictionNetwork("StudentsPerformance.csv", 1)
