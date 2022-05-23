import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


class GradePredictionNetwork:
    def __init__(self, input_file, target_columns, epochs, input_layer_size):
        np.set_printoptions(precision=3, suppress=True)
        np.set_printoptions(threshold=np.inf)
        student_dataset = pd.read_csv(input_file)
        self.prepare_datasets(student_dataset, target_columns)

        math_model = self.train(epochs, input_layer_size, target_columns[0])
        writing_model = self.train(epochs, input_layer_size, target_columns[1])
        reading_model = self.train(epochs, input_layer_size, target_columns[2])

        self.evaluate_models([math_model, writing_model, reading_model])

    def prepare_datasets(self, dataset, target_columns):
        feature_columns = dataset.columns.drop(target_columns)
        dataset = pd.get_dummies(dataset, columns=feature_columns, prefix='', prefix_sep='')
        self.train_features, self.test_features, self.valid_features = np.split(dataset.sample(frac=1),
                                                                                [int(0.8 * len(dataset)),
                                                                                 int(0.9 * len(dataset))])
        self.train_labels = self.train_features[target_columns]
        self.train_features = self.train_features.drop(columns=target_columns)
        self.test_labels = self.test_features[target_columns]
        self.test_features = self.test_features.drop(columns=target_columns)
        self.valid_labels = self.valid_features[target_columns]
        self.valid_features = self.valid_features.drop(columns=target_columns)

    def train(self, epochs, input_layer_size, target):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(input_layer_size, activation="relu"),
            tf.keras.layers.Dense(10, activation="relu"),

            tf.keras.layers.Dense(1, activation="relu")

        ])
        model._name = target.replace(" ", "_")
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        history = model.fit(self.train_features,
                            self.train_labels[target],
                            epochs=epochs,
                            batch_size=64,
                            # Suppress logging.
                            verbose=0,
                            # callbacks=a
                            validation_data=(self.test_features, self.test_labels)
                            )
        return model, history

    def evaluate_models(self, models):
        plt.figure(figsize=(10, 8))
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy", fontsize=16)
        optimizer = models[0][1].model.optimizer._name
        layers = len(models[0][0].layers) - 2
        loss = models[0][1].model.loss.upper()
        metrics = models[0][1].model.metrics
        title = "Hidden layers: {}, target variables: {} x 3   loss fn: {} optimizer: {}  ".format(layers,
                                                                                               self.train_labels.shape[
                                                                                                   1] - 2, loss,
                                                                                               optimizer, )

        for model in models:
            target_column = model[0].name.replace("_", " ")
            print(model[0].evaluate(self.test_features, self.test_labels[target_column], verbose=2))
            test_loss, test_metric = model[0].evaluate(self.test_features, self.test_labels[target_column], verbose=2)
            valid_loss, valid_metric = model[0].evaluate(self.valid_features, self.valid_labels[target_column],
                                                         verbose=2)

            plt.plot(model[1].history['loss'], label='train: {} '.format(target_column))
            plt.plot(model[1].history['val_loss'], label='valid: {} '.format(target_column))

            valid_results = '\n{}   {}: {} {}:{}'.format(target_column, loss, "{:.2f}".format(valid_loss),
                                                         metrics[1]._name, "{:.2f}".format(valid_metric))

            print("Average Error: %.3f" % abs(
                self.test_labels['math score'].to_list() - model[0].predict(self.test_features)).mean())

            title = title + valid_results

        plt.title(title)
        plt.legend(loc="upper right")
        plt.savefig('results.png')
        plt.show()


if __name__ == '__main__':
    print('Starting...')

    GradePredictionNetwork(input_file="StudentsPerformance.csv", target_columns=["math score", "reading score"
        , "writing score"], epochs=100, input_layer_size=17)
