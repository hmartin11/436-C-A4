import numpy as np
import pickle 

def parse_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # We don't use these but I left them in as a useful template for future development
    parser.add_argument("--copy_X",        type=bool, default=True)
    parser.add_argument("--fit_intercept", type=bool, default=True)
    
    # Data directories
    # In Azure, use Azure ML Datasets or paths in Azure Blob Storage
    # Here, we assume the paths are passed directly as arguments
    parser.add_argument("--train", type=str, default=os.environ.get("AZUREML_DATAREFERENCE_train"))
    parser.add_argument("--test", type=str, default=os.environ.get("AZUREML_DATAREFERENCE_test"))

    # Model directory: in Azure, typically set to './outputs'
    # Model artifacts saved here can be registered to Azure ML workspace after training
    parser.add_argument("--model_dir", type=str, default='./outputs')

    return parser.parse_known_args()




# CODE HELPER 1 - notice how data is saved -> training.cnn format
def pickleTrainingData():
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    train_data = np.empty((0, 32*32*3))
    train_labels = []

    for i in range(1, 2):
        fileNameDataBatch = './cifar-10-batches-py/data_batch_' + str(i)
        batch = unpickle(fileNameDataBatch)
        train_data = np.vstack((train_data, batch[b'data']))
        train_labels += batch[b'labels']

    train_labels = np.array(train_labels)
    train_data = train_data.reshape(-1, 32, 32, 3) / 255.0
    
    # !!!!! NOTICE HOW THE DATA IS SAVED !!!!!!
    # Will be returned in form of:
    # train_label, train_data  = getDataBack()
    pickle.dump([train_labels,train_data], open('./train.cnn', 'wb'))



# CODE HELPER 2
def getTestData():
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    fileNameTestBatch = './cifar-10-batches-py/test_batch'
    test_data = unpickle(fileNameTestBatch)[b'data']
    test_data = test_data.reshape(-1, 32, 32, 3) / 255.0
    test_labels = np.array(unpickle(fileNameTestBatch)[b'labels'])
    
    num_samples_to_select = 600
    random_indices = np.random.choice(test_data.shape[0], num_samples_to_select, replace=False)
    selected_test_data = test_data[random_indices]
    selected_test_labels = test_labels[random_indices]
    
    return selected_test_data, selected_test_labels

def load_dataset(path):
    """
    Load entire dataset.
    """
    # Find all files with a pickle ext but we only load the first one in this sample:
    files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".cnn")]

    if len(files) == 0:
        raise ValueError("Invalid # of files in dir: {}".format(path))
    
    [X, y] = pickle.load(open(files[0], 'rb'))
    
    return X, y


# CODE HELPER 3
from sklearn.metrics import accuracy_score
def getAccuracyOfPrediction(cnn_predictions, test_labels):
    cnn_predicted_labels = np.argmax(cnn_predictions, axis=1)
    accuracy = accuracy_score(test_labels, cnn_predicted_labels)
    print("Accuracy:", accuracy)


# CODE HELPER 4
import argparse
import os
import pickle
import subprocess
#subprocess.run(["pip", "install", "Werkzeug==2.0.3"])
#subprocess.run(["pip", "install", "tensorflow==2.4"])
import tensorflow as tf
from tensorflow import keras

if __name__ == "__main__":
    
    args, _ = parse_args()
    
    train_labels, train_data = load_dataset(args.train)

    train_data = train_data.reshape(-1, 32, 32, 3) / 255.0
    
    #train_data, train_labels = pickleTrainingData()
    
    hyperparameters = {
            "copy_X": args.copy_X,
            "fit_intercept": args.fit_intercept
        }

    

    # CODE HELPER 5 -> notice that the file name ends in .cnn
    #files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".cnn")]

    # CODE HELPER 6
    #loaded_model = tf.keras.models.load_model(os.path.join(args.model_dir, "modelCNN"))

    # CODE HELPER 7
    model = keras.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(units=10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels, epochs=3, batch_size=32, validation_split=0.1)

    model.save(os.path.join(args.model_dir, "modelCNN"))
    
    pickle.dump(model, open(os.path.join(args.model_dir, "model.pkl"), 'wb'))
    print("Model saved in: " + os.path.join(args.model_dir, "model.pkl"))