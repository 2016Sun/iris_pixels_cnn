# Take a lot of inspiration from http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy

def build_model():
    model = Sequential()
    model.add(Dense(400, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == '__main__':

    seed = 42
    numpy.random.seed(seed)

    # Load the iris dataset
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    # convert integers to dummy variables (i.e. one hot encoded)
    Y = np_utils.to_categorical(encoded_Y)
    
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    estimator = KerasClassifier(build_fn=build_model, nb_epoch=2000, batch_size=50, verbose=0)

    results = cross_val_score(estimator, X, Y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
