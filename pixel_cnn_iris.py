# Got inspiration from https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# and from http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy
import pygal
from keras.preprocessing.image import load_img, img_to_array

IMAGE_SIZE = 64

def build_model():
    model = Sequential()
    model.add(Conv2D(128, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, kernel_initializer='normal', activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

if __name__ == '__main__':

    seed = 42
    numpy.random.seed(seed)

    # Load the iris dataset
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    # convert integers to dummy variables (i.e. one hot encoded)
    Y = np_utils.to_categorical(encoded_Y)
    X_images = []
    real_max = max([item for sublist in X for item in sublist])
    for i, x in enumerate(iris.data):
        radar_chart = pygal.Radar(fill=True)
        radar_chart.secondary_range = (0, real_max)
        radar_chart.range = (0, real_max)
        radar_chart.show_legend = False
        radar_chart.show_minor_x_labels = False
        radar_chart.show_major_x_labels = False
        radar_chart.show_minor_y_labels = False
        radar_chart.show_major_y_labels = False
        radar_chart.height = IMAGE_SIZE
        radar_chart.width = IMAGE_SIZE
        radar_chart.spacing = 0
        radar_chart.margin = 0
        radar_chart.include_x_axis = False
        radar_chart.include_y_axis = False
        radar_chart.dots_size = 1
        radar_chart.add('a record', x)
        filename = '{}.png'.format(i)
        radar_chart.render_to_png(filename)
        img = load_img(filename)
        current_x = img_to_array(img)
        print(len(current_x))
        #current_x = current_x.reshape((1,) + current_x.shape)
        #current_x = current_x.reshape(current_x.shape + (1,))
        X_images.append(current_x)
    X = numpy.array(X_images)
    #X = X_images
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    estimator = KerasClassifier(build_fn=build_model, nb_epoch=20000, batch_size=50, verbose=0)
    print(len(X))
    results = cross_val_score(estimator, X, Y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
