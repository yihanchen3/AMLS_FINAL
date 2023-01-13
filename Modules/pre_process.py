import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import dlib

# PATH TO ALL IMAGES
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# how to find frontal human faces in an image using 68 landmarks.  These
# are points on the face such as the corners of the mouth, along the
# eyebrows, on the eyes, and so forth.

# The face detector we use is made using the classic Histogram of Oriented
# Gradients (HOG) feature combined with a linear classifier, an image pyramid,
# and sliding window detection scheme.  The pose estimator was created by
# using dlib's implementation of the paper:
# One Millisecond Face Alignment with an Ensemble of Regression Trees by
# Vahid Kazemi and Josephine Sullivan, CVPR 2014
# and was trained on the iBUG 300-W face landmark Datasets (see https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
#     C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
#     300 faces In-the-wild challenge: Database and results.
# Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark
# Localisation "In-The-Wild". 2016.


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find the largest face and keep
    dlibout = np.reshape(np.transpose(
        face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image


def extract_features_labels(dataset_dir, label_col):
    """
    <dataset_dir> refers to celeba or cartoons.
    <set_dir> refers to train or test.

    This funtion extracts the landmarks features for all images in the folder 'Datasets/celeba'.
    It also extract the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        feature_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """
    images_dir = os.path.join('.\\Datasets', dataset_dir, 'img')
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None

    labels_file = open(os.path.join('.\\Datasets', dataset_dir, 'labels.csv'), 'r')
    lines = labels_file.readlines()

    feature_labels = {line.split('\t')[0]: int(
        line.split('\t')[label_col]) for line in lines[1:]}
    # print(image_paths)
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        cnt = 0
        for img_path in image_paths:
            # cnt += 1
            # print(cnt)
            file_name = img_path.split('.')[1].split('\\')[-1]

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                # cnt -= 1
                all_features.append(features)
                all_labels.append(feature_labels[file_name])

    landmark_features = np.array(all_features)
    # simply converts the -1 into 0, so male=0 and female=1
    feature_labels = (np.array(all_labels) + 1) / 2
    return landmark_features, feature_labels


def get_data(trainset_dir, testset_dir, label_col):
    tr_X, y_train = extract_features_labels(trainset_dir, label_col)
    tr_Y = np.array([y_train, -(y_train - 1)]).T

    te_X, y_test = extract_features_labels(testset_dir, label_col)
    te_Y = np.array([y_test, -(y_test - 1)]).T

    return tr_X, tr_Y, te_X, te_Y


def ImgDataGenerator_process(trainset_dir, testset_dir, label_col, img_col='file_name', batches_size=16,
                       validation_split=0.20, color_mode='rgb', horizontal_flip=True):
    """
        Whether a test dataset folder does not already exist, it divides the dataset in training and test sets .
        In the second part it prepares the training, validation and test batches.

        :param data_directory: name (not path) of the images folder.
        :param filename_column: name of the column in the csv file where all the related filenames are declared.
        :param target_column: name of the column in the csv file where the labels are declared.
        :param training_percentage_size: percentage size of the entire dataset dedicated to the training dataset.
            default_value=0.85.
        :param batches_size: dimension of every batch. default_value=16.
        :param validation_split: percentage size of the entire dataset dedicated to the validation set.
            default_value=0.15.
        :param img_size: size of images after the reshaping. default_value=(96,96).
        :param color_mode: state 'grayscale' if the images have only one channel. default_value='rgb'.
        :param horizontal_flip: state False if it is not desired images are randomly flipped. default_value=True.
        :return: the training, validation and test batches.
    """
    # Loading the csv file
    train_images_dir = os.path.join('.\\Datasets', trainset_dir, 'img')
    train_labels = pd.read_csv(os.path.join('.\\Datasets', trainset_dir, 'labels.csv'), sep='\t', dtype='str')
    test_images_dir = os.path.join('.\\Datasets', testset_dir, 'img')
    test_labels = pd.read_csv(os.path.join('.\\Datasets', testset_dir, 'labels.csv'), sep='\t', dtype='str')

    # ImageDataGenerator generates batches of images with real-time data augmentation
    image_generator = ImageDataGenerator(rescale=1. / 255., validation_split=validation_split,
                                         horizontal_flip=horizontal_flip)
    # It produces batches of images everytime it is called
    training_batches = image_generator.flow_from_dataframe(dataframe=train_labels, directory=train_images_dir,
                                                           x_col=img_col, y_col=label_col,
                                                           subset="training", batch_size=batches_size, seed=7,
                                                           color_mode=color_mode, shuffle=True)
    # No data augmentation applied for validation and test data
    image_generator = ImageDataGenerator(rescale=1. / 255., validation_split=validation_split)
    valid_batches = image_generator.flow_from_dataframe(dataframe=train_labels, directory=train_images_dir,
                                                        x_col=img_col, y_col=label_col, subset="validation",
                                                        batch_size=batches_size, seed=7, shuffle=True,
                                                        color_mode=color_mode)
    test_batches = image_generator.flow_from_dataframe(dataframe=test_labels, directory=test_images_dir,
                                                       x_col=img_col, y_col=label_col,
                                                       color_mode=color_mode, batch_size=batches_size,
                                                       shuffle=False)
    return training_batches, valid_batches, test_batches