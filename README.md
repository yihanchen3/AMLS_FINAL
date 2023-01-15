# AMLS_FINAL
In this project, two classification tasks listed below are addressed under the supervised machine learning methods and reached good performance. 

1.Binary tasks of Gender &amp; Emotion Detection (celeba dataset) 

2.Multiclass tasks of Face shape &amp; Eye color recognition (cartoon_set dataset)


## How to run the project

To run the code, please use the command to run the python file `main.py` .
Four tasks of this project can be run separately by entering the corresponding parser of the python file.
There are two parsers for main.py. The first one is `-task_AB`, shorter version as `-t1`, which requires entering **A** or **B** as the binary or multiclass task you want to run.
The second one is `-task_12`, shorter version as `-t2`, which requires entering **1** or **2** as the specific task you want to run.
The exact command to run each four task is listed below.

#### To run Task A1:
```python main.py -t1 A -t2 1```

#### To run Task A2:
```python main.py -t1 A -t2 2```

#### To run Task B1:
```python main.py -t1 B -t2 1```

#### To run Task B2:
```python main.py -t1 B -t2 2```

**It should be noticed**
that since some models developed in the project use the plt.show() to display the image in the middle of the whole process of the task, you might need to close the image window manly to view the subsequent outputs of the program.
The models involving with image display are Task A1, Task B1 and Task B2.


## Packages required

The following lists present all the packages needed to run the project code.
The `requirement.txt` file that refers to the environment of the  project is also generated in the folder.


- **argparse** provides parsers that make it easy to write user-friendly command-line interfaces.

- **numpy** is the fundamental package for array computing with Python.

- **pandas** provides fast, flexible, and expressive data structures in Python.

- **matplotlib** is a comprehensive library for creating static, animated, and interactive visualizations in Python.

- **os** provides a portable way of using operating system dependent functionality.

- **dlib** is a modern C++ toolkit containing machine learning algorithms. It is used in this project to provide a face detection method.

- **cv2** is an open-source library that includes several hundreds of computer vision algorithms.

- **tensorflow** is a free and open-source software library for machine learning and artificial intelligence that provides high performance numerical computation.

- **scikit-learn** offers simple and efficient tools for predictive data analysis that include many classical and advanced machine learning algorithms.

## Role of each file

**main.py** is where to run the four tasks. This file calls models from their definition files  and run them separately under the control of argparse.
is the starting point of the entire project. It defines the order in which instructions are realised. More precisely, it is responsible to call functions from other files in order to divide the datasets provided, pre-process images and instantiate, train and test models.

**A1.py** defines the multilayer perceptron structure and its parameters allocation method.This file also provides function `A1` to train, test, and predict on the dataset. Results analysis part is included in the evaluate part.

**A2.py** defines the support vector classification structure and the GridSearchCV method to optimize hyperparameters.This file also provides function `A2` to train, test, and predict on the dataset. 

**B1.py** defines the CNN structure and the parameters set for each layer. This file also provides function `B1` to train, test, and predict on the dataset. Results analysis part is included in the evaluate part.

**B2.py** defines the CNN structure and the parameters set for each layer. This file also provides function `B2` to train, test, and predict on the dataset. Results analysis part is included in the evaluate part.

**pre_processing.py** provides functions to the data preparation. `extract_features_labels` pre-processes the original data with a face detector to offer input data for Task A; `ImgDataGenerator_process` pre-processes the original data with the imagedatagenerator method to offer input data for Task B. 

**result_process.py** includes functions to visualize the training process and results of the model.

**shape_predictor_68_face_landmarks.dat** serves as the pre-trained model for the dlib 68 face detector function.