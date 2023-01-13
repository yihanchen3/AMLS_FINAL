import argparse
from Modules.pre_process import get_data, ImgDataGenerator_process
import joblib
from A1.A1 import A1
from A2.A2 import A2_SVC, A2_GridSVC
from B1.B1 import B1

import warnings

# warnings.filterwarnings("ignore", category=FutureWarning)


def main(task_AB, task_12):
    print('performing task: %s%d' % (task_AB, task_12))
    if task_AB == 'A':
        if task_12 == 1:  # gender label
            trainset_dir, testset_dir, label_col = 'celeba', 'celeba_test', 2
            training_images, training_labels, test_images, test_labels = get_data(trainset_dir, testset_dir, label_col)
            train_acc, test_acc = A1(training_images, training_labels, test_images, test_labels)


        elif task_12 == 2:
            trainset_dir, testset_dir, label_col = 'celeba', 'celeba_test', 3
            training_images, training_labels, test_images, test_labels = get_data(trainset_dir, testset_dir, label_col)
            model_A2, train_acc, test_acc = A2_SVC(training_images.reshape((-1, 68 * 2)),
                                                   list(zip(*training_labels))[0], test_images.reshape(-1, 68 * 2),
                                                   list(zip(*test_labels))[0])
            # model_A2 = A2_GridSVC(training_images.reshape((-1,68*2)), list(zip(*training_labels))[0], test_images.reshape(-1,68*2), list(zip(*test_labels))[0])

        print('Task A%d : train_acc = %.4f, test_acc = %.4f\n' % (task_12, train_acc, test_acc))

    elif task_AB == 'B':
        if task_12 == 1:  # face shape label
            training_batches, valid_batches, test_batches = ImgDataGenerator_process(trainset_dir='cartoon_set',
                                                                                     testset_dir='cartoon_set_test',
                                                                                     label_col='face_shape',
                                                                                     batches_size=16,
                                                                                     validation_split=0.2,
                                                                                     horizontal_flip=False)
            input_shape = training_batches.image_shape
            model_B1, train_acc, valid_acc, test_acc = B1(input_shape, training_batches, valid_batches, test_batches,
                                                          epochs=10, verbose=2, plot=True)

        elif task_12 == 2:
            training_batches, valid_batches, test_batches = ImgDataGenerator_process(trainset_dir='cartoon_set',
                                                                                     testset_dir='cartoon_set_test',
                                                                                     label_col='eye_color',
                                                                                     batches_size=16,
                                                                                     validation_split=0.2,
                                                                                     horizontal_flip=False)
            input_shape = training_batches.image_shape
            model_B2, train_acc, valid_acc, test_acc = B1(input_shape, training_batches, valid_batches, test_batches,
                                                          epochs=10, verbose=2, plot=True)
        print('Task B%d : train_acc = %.4f, valid_acc = %.4f, test_acc = %.4f\n' % (
        task_12, train_acc, valid_acc, test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AMLS_FINAL_PROJECT_22049064')
    parser.add_argument('-t1', '--task_AB', type=str, choices=['A', 'B'], default='A', help='Choose Task A or B')
    parser.add_argument('-t2', '--task_12', type=int, choices=[1, 2], default=1, help='Choose Specific Task 1 or 2')
    args = parser.parse_args()
    main(args.task_AB, args.task_12)
