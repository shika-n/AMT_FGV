import sys
sys.path.append('')
print(sys.path)

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from preprocess import generate_cqt, process_csv_data, PITCH_RANGE


def main():  # ENSTDkAm/MUS/
    test_file_name = '../data/predict/MAPS_MUS-bk_xmas1_ENSTDkAm'
    cqt_result = generate_cqt(0, test_file_name + '.wav', duration=10)
    label_result = process_csv_data(0, test_file_name + '.txt', cqt_result.shape[1])
    cqt_result = cqt_result.T

    mean = np.mean(cqt_result, axis=0, keepdims=True)
    std = np.std(cqt_result, axis=0, keepdims=True)
    cqt_result = np.divide(np.subtract(cqt_result, mean), std)

    is_88 = True
    if is_88:
        predictions_88 = predict88(cqt_result)
        labels_88 = label_result[:, PITCH_RANGE:]

        predictions_88 = predictions_88.reshape((PITCH_RANGE, -1)).T
        predictions_88 = np.ceil(np.clip(predictions_88 - 0.5, 0, 1))

        '''predictions = np.zeros((len(labels_88), PITCH_RANGE))
        
        for pitch, arr in enumerate(predictions_88):
            for i, value in enumerate(arr):
                predictions[i, pitch] = value'''

        np.savetxt('predict.txt', predictions_88, '%i')

        print(predictions_88.shape, np.asarray(labels_88).shape)
        calculate_accuracy(labels_88, predictions_88)

    else:
        predictions = predict(cqt_result)
        np.savetxt('predict.txt', predictions, '%i')
        calculate_accuracy(label_result, predictions)

def calculate_accuracy(labels, predictions):
    acc_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores = []

    for i in range(labels.shape[0]):
        pred = predictions[i]
        target = labels[i]

        val_acc = accuracy_score(target, pred)
        val_f1 = f1_score(target, pred)
        val_recall = recall_score(target, pred)
        val_precision = precision_score(target, pred)

        acc_scores.append(val_acc)
        f1_scores.append(val_f1)
        recall_scores.append(val_recall)
        precision_scores.append(val_precision)

    acc = np.mean(acc_scores)
    f1 = np.mean(f1_scores)
    recall = np.mean(recall_scores)
    precision = np.mean(precision_scores)

    return acc, f1, recall, precision


'''
def calculate_accuracy(labels, predictions):
    true_positive = 0  # 1 if sum - 1
    true_negative = 0  # 1 if true - pred
    false_positive = 0  # -1 if true - pred
    false_negative = 0
    for i in range(labels.shape[0]):
        diff = np.subtract(labels[i], predictions[i])
        true_positive += np.count_nonzero(np.clip(np.add(labels[i], predictions[i]) - 1, 0, 1))  # v
        false_negative += np.count_nonzero(np.clip(diff, 0, 1))
        false_positive += np.count_nonzero(np.clip(diff, -1, 0))
        true_negative += len(labels[i]) - np.count_nonzero(np.add(labels[i], predictions[i]))

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * ((precision * recall) / (precision + recall))

    print('TP:', true_positive, ', FN:', false_negative, ', FP:', false_positive, 'TN:', true_negative)
    print('Acc: {}'.format(accuracy * 100))
    print('Precision: {}'.format(precision * 100))
    print('Recall: {}'.format(recall * 100))
    print('F1: {}'.format(f1 * 100))
'''

def predict88(cqt_result):
    model = load_model('model_train_88.best.h5')
    predictions = model.predict(cqt_result)
    return np.asarray(predictions)


def predict(cqt_result):
    model = load_model('model_train.best.h5')
    predictions = model.predict(cqt_result)
    predictions = np.ceil(np.subtract(predictions, 0.8))
    return np.asarray(predictions)


if __name__ == '__main__':
    main()