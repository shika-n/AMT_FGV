import numpy as np
from tensorflow.keras.models import load_model
from preprocess import generate_cqt, process_csv_data, PITCH_RANGE


def main():  # ENSTDkAm/MUS/
    test_file_name = '../data/predict/MAPS_MUS-bk_xmas1_ENSTDkAm'
    cqt_result = generate_cqt(0, test_file_name + '.wav', duration=10)
    cqt_result = np.transpose(cqt_result)
    label_result = process_csv_data(0, test_file_name + '.txt', len(cqt_result))

    is_88 = True
    predictions = []
    if is_88:
        predictions_88 = predict88(cqt_result);
        predictions = np.zeros((len(cqt_result), PITCH_RANGE * 2))
        for i, arr in enumerate(predictions_88):
            for j, result in enumerate(arr):
                index = np.argmax(result)
                if index == 0: # Onset
                    predictions[j, i] = 1
                    predictions[j, i + PITCH_RANGE] = 1
                elif index == 1: # Still
                    predictions[j, i + PITCH_RANGE] = 1
    else:
        predictions = predict(cqt_result)

    np.savetxt('predict.txt', predictions, '%i')

    true_positive = 0  # 1 if sum - 1
    true_negative = 0  # 1 if true - pred
    false_positive = 0  # -1 if true - pred
    false_negative = 0
    for i in range(label_result.shape[0]):
        diff = np.subtract(label_result[i], predictions[i])
        true_positive += np.count_nonzero(np.clip(np.add(label_result[i], predictions[i]) - 1, 0, 1))  # v
        false_negative += np.count_nonzero(np.clip(diff, 0, 1))
        false_positive += np.count_nonzero(np.clip(diff, -1, 0))
        true_negative += len(label_result[i]) - np.count_nonzero(np.add(label_result[i], predictions[i]))

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * ((precision * recall) / (precision + recall))

    print('TP:', true_positive, ', FN:', false_negative, ', FP:', false_positive, 'TN:', true_negative)
    print('Acc: {}%'.format(accuracy * 100))
    print('Precision: {}%'.format(precision * 100))
    print('Recall: {}%'.format(recall * 100))
    print('F1: {}%'.format(f1 * 100))


def predict88(cqt_result):
    model = load_model('model_train_88.best.h5')
    predictions = model.predict(cqt_result)
    return predictions


def predict(cqt_result):
    model = load_model('model_train.best.h5')
    predictions = model.predict(cqt_result)
    predictions = np.ceil(np.subtract(predictions, 0.8))
    return predictions


if __name__ == '__main__':
    main()