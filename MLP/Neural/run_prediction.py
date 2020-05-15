import sys
sys.path.append('')
print(sys.path)

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from preprocess import generate_cqt, process_csv_data, PITCH_RANGE

from eval_collection import save_eval, evaluate_model_sk, evaluate_model_cnn, evaluate_model_lstm


def main():  # ENSTDkAm/MUS/
    test_file_name = '../data/predict/MAPS_MUS-alb_se2_StbgTGd2'
    cqt_result = generate_cqt(0, test_file_name + '.wav', duration=10)
    label_result = process_csv_data(0, test_file_name + '.txt', cqt_result.shape[1])
    cqt_result = cqt_result.T

    mean = np.mean(cqt_result, axis=0, keepdims=True)
    std = np.std(cqt_result, axis=0, keepdims=True)
    cqt_result = np.divide(np.subtract(cqt_result, mean), std)
    
    model = load_model('model_88_mlp_0.h5')
    labels_88 = np.asarray(label_result[PITCH_RANGE:, :])
    print(labels_88.shape)

    evaluate_model_cnn(model, cqt_result, labels_88, reshape=(88, -1))
    evaluate_model_lstm(model, cqt_result, labels_88, reshape=(88, -1))
    evaluate_model_sk(model, cqt_result, labels_88, average='macro', reshape=(88, -1), transpose=True)
    evaluate_model_sk(model, cqt_result, labels_88, average='micro', reshape=(88, -1), transpose=True)
        

if __name__ == '__main__':
    main()