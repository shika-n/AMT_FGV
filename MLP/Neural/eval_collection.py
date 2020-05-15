# from eval_collection import save_eval, evaluate_model_sk, evaluate_model_cnn, evaluate_model_lstm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score 

def save_eval(file_name, arr_dict):
    with open(file_name + '.txt', 'w') as f:
        accs = []
        f1s = []
        recalls = []
        precisions = []
        
        for i in range(len(arr_dict)):
            i_dict = arr_dict[i]
            
            acc = i_dict['Accuracy']
            f1 = i_dict['F1']
            recall = i_dict['Recall']
            precision = i_dict['Precision']
            
            f.writelines([
                '\n------------\n',
                'Model #{}\n'.format(i),
                'Acc: {}\n'.format(acc),
                'F1: {}\n'.format(f1),
                'Recall: {}\n'.format(recall),
                'Precision: {}\n'.format(precision)
            ])
            
            accs.append(acc)
            f1s.append(f1)
            recalls.append(recall)
            precisions.append(precision)

        f.writelines([
            '================\n',
            'Acc: {} (+/- {})\n'.format(np.mean(accs), np.std(accs)),
            'F1: {} (+/- {})\n'.format(np.mean(f1s), np.std(f1s)),
            'Recall: {} (+/- {})\n'.format(np.mean(recalls), np.std(recalls)),
            'Precision: {} (+/- {})\n'.format(np.mean(precisions), np.std(precisions))
        ])

# sklearn expect each row to be a single prediction (axis=1)
# axis=0 to be n-predictions
def evaluate_model_sk(model, X, Y, reshape=None, average='macro', transpose=False):
    print('\n>>>> Evaluate sklearn\'s', average)
    val_predict = np.asarray(model.predict(X))
    val_predict = np.clip(val_predict.round(), 0, 1)
    val_target = Y
    
    if not reshape is None:
        val_predict, val_target = reshape_data(val_predict, val_target, reshape, transpose=transpose)
    
    f1 = f1_score(val_target, val_predict, average=average)
    recall = recall_score(val_target, val_predict, average=average)
    precision = precision_score(val_target, val_predict, average=average)
    acc = accuracy_score(val_target, val_predict)
    
    print('F1: {}, Recall: {}, Precision: {}, Acc: {}'.format(f1, recall, precision, acc))
    
    return {'F1': f1, 'Recall': recall, 'Precision': precision, 'Accuracy': acc}

# Doesn't really expect anything
def evaluate_model_lstm(model, X, y, reshape=None):
    print('\n>>>> Evaluate LSTM\'s')
    print("Predicting model. . . ")
    '''
    if original:
        predictions = model.predict(X, batch_size=100, verbose = 1) 
        predictions = np.reshape(predictions,(y.shape[0]*y.shape[1],y.shape[2]))
        y = np.reshape(y,(y.shape[0]*y.shape[1],y.shape[2]))
        predictions = np.array(predictions).round()
        predictions[predictions > 1] = 1
        #np.save('{}predictions'.format(weights_dir), predictions)
    '''
    predictions = np.asarray(model.predict(X))
    predictions = np.clip(predictions.round(), 0, 1)
        
    if not reshape is None:
        predictions, y = reshape_data(predictions, y, reshape)
        
    print("\nCalculating accuracy. . .")
    TP = np.count_nonzero(np.logical_and( predictions == 1, y == 1 ))
    FN = np.count_nonzero(np.logical_and( predictions == 0, y == 1 ))
    FP = np.count_nonzero(np.logical_and( predictions == 1, y == 0 ))
    
    ##
    TN = np.count_nonzero(np.logical_and( predictions == 0, y == 0 ))
    if (TP + FN) > 0:
        if TP + FN <= 0:
            R = 0
        else:
            R = TP/float(TP + FN)
        
        if TP + FP <= 0:
            P = 0
        else:
            P = TP/float(TP + FP)
        
        if TP + FP + FN <= 0:
            A = 0
        else:
            A = 100*TP/float(TP + FP + FN)
        
        ####
        A = 100 * (TP + TN) / (TP + FP + FN + TN)
        
        if P == 0 and R == 0:
            F = 0
        else: 
            F = 100*2*P*R/(P + R)
    else: 
        A = 0
        F = 0
        R = 0
        P = 0

    #print('\n F-measure pre-processed: ', F)
    #print('\n Accuracy pre-processed: ', A)
    
    print('F1: {}, Recall: {}, Precision: {}, Acc: {}'.format(F, R, P, A))
    
    return {'F1': F, 'Recall': R, 'Precision': P, 'Accuracy': A}


# Copied from music-transcription-master
# Expect each row to be one label (axis=1) or n-predictions
# axis=0 is n-labels or a single prediction
def evaluate_model_cnn(model, X, Y, reshape=None, transpose=False):
    print('\n>>>> Evaluate CNN\'s')
    acc_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores =[]

    val_predict = np.asarray(model.predict(X))
    val_predict = np.clip(val_predict.round(), 0, 1)
    val_target = Y
    
    if not reshape is None:
        val_predict, val_target = reshape_data(val_predict, val_target, reshape, transpose=transpose)
    
    for i in range(val_predict.shape[0]):
        pred = val_predict[i]
        target = val_target[i]
        
        val_acc = accuracy_score(target, pred)
        val_f1 = f1_score(target, pred)
        val_recall = recall_score(target, pred)
        val_precision = precision_score(target, pred)
        
        acc_scores.append(val_acc)
        f1_scores.append(val_f1)
        recall_scores.append(val_recall)
        precision_scores.append(val_precision)
                                
    f1 = sum(f1_scores) / float(len(f1_scores))
    recall = sum(recall_scores) / float(len(recall_scores))
    precision = sum(precision_scores) / float(len(precision_scores))
    acc = sum(acc_scores) / float(len(acc_scores))
    #print ('---> F1:', f1, )
    #print ('| Recall:', recall,)
    #print ('| Precision:', precision,)
    #print ('| Acc:', acc)
    
    print('F1: {}, Recall: {}, Precision: {}, Acc: {}'.format(f1, recall, precision, acc))
    
    return {'F1': f1, 'Recall': recall, 'Precision': precision, 'Accuracy': acc}


def reshape_data(X, Y, shape, transpose=False):
    X = np.asarray(X)
    Y = np.asarray(Y)
    print('Original Shapes:', X.shape, Y.shape)
    
    new_X = np.reshape(X, shape)
    new_Y = np.reshape(Y, shape)
    
    if transpose:
        new_X = np.transpose(new_X)
        new_Y = np.transpose(new_Y)
    
    print('Reshaped:', new_X.shape, new_Y.shape)
    
    return new_X, new_Y