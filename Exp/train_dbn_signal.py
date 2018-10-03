from dbn import SupervisedDBNClassification
from input_output import import_all_data, import_data, verify_create_dir, init_stats
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics.classification import accuracy_score
import numpy as np

train_percent = .8
balance = 1
train_type = 0

files_wav75_01=['data/data_75/wav75_ex1_01.csv', 'data/data_75/wav75_ex2_01.csv', 'data/data_75/wav75_ex3_01.csv',
				'data/data_75/wav75_ex4_01.csv', 'data/data_75/wav75_ex5_01.csv', 'data/data_75/wav75_ex6_01.csv',
				'data/data_75/wav75_ex7_01.csv', 'data/data_75/wav75_ex8_01.csv']

files_PSD = ['data/data_signal/wavPSD_ex1_.csv', 'data/data_signal/wavPSD_ex2_.csv', 'data/data_signal/wavPSD_ex3_.csv',
			'data/data_signal/wavPSD_ex4_.csv', 'data/data_signal/wavPSD_ex5_.csv', 'data/data_signal/wavPSD_ex6_.csv',
			'data/data_signal/wavPSD_ex7_.csv', 'data/data_signal/wavPSD_ex8_.csv']

files_SIG = ['data/data_signal/wavSIG_ex1_.csv', 'data/data_signal/wavSIG_ex2_.csv', 'data/data_signal/wavSIG_ex3_.csv',
			'data/data_signal/wavSIG_ex4_.csv', 'data/data_signal/wavSIG_ex5_.csv', 'data/data_signal/wavSIG_ex6_.csv',
			'data/data_signal/wavSIG_ex7_.csv', 'data/data_signal/wavSIG_ex8_.csv']

#X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75_01,1, 1-train_percent, balance, train_type)
#X_train, y_train, X_test, y_test, n_att = import_all_data(files_SIG,1, 1-train_percent, balance, train_type)
X_train, y_train, X_test, y_test, n_att = import_all_data(files_PSD,1, 1-train_percent, balance, train_type)
X_train = np.array(X_train, dtype = 'float32')
X_test = np.array(X_test, dtype = 'float32')
y_train = np.array([y[0] for y in y_train])
y_test = np.array([y[0] for y in y_test])

print(sum(y_train)/len(y_train))
print(sum(y_test)/len(y_test))

#[128, 256, 256]
#[128, 256, 512, 256]
#[128, 256, 512, 512]

classifier = SupervisedDBNClassification(hidden_layers_structure=[1024, 1024],
                                         learning_rate_rbm=0.01,
                                         learning_rate=0.01,
                                         n_epochs_rbm=1000,
                                         n_iter_backprop=1000,
                                         batch_size=16,
                                         activation_function='relu',
                                         dropout_p=0.1)

classifier.fit(X_train, y_train)
Y_pred = classifier.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, Y_pred)
AUC = auc(fpr, tpr)
print('Done.\nAccuracy: %f' % accuracy_score(y_test, Y_pred))
print('Done.\nAUC:      %f' % AUC)

classifier.save('model.pkl')
