from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics.classification import accuracy_score
import numpy as np
import numpy as np
from input_output import import_all_data, import_data, verify_create_dir, init_stats
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from dbn.models import UnsupervisedDBN
from dbn import SupervisedDBNClassification
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier as MLP

files_wav75_01=['data/data_75/wav75_ex1_.csv', 'data/data_75/wav75_ex2_.csv', 'data/data_75/wav75_ex3_.csv',
				'data/data_75/wav75_ex4_.csv', 'data/data_75/wav75_ex5_.csv', 'data/data_75/wav75_ex6_.csv',
				'data/data_75/wav75_ex7_.csv', 'data/data_75/wav75_ex8_.csv']

files_wav75_FN =['data/data_75/wav_ex1_Filtered_Norm1_STP.csv', 'data/data_75/wav_ex2_Filtered_Norm1_STP.csv', 'data/data_75/wav_ex3_Filtered_Norm1_STP.csv',
					'data/data_75/wav_ex4_Filtered_Norm1_STP.csv', 'data/data_75/wav_ex5_Filtered_Norm1_STP.csv', 'data/data_75/wav_ex6_Filtered_Norm1_STP.csv',
					'data/data_75/wav_ex7_Filtered_Norm1_STP.csv', 'data/data_75/wav_ex8_Filtered_Norm1_STP.csv']


train_percent = .8
balance = 1
train_type = 0


X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75_FN,1, 1-train_percent, balance, train_type)
X_train = np.array(X_train, dtype = 'float32')
X_test = np.array(X_test, dtype = 'float32')
y_train = np.array([y[0] for y in y_train])
y_test = np.array([y[0] for y in y_test])

#[512,256,128,64,32]

classifier = SupervisedDBNClassification(hidden_layers_structure=[512,512,2048],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.05,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=5000,
                                         batch_size=20,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(X_train,y_train)
print()
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        y_test,
        classifier.predict(X_test))))


classifier.save('model1.pkl')
