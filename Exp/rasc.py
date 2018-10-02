tf.reset_default_graph()
n_output = 2

# Building 'AlexNet'
network = input_data(shape=[None, 5, 5, 3])
network = conv_2d(network, 227, 227, strides=4, activation='tanh')
network = max_pool_2d(network, 3, strides=2)
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, n_output, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='roc_auc_score',
                     learning_rate=0.0001)
# network = regression(network, optimizer='momentum',
#                    loss='categorical_crossentropy',
#                     learning_rate=0.001)

#from DogsAndCatsHelper import DogsAndCatsHelper
#train_images, train_labels = DogsAndCatsHelper.get_data()

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet', max_checkpoints=1, tensorboard_verbose=2)
model.fit(X_train, y_train, n_epoch=5, validation_set=0.05, shuffle=True, show_metric=True, batch_size=16, snapshot_step=25,snapshot_epoch=False, run_id='alexnet_spindles')



tf.reset_default_graph()
net = tflearn.input_data([None, 75])
net = tflearn.embedding(net, input_dim=75, output_dim=1024)
net = tflearn.lstm(net, 1024, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='roc_auc_score')

# Training
model =  tflearn.DNN(network, checkpoint_path='model_lstm', max_checkpoints=1, tensorboard_verbose=2)
model.fit(X_train, y_train, n_epoch = 1, validation_set=(X_test, y_test), show_metric=True,batch_size=16, snapshot_step=25,snapshot_epoch=False, run_id='lstm_spindles')


train_percent = .8
balance = 1
train_type = 1

files_wav75_01=['data/data_75/wav75_ex1_01.csv', 'data/data_75/wav75_ex2_01.csv', 'data/data_75/wav75_ex3_01.csv',
				'data/data_75/wav75_ex4_01.csv', 'data/data_75/wav75_ex5_01.csv', 'data/data_75/wav75_ex6_01.csv',
				'data/data_75/wav75_ex7_01.csv', 'data/data_75/wav75_ex8_01.csv']

X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75_01,1, 1-train_percent, balance, train_type)
X_train = np.array(X_train, dtype = 'float32')
X_test = np.array(X_test, dtype = 'float32')
y_train = np.array([y[0] for y in y_train])
y_test = np.array([y[0] for y in y_test])

[128, 256, 256]
[128, 256, 512, 256]
[128, 256, 512, 512]
from dbn import SupervisedDBNClassification
classifier = SupervisedDBNClassification(hidden_layers_structure=[128, 256, 512, 512],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.01,
                                         n_epochs_rbm=500,
                                         n_iter_backprop=10000,
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

