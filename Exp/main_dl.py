import numpy as np
from input_output import import_all_data, import_data, verify_create_dir
import os
import tensorflow as tf
from datetime import datetime
from alexnet import AlexNet
from scipy.misc import imresize


train_percent = .8
balance = 1
train_type = 0

files_wav75_01=['data/data_75/wav75_ex1_01.csv', 'data/data_75/wav75_ex2_01.csv', 'data/data_75/wav75_ex3_01.csv',
				'data/data_75/wav75_ex4_01.csv', 'data/data_75/wav75_ex5_01.csv', 'data/data_75/wav75_ex6_01.csv',
				'data/data_75/wav75_ex7_01.csv', 'data/data_75/wav75_ex8_01.csv']

X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75_01,1, 1-train_percent, balance, train_type)

X_train = features_1D_to_3D(X_train, 5, 5)
X_test  = features_1D_to_3D(X_test,  5, 5)

def features_1D_to_3D(X, n_dwt, n_att):
    imgs = np.zeros((len(X),5,5,3),dtype='float32')

    for i in range(0,len(X)):
        for j in range(0,n_dwt):
            for k in range(0,n_att):
                for l in range(0,3):
                    imgs[i,j,k,l] = X[i,k+l*n_att*3+j*n_att]
    return imgs

# Learning params
learning_rate = 0.01
num_epochs = 10
batch_size = 8

# Network params
dropout_rate = 0.5
num_classes = 2
train_layers = ['fc8', 'fc7']

# How often we want to write the tf.summary data to disk
display_step = 1

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "/tmp/finetune_alexnet/spindles"
checkpoint_path = "/tmp/finetune_alexnet/"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)


# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227,227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)
#model = tf.train.Saver().restore(session, "/tmp/finetune_alexnet/model_epoch1.ckpt")

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))  

# Train op
with tf.name_scope("train"):
  # Get gradients of all trainable variables
  gradients = tf.gradients(loss, var_list)
  gradients = list(zip(gradients, var_list))
  
  # Create optimizer and apply gradient descent to the trainable variables
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary  
for gradient, var in gradients:
  tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary  
for var in var_list:
  tf.summary.histogram(var.name, var)
  
# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)
  

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
  correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Evaluation op: Auc of the model
with tf.name_scope("auc"):
  auc, _ = tf.metrics.auc(tf.argmax(y, 1), tf.argmax(score, 1))
  
# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Add the auc to the summary
tf.summary.scalar('auc', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
#train_generator = ImageDataGenerator(train_file, horizontal_flip = True, shuffle = True)
#val_generator = ImageDataGenerator(val_file, shuffle = False) 

# Get the number of training/validation steps per epoch
#train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
#val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)

train_batches_per_epoch = np.floor(len(X_train) / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(len(X_test) / batch_size).astype(np.int16)

# Start Tensorflow session


with tf.Session() as sess:
 
  # Initialize all variables
  sess.run(tf.global_variables_initializer())
  
  # Add the model graph to TensorBoard
  writer.add_graph(sess.graph)
  
  # Load the pretrained weights into the non-trainable layer
  model.load_initial_weights(sess)
  
  print("{} Start training...".format(datetime.now()))
  print("{} Open Tensorboard at --logdir {}".format(datetime.now(), 
                                                    filewriter_path))
  
  # Loop over number of epochs
  count = 0
  for epoch in range(num_epochs):
    
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        
        step = 1
        
        while step < train_batches_per_epoch:
            
            # Get a batch of images and labels
            #batch_xs, batch_ys = train_generator.next_batch(batch_size)
            if ((count+1)*batch_size) < len(X_train):
                batch_xs = X_train[count*batch_size:((count+1)*batch_size)]
                batch_ys = y_train[count*batch_size:((count+1)*batch_size)]
                count += 1
            elif count*batch_size == len(X_train):
                batch_tx = X_train[0:batch_size]
                batch_ty = y_train[0:batch_size]
                count = 1 
            else:
                batch_xs = X_train[count*batch_size:len(X_train)]
                batch_ys = y_train[count*batch_size:len(y_train)]
                count = 0
            # And run the training op
            sess.run(train_op, feed_dict={x: batch_xs, 
                                          y: batch_ys, 
                                          keep_prob: dropout_rate})
            
            # Generate summary with the current batch of data and write to file
            if step%display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: batch_xs, 
                                                        y: batch_ys, 
                                                        keep_prob: 1.})
                writer.add_summary(s, epoch*train_batches_per_epoch + step)
                
            step += 1
            
        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        test_acc = 0.
        test_count = 0
        count2 = 0
        for _ in range(val_batches_per_epoch):
            #batch_tx, batch_ty = val_generator.next_batch(batch_size)

            if ((count2+1)*batch_size) < len(X_test):
                batch_tx = X_test[count2*batch_size:((count2+1)*batch_size)]
                batch_ty = y_test[count2*batch_size:((count2+1)*batch_size)]
                count2 += 1
            elif count2*batch_size == len(X_test):
                batch_tx = X_test[0:batch_size]
                batch_ty = y_test[0:batch_size]
                count2 = 1
            else:    
                batch_tx = X_test[count2*batch_size:len(X_test)]
                batch_ty = y_test[count2*batch_size:len(y_test)]
                count2 = 0

            auc = sess.run(auc, feed_dict={x: batch_tx, 
                                                y: batch_ty, 
                                                keep_prob: 1.})
            test_auc += auc
            test_count += 1
        test_auc /= test_count
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_auc))
        
        # Reset the file pointer of the image data generator
        #val_generator.reset_pointer()
        #train_generator.reset_pointer()
        
        print("{} Saving checkpoint of model...".format(datetime.now()))  
        
        #save checkpoint of the model
#        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
        checkpoint_name = os.path.join(checkpoint_path, 'model_trained.ckpt')
        save_path = saver.save(sess, checkpoint_name)  
        
        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
        