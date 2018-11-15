import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn.preprocessing as sklrn
import time as time

"""
Creating a convolutional neural network to perform ~~ 95% on MNIST
Current plan:
    conv -> relu with dropout -> conv ->relu -> max pool -> conv -> hidden/relu -> out
    
"""

t0 = time.time()

filename = "to_my_path/train.csv"

df_init = pd.read_csv(filename).values
#df_init_val = pd.read_csv(filetest).values

train_labels = df_init[:,0]
train_dataset = df_init[:,1:]

#valid_labels = df_init_val[:,0]
#valid_dataset = df_init_val[:,1:]

image_size = 28
num_channels = 1 ## Grayscale images
batch_size = 16 # mini-batch SGD
num_hidden = 64
num_hidden2 = 32
num_labels = 10 # different digits

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def write_to_csv(prediction):
    mx = np.argmax(prediction,1)
    if np.shape(np.shape(mx))[0] == 1:
        mx = mx[:,np.newaxis]
    else:
        pass
    output_lst = []
    length_of_prediction = np.shape(mx)[0]
    for i in range(0,length_of_prediction):
        imgid = i+1
        output_lst.append({'ImageId':imgid,'Label':mx[i][0]})
    
    df_output = pd.DataFrame(output_lst)
    df_output.to_csv("results.csv",header=True,index=False)


def reformat(labels,dataset):
  dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)  
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return labels, dataset

train_dataset = sklrn.scale(train_dataset,axis=0)
#valid_dataset = sklrn.scale(valid_dataset,axis=0)

#valid_labels = np.zeros([np.shape(valid_dataset)[0],1])

train_labels, train_dataset = reformat(train_labels, train_dataset)
#valid_labels, valid_dataset = reformat(valid_labels, valid_dataset)


graph = tf.Graph()
with graph.as_default():
    
    ## Define constants and placeholders
    tf_train_dataset = tf.placeholder(
                tf.float32,shape=[batch_size, image_size, image_size, num_channels])
    tf_train_labels = tf.placeholder(tf.float32, shape= [batch_size, num_labels])
    #tf_valid_dataset = tf.constant(valid_dataset)
    #tf_valid_labels = tf.constant(valid_labels)    
    keep_prob = tf.placeholder("float")
    test_prediction = tf.placeholder("float", name="evaluate_me")
    
    ## Define variables/weight matrices and biases.
    
    layer1_weights = tf.Variable(
                tf.truncated_normal([11,11,num_channels,12], stddev=0.1), name='w1')
    layer1_biases = tf.Variable(tf.zeros(12), name='w1_bias')
    
    layer2_weights = tf.Variable(tf.truncated_normal(
                [7, 7, 12, 32], stddev=0.1), name='w2')
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[32]), name='w2_bias')
    
    layer3_weights = tf.Variable(tf.truncated_normal(
                [5, 5, 32, 48], stddev=0.1), name='w3')  ## AFter chacnging to 96 layers deep, 92.0%
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[48]), name='w3_bias')
    
    layer4_weights = tf.Variable(tf.truncated_normal(  ## For fully connected layer
                [7*7*48,num_hidden], stddev=0.1), name='w4') ## num_hidden = number of hidden layer nodes.
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]), name='w4_bias') 
    
    layer5_weights = tf.Variable(tf.truncated_normal(
                [num_hidden, num_labels], stddev=0.1), name='w5')
    layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='w5_bias')

    layer6_weights = tf.Variable(tf.truncated_normal(
                [5,5,32,32], stddev=0.1), name='w7')
    layer6_biases = tf.Variable(tf.constant(1.0, shape=[32]), name='w7_bias')
    
    # Model
    def model(data, keep_prob): ## similar to LeNet. 
        
        conv = tf.nn.conv2d(data,layer1_weights, [1,1,1,1], padding='SAME') # 28x28 
        hidden = tf.nn.relu(conv + layer1_biases)
        hidden_maxed = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding="SAME") # 14x14
        
        conv = tf.nn.conv2d(hidden, layer2_weights, [1,1,1,1], padding='SAME') # Keeps same shape 14x14
        hidden = tf.nn.relu(conv + layer2_biases)
        hidden_maxed = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding="SAME") # 14x14
        
        conv = tf.nn.conv2d(hidden_maxed,layer6_weights, [1,1,1,1], padding="SAME") # 14x14
        hidden = tf.nn.dropout(tf.nn.relu(conv + layer6_biases), keep_prob)
        
        conv = tf.nn.conv2d(hidden_maxed, layer3_weights, [1,1,1,1],padding="SAME") # 7x7
        hidden = tf.nn.relu(conv + layer3_biases)
        hidden_maxed = tf.nn.max_pool(hidden, [1,2,2,1],[1,2,2,1],padding="SAME")
        
        shape = hidden_maxed.get_shape().as_list()
        reshape = tf.reshape(hidden_maxed, [shape[0], shape[1]*shape[2]*shape[3]]) ## Makes a fully connected layer
        
        hidden = tf.nn.relu(tf.matmul(reshape, layer4_weights) + layer4_biases)
        
        final = tf.matmul(hidden,layer5_weights) + layer5_biases
        return final
    
    # Define operations/computations
    
    logits = model(tf_train_dataset, keep_prob)
    loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels=tf_train_labels))
    
    ## Define optimizer
    global_step = tf.Variable(0)
    start_learning_rate = 0.01        
    learning_rate = tf.train.exponential_decay(start_learning_rate,global_step,10000,0.96,
                                               staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    
    ## Predictions
    train_prediction = tf.nn.softmax(logits)
    
    # For evaluating a dataset after restoring this graph in a different file.  
    final_test_prediction = tf.nn.softmax(model(test_prediction, 1.0),name="final_prediction")
    
        

num_steps = 1051

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:0.5}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      #print('Validation accuracy: %.1f%%' % accuracy(
       # test_prediction.eval(), valid_labels))
  
  #test_prediction = tf.placeholder("float",name="test")
  #final_test_prediction = tf.nn.softmax(model(test_prediction, 1.0),name="final_prediction")
  preds = final_test_prediction.eval()
  #print('Test accuracy: %.1f%%' % accuracy(preds, valid_labels))
  
#write_to_csv(pred)
print(time.time()-t0,'s')
        