import cPickle
import gzip
import numpy as np

filename='mnist.pkl.gz'
f=gzip.open(filename,'rb')
training_data, validation_data, test_data = cPickle.load(f)
f.close()
inp_training_data=np.array(training_data[0])
opt_training_data=np.array(training_data[1])
# print inp_training_data[0]

N=10
# number of digits to be recognized
# -------------------------------------------------------------------------------------------
w=np.random.random((len(inp_training_data[0]), N))
b=np.random.random((len(inp_training_data), N))
etta=0.001

for j in range(50):
    for x in range(0, len(inp_training_data)):
        ak=np.dot(inp_training_data[x],w)+b[x]
        y=np.exp(ak)/np.sum(np.exp(ak))
        inp=opt_training_data[x]
        tl=[]
        for xy in range(0,N):
            if(inp==xy):
                tl.append(1)
            else:
                tl.append(0)
        ttemp=np.array(tl)
        E=np.outer(inp_training_data[x],y - ttemp)
        w=w-E*etta

def computationLR(w1,inputt,outputt):
    sim = 0
    for x in range(0, len(inputt)):
        ak1 = np.dot(inputt[x], w1) + b[x]
        y1 = np.exp(ak1) / np.sum(np.exp(ak1))
        if(np.argmax(y1)==outputt[x]):
                sim+=1
    return sim
print "Logic Regression:"
print float(computationLR(w,inp_training_data,opt_training_data))/len(inp_training_data)*100
print ""
print "The efficiency for Validation data:"
print float(computationLR(w,np.array(validation_data[0]),np.array(validation_data[1])))/len(np.array(validation_data[0]))*100
print ""
print "The efficiency for Test data:"
print float(computationLR(w,np.array(test_data[0]),np.array(test_data[1])))/len(np.array(test_data[0]))*100
#
#
# Simple Layer Neural Network
M=100
neural_w1=np.random.random((len(inp_training_data[0]), M))*0.1
neural_w2=np.random.random((M, N))*0.1
neural_b1=np.random.random((len(inp_training_data), M))
neural_b2=np.random.random((len(inp_training_data), N))
neural_etta=0.01
for i in range(1):
    print i
    for a in range(len(inp_training_data)):
        neural_zjh=inp_training_data[a].dot(neural_w1)+neural_b1[a]
        neural_zj=float(1)/(1+np.exp(-neural_zjh))
        neural_ak=np.dot(neural_zj,neural_w2)+neural_b2[a]
        neural_yk=np.exp(neural_ak)/np.sum(np.exp(neural_ak))
        neural_inp = opt_training_data[a]
        neural_tl=[]
        for xy in range(0,N):
            if(neural_inp==xy):
                neural_tl.append(1)
            else:
                neural_tl.append(0)
        neural_ttemp=np.array(neural_tl)
        delk=neural_yk-neural_ttemp
        h_delj=np.dot(neural_zj,(1-neural_zj))
        delj=h_delj*(neural_w2.dot(delk))
        delE1=np.outer(inp_training_data[a],delj)
        delE2=np.outer(delk,neural_zj)
        neural_w1-=neural_etta*delE1
        neural_w2-=neural_etta*delE2.transpose()

def neuralSim(neural_w1,neural_w2,inp_training_data,opt_training_data,neural_b1,neural_b2):
    sim = 0
    for a in range(len(inp_training_data)):
        neural_zjh = inp_training_data[a].dot(neural_w1) + neural_b1[a]
        neural_zj = float(1) / (1 + np.exp(-neural_zjh))
        neural_ak = np.dot(neural_zj,neural_w2) + neural_b2[a]
        neural_yk = np.exp(neural_ak) / np.sum(np.exp(neural_ak))
        if (np.argmax(neural_yk) == opt_training_data[a]):
            sim+=1
    return sim

print "The efficiency for neural network"
print (float(neuralSim(neural_w1,neural_w2,inp_training_data,opt_training_data,neural_b1,neural_b2))/len(inp_training_data))*100
print (float(neuralSim(neural_w1,neural_w2,validation_data[0],validation_data[1],neural_b1,neural_b2))/len(validation_data[0]))*100
print (float(neuralSim(neural_w1,neural_w2,test_data[0],test_data[1],neural_b1,neural_b2))/len(test_data[0]))*100


usps_test_x = []
usps_test_t = []
from PIL import Image
import glob
for i in range(10):
    for imgall in glob.glob('Numerals/'+str(i)+'/*.png'):
        img = Image.open(imgall)
        new_width = 28
        new_height = 28
        img = img.resize((new_width, new_height))
        pix = 1 - (np.array(list(img.getdata()))/255)
        usps_test_x.append(pix)
        usps_test_t.append(i)

print "Logistic regression USPS efficiency:"
print float(computationLR(w,np.array(usps_test_x),np.array(usps_test_t)))/len(usps_test_x)*100.0
print "Simple Layer Neural Network USPS efficiency:"
print (float(neuralSim(neural_w1,neural_w2,usps_test_x,usps_test_t,neural_b1,neural_b2))/len(usps_test_x))*100



import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

for i in range(1000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
      print "step %d, training accuracy %g"%(i, train_accuracy)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print "Efficiency on test data"
print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})*100

usps_test_t_v = []
for element in usps_test_t:
    temp=[]
    for ab in range(10):
        if (element == ab):
            temp.append(1)
        else:
            temp.append(0)
    usps_test_t_v.append(temp)

np_usps_test_x = np.array(usps_test_x)
usps_output = np.array(usps_test_t_v)

print "Efficiency on usps data"
print accuracy.eval(feed_dict={x: np_usps_test_x, y_: usps_output, keep_prob: 1.0})*100
