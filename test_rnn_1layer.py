import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# hyperparams
n_epochs=10 # epochs
bsize=32 # batch size
hsize=100 # hidden
seqlen=32 # unrolled sequence length
xsize=7 # length of each input vector
ysize=7 # length of each output vector
# creating input
a=[1,0,1,5,5,1,0,1,3, 1,0,1,-2,3,3,2,-1, 0,1,2,
   1,0,1,5,5,6,7,6,3, 6,7,8, 5,3,3,4, 3,-2,2,1]
a=np.tile(a,257)
l=len(a)
for i in range(l):
    if (np.random.rand()<1.0/40): # 1/40 rate of wrong pitch
        a[i]=a[i]+np.random.randint(1,xsize)
    a[i]=a[i]%xsize
# formating input
b=np.zeros((l,xsize))
for i in range(l):
    b[i][a[i]]=1
cy=b[1:,:]
cx=b[:-1,:]
l=l-1
# c=np.reshape(b,(1,l,ysize))

# define the model
def model_rnn_fn(x0, h0, yt, step):
    # INPUT  x0[batchsize,seqlen,in_featuresize]
    # HIDDEN h0[batchsize,hiddensize]
    # OUTPUT yt[batchsize,seqlen,out_featuresize]
    x=x0
    bsize=tf.shape(x)[0]
    seqlen=tf.shape(x)[1]
    ysize=yt.get_shape().as_list()[2]
    hsize=h0.get_shape().as_list()[1]
    
    cell = tf.nn.rnn_cell.GRUCell(hsize)
    # makes an RNN cell
    
    yn, h = tf.nn.dynamic_rnn(cell, x, initial_state=h0)
    # make a runable RNN defined by the CELL input
    
    yn=tf.reshape(yn,[bsize*seqlen,hsize])
    yr=tf.layers.dense(yn,ysize) # predicting vector of 1 element
    yr=tf.reshape(yr,[bsize,seqlen,ysize])
    
    yout=yr[:,-1,:] # take the output at the last time point
    
    loss=tf.losses.mean_squared_error(yr,yt)
    lr = 0.001 + tf.train.exponential_decay(0.01, step, 400, 0.5)
    # 0.001+0.01*0.5^(step/400)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss)
    
    return yout,h,loss,train_op
    
# instantiate (实例化) the model
tf.reset_default_graph()
# Placeholders must be fed with feed_dict for it to operate properly
h0=tf.placeholder(tf.float32, [None, hsize]) # bsize, hsize
x0=tf.placeholder(tf.float32, [None,None,xsize]) # bsize,seqlen,xsize
yt=tf.placeholder(tf.float32,[None,None,ysize]) # bsize,seqlen,ysize
step=tf.placeholder(tf.int32)

yout, h, loss, train_op = model_rnn_fn(x0, h0, yt, step)

# inference
def prediction_run(xin, ly):
    htest=np.zeros([1,hsize])
    ytest=np.zeros([1,ysize])
    lx=xin.shape[0]
    # prime the state from data
    if (lx>0):
        xtest=np.array(xin) # I don't see why we need this
        xtest=np.reshape(xtest,[1,lx,xsize]) #reshape as one sequence
        feed={h0:htest, x0:xtest}
        ytest,htest=sess.run([yout,h],feed_dict=feed)
    
    # run prediction
    results=[]
    for i in range(ly):
        ytest=np.reshape(ytest,[1,1,ysize])
        feed={h0:htest, x0:ytest}
        ytest,htest=sess.run([yout,h],feed_dict=feed)
        results.append(ytest[0,:])
    
    return np.array(results)

# Initialize Tensorflow session
h_init=np.zeros([bsize,hsize])

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run([init])

# Let's train!
htest=h_init
losses=[]
indices=[]
bpe = l//(bsize*seqlen) # batches_per_epoch
for i in range(bpe*n_epochs):
    # processing batch
    epoch=i//bpe
    batch=i%bpe
    if (batch==0): # new epoch
        xdata=np.roll(cx,-(epoch%seqlen),axis=0)
        ydata=np.roll(cy,-(epoch%seqlen),axis=0)
        xdata=np.reshape(xdata[0:(bpe*bsize*seqlen),:],[bsize,bpe*seqlen,xsize])
        ydata=np.reshape(ydata[0:(bpe*bsize*seqlen),:],[bsize,bpe*seqlen,ysize])
    next_x=xdata[:,batch*seqlen:(batch+1)*seqlen,:]
    next_y=ydata[:,batch*seqlen:(batch+1)*seqlen,:]
    # start training
    feed={h0:htest, x0:next_x, yt:next_y, step:i}
    ytest,htest,losstest,_=sess.run([yout, h, loss, train_op], feed_dict=feed)
    # print
    if (i%1==0):
        print("epoch " + str(epoch+1) + ", batch " + str(batch+1) + ", loss=" + str(np.mean(losstest)))
    if (i%1==0):
        losses.append(np.mean(losstest))
        indices.append(i)
        
# Show the results
plt.ylim(ymax=np.amax(losses[1:])) # ignore first value for scaling
plt.plot(indices, losses)
plt.show()

res=prediction_run(b[1:40,:],40)
p=np.argmax(res,axis=1)
p=np.roll(p,1)
print(a[0:40]);print(p)