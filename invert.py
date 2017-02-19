import numpy as np

name = 't10k-images-idx3-ubyte'
data_dir = 'data/mnist/'
fd = open(data_dir+name)
data = np.fromfile(file=fd,dtype=np.uint8)
trX = data[16:].reshape((-1,28,28,1)).astype(np.float)
invert_trX = 255-trX
np.save(name, invert_trX)

trX = np.load(name+'.npy')
if np.sum(trX-invert_trX) == 0:
    print 'Success'
else:
    print 'Fail'

name = 'train-images-idx3-ubyte'
data_dir = '../'
fd = open(data_dir+name)
data = np.fromfile(file=fd,dtype=np.uint8)
trX = data[16:].reshape((-1,28,28,1)).astype(np.float)
invert_trX = 255-trX
np.save(name, invert_trX)

trX = np.load(name+'.npy')
if np.sum(trX-invert_trX) == 0:
    print 'Success'
else:
    print 'Fail'
