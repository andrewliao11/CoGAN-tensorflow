from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *
import pdb

# if use mnist, the y denotes the number in the image
# in generator, the z(noise) will concat with y as input
class CoGAN(object):
    def __init__(self, sess, image_size=108, is_crop=True,
                 batch_size=64, sample_size = 64, output_size=64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir=None):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

	# y_dim is the conditional signal
        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

	# ------------batch norm-------------------
	# batchnorm that share vars
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
	self.d_bn2 = batch_norm(name='d_bn2')

        # batchnorm that doesn't share vars
	self.d1_bn1 = batch_norm(name='d1_bn1')
        self.d2_bn1 = batch_norm(name='d2_bn1')
	# -----------------------------------------
	
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):

	# G1, D1
        self.images1 = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim],
                                    name='real_images1')
        self.sample_images1 = tf.placeholder(tf.float32, [self.sample_size] + [self.output_size,self.output_size,self.c_dim],
                                        name='sample_images1')
	# G2, D2
	self.images2 = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim],
                                    name='real_images2')
        self.sample_images2 = tf.placeholder(tf.float32, [self.sample_size] + [self.output_size,self.output_size,self.c_dim],
                                        name='sample_images2')
	# Generative model input
        if self.y_dim:
            self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
	# latent variable
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)

	'''
	The share_params FLAG denotes the share weight btn two network(G, D)
	The reuse FLAG denotes that we will use it to do the inference
	Note: every network should be declared with (False, False) FLAG
	'''
	# input of the generator is the concat of z, y
        self.G1 = self.generator(self.z, self.y, share_params=False, reuse=False, name='G1')
	self.G2 = self.generator(self.z, self.y, share_params=True, reuse=False, name='G2')
        # input the paired input image(natural images)
        self.D1_logits, self.D1 = self.discriminator(self.images1, self.y, share_params=False, reuse=False, name='D1')
	self.D2_logits, self.D2 = self.discriminator(self.images2, self.y, share_params=True, reuse=False, name='D2')
	# generate sample
        self.sampler1 = self.generator(self.z, self.y, share_params=True, reuse=True, name='G1')
	self.sampler2 = self.generator(self.z, self.y, share_params=True, reuse=True, name='G2')
	# input the fake images
        self.D1_logits_, self.D1_ = self.discriminator(self.G1, self.y, share_params=True, reuse=True, name='D1')
	self.D2_logits_, self.D2_ = self.discriminator(self.G2, self.y, share_params=True, reuse=True, name='D2')
        
	# B1
        self.d1_sum = tf.summary.histogram("d1", self.D1)
        self.d1__sum = tf.summary.histogram("d1_", self.D1_)
        self.G1_sum = tf.summary.image("G1", self.G1)

	# B2
	self.d2_sum = tf.summary.histogram("d2", self.D2)
        self.d2__sum = tf.summary.histogram("d2_", self.D2_)
        self.G2_sum = tf.summary.image("G2", self.G2)

	# B1
        self.d1_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D1_logits, tf.ones_like(self.D1)*0.9))
        self.d1_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D1_logits_,tf.ones_like(self.D1_)*0.1))
        self.g1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D1_logits_, tf.ones_like(self.D1_)*0.9))
	self.d1_loss_real_sum = tf.summary.scalar("d1_loss_real", self.d1_loss_real)
        self.d1_loss_fake_sum = tf.summary.scalar("d1_loss_fake", self.d1_loss_fake)
	self.d1_loss = self.d1_loss_real + self.d1_loss_fake
	self.g1_loss_sum = tf.summary.scalar("g1_loss", self.g1_loss)
        self.d1_loss_sum = tf.summary.scalar("d1_loss", self.d1_loss)

	# B2
        self.d2_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D2_logits, tf.ones_like(self.D2)*0.9))
        self.d2_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D2_logits_,tf.ones_like(self.D2_)*0.1))
        self.g2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D2_logits_, tf.ones_like(self.D2_)*0.9))
        self.d2_loss_real_sum = tf.summary.scalar("d2_loss_real", self.d2_loss_real)
        self.d2_loss_fake_sum = tf.summary.scalar("d2_loss_fake", self.d2_loss_fake)
        self.d2_loss = self.d2_loss_real + self.d2_loss_fake
        self.g2_loss_sum = tf.summary.scalar("g2_loss", self.g2_loss)
        self.d2_loss_sum = tf.summary.scalar("d2_loss", self.d2_loss)

	# sum together
	self.d_loss = self.d1_loss+self.d2_loss
	self.g_loss = self.g1_loss+self.g2_loss
	self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
	# all variable
        t_vars = tf.trainable_variables()
	# variable list
	self.g_vars = [var for var in t_vars if 'g1_' in var.name] + [var for var in t_vars if 'g2_' in var.name] \
				+ [var for var in t_vars if 'g_' in var.name]
	self.d_vars = [var for var in t_vars if 'd1_' in var.name] + [var for var in t_vars if 'd2_' in var.name] \
				+ [var for var in t_vars if 'd_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train CoGAN"""
	# data_X1 is the original image
	# data_X2 is the black-white image
	# data_y is the label
        data_X1, data_y = self.load_mnist()
	data_X2 = self.load_invert_mnist()

	# do the random shuffle for two sets -> without paired images
	idx = np.arange(len(data_y))
	np.random.shuffle(idx)
	data_X1 = data_X1[idx]
	data_y1 = data_y[idx]
	idx = np.arange(len(data_y))
	np.random.shuffle(idx)
	data_X2 = data_X2[idx]
	data_y2 = data_y[idx]

	d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)

	tf.global_variables_initializer().run()

        self.g1_sum = tf.summary.merge([self.z_sum, self.d1__sum, 
            self.G1_sum, self.d1_loss_fake_sum, self.g1_loss_sum])
        self.d1_sum = tf.summary.merge([self.z_sum, self.d1_sum, self.d1_loss_real_sum, self.d1_loss_sum])

        self.g2_sum = tf.summary.merge([self.z_sum, self.d2__sum,
            self.G2_sum, self.d2_loss_fake_sum, self.g2_loss_sum])
        self.d2_sum = tf.summary.merge([self.z_sum, self.d2_sum, self.d2_loss_real_sum, self.d2_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

	# sample noise
        sample_z = np.random.normal(size=(self.batch_size , self.z_dim))
        sample_images1 = data_X1[0:self.batch_size]
	sample_images2 = data_X2[0:self.batch_size]
        sample_labels1 = data_y1[0:self.batch_size]
        sample_labels2 = data_y2[0:self.batch_size]
            
        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            batch_idxs = min(len(data_X1), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_images1 = data_X1[idx*config.batch_size:(idx+1)*config.batch_size]
		batch_images2 = data_X2[idx*config.batch_size:(idx+1)*config.batch_size]
                batch_labels1 = data_y1[idx*config.batch_size:(idx+1)*config.batch_size]
		batch_labels2 = data_y2[idx*config.batch_size:(idx+1)*config.batch_size]
		# z is the noise
                batch_z = np.random.normal(size=[config.batch_size, self.z_dim]).astype(np.float32)
		# Update D network
                _, summary_str = self.sess.run([d_optim, self.d1_sum],
                        feed_dict={ self.images1: batch_images1, self.images2: batch_images2, 
					self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g1_sum],
                        feed_dict={ self.z: batch_z})
                self.writer.add_summary(summary_str, counter)
                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g1_sum],
                        feed_dict={ self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

		errD = self.d_loss.eval({self.z: batch_z, self.images1: batch_images1, self.images2: batch_images2})
		errG = self.g_loss.eval({self.z: batch_z})
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD, errG))

                if np.mod(counter, 100) == 1:
		    self.evaluate(sample_images1,sample_images2,sample_labels1,batch_labels1,sample_labels2,batch_labels2, 
				sample_z, './samples/top/train_{:02d}_{:04d}.png'.format(epoch, idx))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def evaluate(self, sample_images1=None, sample_images2=None, sample_labels1=None,batch_labels1=None, 
			sample_labels2=None,batch_labels2=None, sample_z=None, img_name=None):

	if sample_images1==None:
	    data_X1, data_y = self.load_mnist()
            data_X2 = self.load_invert_mnist()
    	    # sample noise
            sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))
            sample_images1 = data_X1[0:self.batch_size]
            sample_images2 = data_X2[0:self.batch_size]
            sample_labels = data_y[0:self.batch_size]
	    img_name = './evaluate/top/testing'

        samples1, d1_loss, g1_loss = self.sess.run(
                 [self.sampler1, self.d1_loss, self.g1_loss],
                 feed_dict={self.z: sample_z, self.images1: sample_images1, self.y:batch_labels1}
             )
        save_images(samples1[:self.sample_size], [8, 8], img_name, type='BW')
        print("[Sample T] d_loss: %.8f, g_loss: %.8f" % (d1_loss, g1_loss))

        # sample is the generated image
        samples2, d2_loss, g2_loss = self.sess.run(
                 [self.sampler2, self.d2_loss, self.g2_loss],
                 feed_dict={self.z: sample_z, self.images2: sample_images2, self.y:batch_labels2}
             )
        save_images(samples2[:self.sample_size], [8, 8], img_name.replace('top', 'bot'), type='BW')
        print("[Sample B] d_loss: %.8f, g_loss: %.8f" % (d2_loss, g2_loss))

    def discriminator(self, image, y=None, share_params=False, reuse=False, name='D'):

	# select the corresponding batchnorm1(not shared)
        if '1' in name:
            d_bn1 = self.d1_bn1
	    branch = '1'
        elif '2' in name:
            d_bn1 = self.d2_bn1
	    branch = '2'

        # layers that don't share variable
	with tf.variable_scope(name):
	    if reuse:
		tf.get_variable_scope().reuse_variables()

            h0 = prelu(conv2d(image, self.c_dim, name='d'+branch+'_h0_conv', reuse=False), 
					name='d'+branch+'_h0_prelu', reuse=False)

            h1 = prelu(d_bn1(conv2d(h0, self.df_dim, name='d'+branch+'_h1_conv', reuse=False), reuse=reuse), 
					name='d'+branch+'_h1_prelu', reuse=False)
            h1 = tf.reshape(h1, [self.batch_size, -1])            

        # layers that share variables
        h2 = prelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin', reuse=share_params),reuse=share_params), 
					name='d_h2_prelu', reuse=share_params)

        h3 = linear(h2, 1, 'd_h3_lin', reuse=share_params)
            
        return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None, share_params=False, reuse=False, name='G'):

	if '1' in name:
            branch = '1'
        elif '2' in name:
            branch = '2'

	# layers that share the variables 
        s = self.output_size
        s2, s4 = int(s/2), int(s/4) 
        h0 = prelu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin', reuse=share_params), reuse=share_params), 
						name='g_h0_prelu', reuse=share_params)

        h1 = prelu(self.g_bn1(linear(z, self.gf_dim*2*s4*s4,'g_h1_lin',reuse=share_params),reuse=share_params),
						name='g_h1_prelu', reuse=share_params)
        h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])

        h2 = prelu(self.g_bn2(deconv2d(h1, [self.batch_size,s2,s2,self.gf_dim * 2], 
			name='g_h2', reuse=share_params), reuse=share_params), name='g_h2_prelu', reuse=share_params)

	# layers that don't share the variable
	with tf.variable_scope(name):
	    if reuse:
		tf.get_variable_scope().reuse_variables()
 	    output = tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g'+branch+'_h3', reuse=False))

        return output

    def load_invert_mnist(self):
	data_dir = os.path.join(os.path.join("./data", self.dataset_name, 'invert'))

	trX = np.load(os.path.join(data_dir, 'train-images-idx3-ubyte.npy'))
	teX = np.load(os.path.join(data_dir, 't10k-images-idx3-ubyte.npy'))

        X = np.concatenate((trX, teX), axis=0)
        # convert to 0~1 is more convenient
        return X/255.

    def load_mnist(self):

        data_dir = os.path.join("./data", self.dataset_name)
        
        fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)
        
        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0)

	# convert label into one-hot
        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i,int(y[i])] = 1.0
        
	# conver to 0~1 is more convenient
        return X/255.,y_vec
            
    def save(self, checkpoint_dir, step):
        model_name = "CoGAN.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
