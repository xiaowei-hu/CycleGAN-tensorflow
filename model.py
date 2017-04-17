from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from collections import namedtuple

from module import *
from utils import *

class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir

        self.discriminator = discriminator
        if args.use_resnet:
            self.generator = generator_resnet
        else:
            self.generator = generator_unet
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc))

        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(maxsize=args.max_size)

    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                         name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")
        self.fake_A = self.generator(self.real_B, self.options, True, name="generatorA2B")
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorB2A")

        self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")
        self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
                            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
                            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
                            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
                            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                            self.output_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                            self.input_c_dim], name='fake_B_sample')
        self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
        self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")
        self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2

        self.g_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.db_sum = tf.summary.merge(
            [self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum]
        )
        self.da_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum]
        )

        self.test_A = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.output_c_dim], name='test_B')
        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")

        t_vars = tf.trainable_variables()
        self.db_vars = [var for var in t_vars if 'discriminatorB' in var.name]
        self.da_vars = [var for var in t_vars if 'discriminatorA' in var.name]
        self.g_vars_a2b = [var for var in t_vars if 'generatorA2B' in var.name]
        self.g_vars_b2a = [var for var in t_vars if 'generatorB2A' in var.name]
        for var in t_vars: print var.name

    def train(self, args):
        """Train cyclegan"""
        self.da_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                        .minimize(self.da_loss, var_list=self.da_vars)
        self.db_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                        .minimize(self.db_loss, var_list=self.db_vars)
        self.g_a2b_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                        .minimize(self.g_loss_a2b, var_list=self.g_vars_a2b)
        self.g_b2a_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                        .minimize(self.g_loss_b2a, var_list=self.g_vars_b2a)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(args.epoch):
            dataA = glob('./datasets/{}/*.jpg'.format(self.dataset_dir+'/trainA'))
            dataB = glob('./datasets/{}/*.jpg'.format(self.dataset_dir+'/trainB'))
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = zip(dataA[idx*self.batch_size:(idx+1)*self.batch_size],
                                  dataB[idx*self.batch_size:(idx+1)*self.batch_size])
                batch_images = [load_data(batch_file) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)

                # Forward G network
                fake_A, fake_B = self.sess.run([self.fake_A, self.fake_B],
                    feed_dict={ self.real_data: batch_images })
                [fake_A, fake_B] = self.pool([fake_A, fake_B])
                # Update G network
                _, summary_str = self.sess.run([self.g_a2b_optim, self.g_a2b_sum],
                    feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)
                # Update D network
                _, summary_str = self.sess.run([self.db_optim, self.db_sum],
                   feed_dict={ self.real_data: batch_images,
                               self.fake_B_sample: fake_B })
                self.writer.add_summary(summary_str, counter)
                # Update G network
                _, summary_str = self.sess.run([self.g_b2a_optim, self.g_b2a_sum],
                    feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)
                # Update D network
                _, summary_str = self.sess.run([self.da_optim, self.da_sum],
                   feed_dict={ self.real_data: batch_images,
                               self.fake_A_sample: fake_A})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f" \
                    % (epoch, idx, batch_idxs, time.time() - start_time))

                if np.mod(counter, 100) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, 1000) == 2:
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):
        dataA = glob('./datasets/{}/*.jpg'.format(self.dataset_dir+'/testA'))
        dataB = glob('./datasets/{}/*.jpg'.format(self.dataset_dir+'/testB'))
        np.random.shuffle(dataA)
        np.random.shuffle(dataB)
        batch_files = zip(dataA[:self.batch_size], dataB[:self.batch_size])
        sample_images = [load_data(batch_file, False, True) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)

        fake_A, fake_B = self.sess.run(
            [self.fake_A, self.fake_B],
            feed_dict={self.real_data: sample_images}
        )
        save_images(fake_A, [self.batch_size, 1],
                    './{}/A_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(fake_B, [self.batch_size, 1],
                    './{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.jpg'.format(self.dataset_dir+'/testA'))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/*.jpg'.format(self.dataset_dir+'/testB'))
        else:
            raise Exception, '--which_direction must be AtoB or BtoA'

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for sample_file in sample_files:
            print 'Processing image: '+sample_file
            sample_image = [load_test_data(sample_file)]
            sample_image = np.array(sample_image).astype(np.float32)
            if args.which_direction == 'AtoB':
                fake_B = self.sess.run(self.testB, feed_dict={self.test_A: sample_image})
                save_images(fake_B, [1, 1], '{}/A2B_{}' \
                    .format(args.test_dir, sample_file.split('/')[-1]))
            else:
                fake_A = self.sess.run(self.testA, feed_dict={self.test_B: sample_image})
                save_images(fake_A, [1, 1], '{}/B2A_{}' \
                    .format(args.test_dir, sample_file.split('/')[-1]))
