



class ConvVAE(object):

    def __init__(self, counters={}):
        self.counters = counters

    def build_graph(self, x, is_training, z_dim, use_mode="test", reg='mmd', beta=1., lam=0., nonlinearity=tf.nn.elu, bn=True, kernel_initializer=None, kernel_regularizer=None):
        self.z_dim = z_dim
        self.use_mode = use_mode
        self.nonlinearity = nonlinearity
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bn = bn
        self.reg = reg
        self.beta = beta
        self.lam = lam
        self.__model(x, is_training)
        self.__loss(self.reg)

    def __model(self, x, is_training):
        print("******   Building Graph   ******")
        self.x = x
        self.is_training = is_training
        if int_shape(x)[1]==64:
            encoder = conv_encoder_64_block
            decoder = conv_decoder_64_block
        elif int_shape(x)[1]==32:
            encoder = conv_encoder_32_block
            decoder = conv_decoder_32_block
        with arg_scope([encoder, decoder], nonlinearity=self.nonlinearity, bn=self.bn, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer, is_training=self.is_training, counters=self.counters):
            self.z_mu, self.z_log_sigma_sq = encoder(x, self.z_dim)
            sigma = tf.exp(self.z_log_sigma_sq / 2.)
            if self.use_mode=='train':
                self.z = z_sampler(self.z_mu, sigma)
            elif self.use_mode=='test':
                self.z = tf.placeholder(tf.float32, shape=int_shape(self.z_mu))
            print("use mode:{0}".format(self.use_mode))
            self.x_hat = decoder(self.z)


    def __loss(self, reg):
        print("******   Compute Loss   ******")
        self.loss_ae = tf.reduce_mean(tf.reduce_sum(tf.square(flatten(self.x)-flatten(self.x_hat)), 1))
        # self.loss_ae = tf.reduce_mean(tf.square(flatten(self.x)-flatten(self.x_hat)))
        if reg is None:
            self.loss_reg = 0
        elif reg=='kld':
            self.loss_reg = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=-1))
            self.loss_reg = self.beta * tf.maximum(self.lam, self.loss_reg)
        elif reg=='mmd':
            self.loss_reg = compute_mmd(tf.random_normal(int_shape(self.z)), self.z)
            self.loss_reg = self.beta * tf.maximum(self.lam, self.loss_reg)
        elif reg=='tc':
            #tc = compute_tc(self.z, self.z_mu, self.z_log_sigma_sq)
            from layers import estimate_tc, estimate_dwkld, estimate_mi
            N = 200000
            mi = estimate_mi(self.z, self.z_mu, self.z_log_sigma_sq, N=N)
            tc = estimate_tc(self.z, self.z_mu, self.z_log_sigma_sq, N=N)
            dwkld = estimate_dwkld(self.z, self.z_mu, self.z_log_sigma_sq, N=N)
            #kld = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=-1))
            self.loss_reg = mi + self.beta * tc + dwkld  #kld + (self.beta-1.) * tc

        print("reg:{0}, beta:{1}, lam:{2}".format(self.reg, self.beta, self.lam))
        self.loss = self.loss_ae + self.loss_reg
