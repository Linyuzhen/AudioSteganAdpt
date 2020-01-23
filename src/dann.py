import numpy as np
from tensorflow import keras
from Grad_Reverse_Layer import GradientReversal

class HPF_Layer(keras.layers.Layer):
    def __init__(self,filters,kernel_size,hpf_kernel,is_train=True,**kwargs):
        super(HPF_Layer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.hpf_kernel = hpf_kernel
        self.is_train = is_train

    def build(self, input_shape):
        k = np.load(self.hpf_kernel)
        k = k.reshape([self.filters,self.kernel_size,1])
        # print(k)
        self.hpf = keras.layers.Conv1D(self.filters,self.kernel_size,
                                  kernel_initializer=keras.initializers.constant(k),
                                  trainable=self.is_train,
                                  padding='same')
        super(HPF_Layer, self).build(input_shape)


    def call(self, inputs, **kwargs):
        return self.hpf(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

# def TLU(x,th=3.0):
#     return keras.backend.minimum(keras.backend.maximum(x,-th),th)

class DANN(object):
    def __init__(self,sample_length,channels,classes=2,features=128,grl='auto',summary=True,model_plot=False,batch_size=64):
        self.learning_phase = keras.backend.variable(1)
        self.domain_invariant_features = None
        self.sample_length = sample_length
        self.channels = channels
        self.input_shape = (sample_length,channels)
        self.classes = classes
        self.features = features
        self.batch_size = batch_size
        self.grl = 'auto'
        # Set reversal gradient value.
        if grl is 'auto':
            self.grl_rate = 1.0
        else:
            self.grl_rate = grl
        self.summary = summary
        self.model_plot = model_plot

        # Build the model
        self.model = self.build_model()

        # Print and Save the model summary if requested.
        if self.summary:
            self.model.summary()
        if self.model_plot:
            keras.utils.plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


    def feature_extractor(self, inputs):

        hpf = HPF_Layer(filters=4,kernel_size=5,hpf_kernel='SRM_k.npy',is_train=True,name='HPF_layer')(inputs)

        # Group 1
        gp1 = keras.layers.Conv1D(8, 1, padding='same',name='Group1_conv1')(hpf)
        gp1 = keras.layers.Conv1D(8, 5, padding='same',name='Group1_conv2')(gp1)
        gp1 = keras.layers.Conv1D(16, 1, padding='same',name='Group1_output')(gp1)

        # Group 2
        gp2 = keras.layers.Conv1D(16, 5, padding='same',activation='relu',name='Group2_conv1')(gp1)
        gp2 = keras.layers.Conv1D(32, 1, padding='same',activation='relu',name='Group2_conv2')(gp2)
        gp2 = keras.layers.AveragePooling1D(pool_size=3,strides=2,padding='same',name='Group2_output')(gp2)

        # Group 3
        gp3 = keras.layers.Conv1D(32, 5, padding='same', activation='relu',name='Group3_conv1')(gp2)
        gp3 = keras.layers.Conv1D(64, 1, padding='same', activation='relu',name='Group3_conv2')(gp3)
        gp3 = keras.layers.AveragePooling1D(pool_size=3, strides=2, padding='same',name='Group3_output')(gp3)

        # Group 4
        gp4 = keras.layers.Conv1D(64, 5, padding='same', activation='relu',name='Group4_conv1')(gp3)
        gp4 = keras.layers.Conv1D(128, 1, padding='same', activation='relu',name='Group4_conv2')(gp4)
        gp4 = keras.layers.AveragePooling1D(pool_size=3, strides=2, padding='same',name='Group4_output')(gp4)

        # Group 5
        gp5 = keras.layers.Conv1D(128, 5, padding='same', activation='relu',name='Group5_conv1')(gp4)
        gp5 = keras.layers.Conv1D(256, 1, padding='same', activation='relu',name='Group5_conv2')(gp5)
        gp5 = keras.layers.AveragePooling1D(pool_size=3, strides=2, padding='same',name='Group5_output')(gp5)

        # Group 6
        gp6 = keras.layers.Conv1D(256, 5, padding='same', activation='relu',name='Group6_conv1')(gp5)
        gp6 = keras.layers.Conv1D(512, 1, padding='same', activation='relu',name='Group6_conv2')(gp6)
        gp6 = keras.layers.GlobalAveragePooling1D(name='Group6_output')(gp6)

        f_output = gp6
        # f_output = keras.layers.Dense(self.features,activation='relu',name='feature_outputs')(gp6)
        self.domain_invariant_features = f_output
        return f_output

    def classifier(self, inputs):
        # c = keras.layers.Dense(self.features//2, activation='relu',name='steganalysis_c1')(inputs)
        # # c = keras.layers.Dense(self.features//4,activation='relu',name='steganalysis_c2')(c)
        # c = keras.layers.Dropout(0.5)(c)
        c_output = keras.layers.Dense(self.classes, activation='softmax', name="classifier_output")(inputs)
        return c_output

    def domain_discriminator(self, inputs):
        d = keras.layers.Dense(self.features//2, activation='relu',name='domain_d')(inputs)
        d = keras.layers.Dropout(rate=0.5)(d)
        d_output = keras.layers.Dense(2, activation='softmax', name="discriminator_output")(d)
        return d_output

    def build_model(self):
        audio_inputs = keras.layers.Input(shape=self.input_shape, name='main_input')
        feature_output = self.feature_extractor(audio_inputs)
        self.grl_layer = GradientReversal(1.0)
        feature_output_grl = self.grl_layer(feature_output)
        # labeled_feature_output = keras.layers.Lambda(lambda x: keras.backend.switch(keras.backend.learning_phase(),keras.backend.concatenate([x[:int(self.batch_size // 2)],x[:int(self.batch_size // 2)]],axis=0), x),output_shape=lambda x: x[0:])(feature_output_grl)
        d_feature_output = keras.layers.Lambda(lambda x: keras.backend.switch(keras.backend.learning_phase(),keras.backend.concatenate([x[:int(self.batch_size // 2)],x[:int(self.batch_size // 2)]],axis=0), x),output_shape=lambda x: x[0:])(feature_output_grl)

        classifier_output = self.classifier(feature_output)
        discriminator_output = self.domain_discriminator(d_feature_output)
        model = keras.models.Model(inputs=audio_inputs, outputs=[discriminator_output, classifier_output])
        # model = keras.utils.multi_gpu_model(model,gpus=2)
        return model

    def batch_generator(self, trainX, trainY=None, batch_size=64, shuffle=True):
        if shuffle:
            index = np.random.randint(0, len(trainX) - batch_size)
        else:
            index = np.arange(0, len(trainX), batch_size)
        while trainX.shape[0] > index + batch_size:
            batch_audios = trainX[index: index + batch_size]
            batch_audios = batch_audios.reshape(batch_size, self.sample_length, self.channels)
            if trainY is not None:
                batch_labels = trainY[index: index + batch_size]
                yield batch_audios, batch_labels
            else:
                yield batch_audios
            index += batch_size

    def compile(self,optimizer):
        self.model.compile(optimizer=optimizer,loss={'classifier_output':'binary_crossentropy','discriminator_output':'binary_crossentropy'},
                           loss_weights={'classifier_output':1.0,'discriminator_output':0.2})

    def train(self,trainX,trainDX,trainY=None,epochs=1,batch_size=1,verbose=True,save_model=None):
        # self.compile(keras.optimizers.Adam(lr=0.001))
        # self.model.fit(x={'main_input':np.concatenate((trainX,trainDX))},
        #                y={'classifier_output': np.concatenate((trainY,trainY)), 'discriminator_output':np.concatenate((np.tile([0, 1], [trainX.shape[0], 1]), np.tile([1, 0], [trainDX.shape[0], 1])))},
        #                epochs=epochs,batch_size=batch_size,verbose=verbose)

        for cnt in range(epochs):
            Labeled = self.batch_generator(trainX,trainY,batch_size=batch_size//2)
            UNLabeled = self.batch_generator(trainDX, batch_size=batch_size // 2)

            p = np.float(cnt)/epochs
            lr = 0.0001
            # lr = 0.0001 / (1. + 10 * p)**0.75

            # Settings for reverse gradient magnitude (if it's set to be automatically calculated, otherwise set by user.)
            if self.grl is 'auto':
                self.grl_layer.l = 2. / (1. + np.exp(-10. * p)) - 1

            self.compile(keras.optimizers.Adam(lr))

            for batchX,batchY in Labeled:
                try:
                    batchDX = next(UNLabeled)
                except:
                    UNLabeled = self.batch_generator(trainDX, batch_size=batch_size // 2)

                # Combine the labeled and unlabeled audios along with the discriminative results.
                combine_batchX = np.concatenate((batchX,batchDX))

                batch2Y = np.concatenate((batchY, batchY))
                combined_batchY = np.concatenate((np.tile([0, 1], [batchX.shape[0], 1]), np.tile([1, 0], [batchDX.shape[0], 1])))

                # Train the model
                metrics = self.model.train_on_batch({'main_input':combine_batchX},{'classifier_output': batch2Y, 'discriminator_output':combined_batchY})
                if verbose:
                                    print(
                                        "Epoch {}/{}\n\t[Generator_loss: {:.4}, Discriminator_loss: {:.4}, Classifier_loss: {:.4}]".format(
                                            cnt + 1, epochs, metrics[0], metrics[1], metrics[2]))

    def evaluate(self,testX,testY=None,weight_loc=None,save_pred=None, verbose=0):
        if weight_loc is not None:
            self.compile(keras.optimizers.Adam())
            self.model.load_weights(weight_loc)
        _, yhat_class = self.model.predict(testX, verbose=verbose)
        if save_pred is not None:
            np.save(save_pred, yhat_class)
        if testY is not None and len(testY) == 2:
            acc = self.evaluate(testX, testY, verbose=verbose)
            # if verbose==True:
            print("The classifier and discriminator metrics for evaluation are [{}, {}]".format(acc[0], acc[1]))
        elif testY is not None and len(testY) == 1:
            acc = self.model.evaluate(testX, [np.ones((testY.shape[0], 2)), testY], verbose=verbose)
            # if verbose==True:
            print("The classifier metric for evaluation is {}".format(acc[1]))



