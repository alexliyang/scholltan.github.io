from u_net_resnet_v1 import get_unet
from data_loader import get_data_generators
from keras.callbacks import TensorBoard, ModelCheckpoint, ProgbarLogger
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

from keras.callbacks import LearningRateScheduler
 

# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.6
	epochs_drop = 5.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate
	
lrate = LearningRateScheduler(step_decay)


def main(args):
    img_rows = 192
    img_cols = 336
    batch_size = 10
    n_epochs = 50
    models_folder = 'models'
    model_name = 'w_net_V12'
    model_path = os.path.join(models_folder, model_name)

    train_generator, val_generator, training_samples, val_samples = get_data_generators(train_folder='/data/stereopairs/train',
                                                                                        val_folder='/data/stereopairs/validation',
                                                                                        img_rows=img_rows,
                                                                                        img_cols=img_cols,
                                                                                        batch_size=batch_size)

    print('found {} training samples and {} validation samples'.format(training_samples, val_samples))
    print('...')
    print('building model...')

    w_net, disp_map_model = get_unet(img_rows=img_rows, img_cols=img_cols, lr=1e-2)

    w_net.compile(optimizer=Adam(lr=lr, decay=0.01),
              loss={'output': 'mean_absolute_error', 'loss_ssim_recon': loss_DSSIM, 
              'weighted_gradient_left':'mean_absolute_error', 'weighted_gradient_right':'mean_absolute_error'},
              loss_weights={'output': 1., 'loss_ssim_recon': 0.075, 'weighted_gradient_left':0.001,'weighted_gradient_right':0.001})
    w_net.summary()


    print('saving model to {}...'.format(model_path))
    model_yaml = w_net.to_yaml()
    with open(model_path + ".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)

    print('begin training model, {} epochs...'.format(n_epochs))
    for epoch in range(n_epochs):

        print('epoch {} \n'.format(epoch))

        model_path = os.path.join(models_folder, model_name + '_epoch_{}'.format(epoch))
        w_net.fit_generator(train_generator,
                            steps_per_epoch=training_samples // batch_size,
                            epochs=1,
                            validation_data=val_generator,
                            validation_steps=val_samples // batch_size,
                            verbose=1,
                            callbacks=[lrate, TensorBoard(log_dir='/tmp/deepdepth'),
                                       ModelCheckpoint(model_path + '.h5', monitor='loss',
                                                       verbose=0,
                                                       save_best_only=False,
                                                       save_weights_only=False,
                                                       mode='auto', period=1)])
        # print()

if __name__ == '__main__':
    main(None)
