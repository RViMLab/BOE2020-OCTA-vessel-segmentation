from manager import ModelManager

# this is an example requiring the data be stored according to the format described in readers.py
model_ckpt_dir = 'models'
train_tfrecord = 'train_fold0.tfrecords'
valid_tfrecord = 'validation_fold0.tfrecords'

# get the manager object and define the graph of the network using default network settings : see ModelManager.init()
# manager = ModelManager(name='iUNET', n_iterations=4, verbose=False)
manager = ModelManager(name='SHN', n_modules=2, verbose=False, num_layers=2)

# using the default setttings described in ModelManager.train() method
# manager.train(train_tfrecord=train_tfrecord, validation_tfrecord=valid_tfrecord,
#               loss_type='i-bce-topo', model_dir=model_ckpt_dir)

manager.train(train_tfrecord=train_tfrecord, validation_tfrecord=valid_tfrecord,
              loss_type='s-bce-topo', model_dir=model_ckpt_dir)
