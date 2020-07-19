from manager import ModelManager

# this is an example requiring the data be stored according to the format described in readers.py
model_ckpt_dir = 'models'
train_tfrecord = 'train_fold0.tfrecords'
valid_tfrecord = 'validation_fold0.tfrecords'

# get the manager object and define the graph of the network using default network settings : see ModelManager.init()
manager = ModelManager(name='iUNET', n_iterations=2, verbose=False)

# using the default training setttings  (see ModelManager.train() method)
# we set i-bce-topo to indicate the use of an iterative (i is associated with iUNET) loss
# comprising balanced cross entropy (bce) and topological (topo) loss terms
manager.train(train_tfrecord=train_tfrecord, validation_tfrecord=valid_tfrecord,
              loss_type='i-bce-topo', model_dir=model_ckpt_dir)

########################################################################################################################
# for defining and training a SHN model we would do the following:

# manager = ModelManager(name='SHN', n_modules=2, verbose=False, num_layers=2)

# we set s-bce-topo to indicate the use of an iterative (s is associated with SHN) loss
# comprising balanced cross entropy (bce) and topological (topo) loss terms

# manager.train(train_tfrecord=train_tfrecord, validation_tfrecord=valid_tfrecord,
#               loss_type='s-bce-topo', model_dir=model_ckpt_dir)
