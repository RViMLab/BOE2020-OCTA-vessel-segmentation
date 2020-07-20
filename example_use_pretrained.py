from manager import ModelManager

# 1) choose a pretrained model

# *note*: the directory's naming convention reflects the settings of the pretrained model
# for example we choose our top-performing method based on iUNET with:
# L(num_layers) = 3, F(feature_maps_root)=64, loss = 'i-bce-topo', iters(refinement iterations) = 4 etc.

model_ckpt_dir = 'pretrained_models/iUNET/L_3_F_64_loss_i-bce-topo_C1_2_C2_2_C3_4_0.01_0.001_0.0001_iters_4'

# 2) choose a directory where octa images are stored

# here we use a directory containing publicly available images downloaded for demonstration purposes
# These images are not used for training and present significant differences compared to our training data:
# scale, imaging device, being a montage of two or more seperate scans
# Our model generalizes reasonably well on these unseen images,
# please see results in the 'segmentations_iunet' directory

path_to_files = 'web_octa_images'

# 3) define the model's graph using the manager object:

#   a) the 'num_layers' argument controls the number of residual blocks the encoder and decoder consists of
#   b) the 'feature_maps_root' argument controls the number filters in the first residual block of the encoder
#   The two above arguments should match the pretrained model's settings.
#   c) Also specify the number of refinement iterations of the iUNET by setting n_iterations
# *note*: for the iUNET model this can be set to any arbitrary value higher or lower than the one used during training.
#         For this example n_iterations was 4 during training but we can set n_iterations = 6
#         in the hope that further refinement iterations improve the segmentation.
#         As a demo we employ 6 refinement iterations
#         using the iUNET model which was trained with only 4 refinement iterations.
#         Please see  the 'segmentations_iunet' directory for results

manager = ModelManager(name='iUNET', n_iterations=6, num_layers=3, feature_maps_root=64)

# 4) choose if intermediate outputs are computed by setting 'get_intermediate_outputs=True'
#    choose if the outputs are visualized and overlayed on the input during execution by setting 'show_outputs=True'
#    choose if the outputs are stored in a directory specified by 'path_to_save_dir' (e.x 'segmentations_iunet')
#    note: either all if 'get_intermediate_outputs=True' else just the final output is stored
#    if the 'path_to_save_dir' argument is not specified no output is saved (defaults to None)

manager.run_on_images(path_to_files, model_ckpt_dir, get_intermediate_outputs=True, show_outputs=True, path_to_save_dir='segmentations_iunet')

########################################################################################################################

# for the SHN pretrained model with n_modules = 4 we would need the following 3 lines of code:
# model_ckpt_dir = 'pretrained_models/SHN/L_3_F_64_loss_s-bce-topo_C1_2_C2_2_C3_4_0.01_0.001_0.0001_modules_4'
# manager = ModelManager(name='SHN', n_modules=4, num_layers=3, feature_maps_root=64)
# manager.run_on_images(path_to_files, model_ckpt_dir, get_intermediate_outputs=True, show_outputs=True, path_to_save_dir='segmentations_shn')
