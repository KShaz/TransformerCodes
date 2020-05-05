import tensorflow as tf


# a = tf.GLUE_Download.is_gpu_available(
#     cuda_only=False, min_cuda_compute_capability=None
# )
#
# print(a)

if tf.test.gpu_device_name():

    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

else:

   print("Please install GPU version of TF")


