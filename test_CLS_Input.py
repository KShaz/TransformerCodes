import tensorflow as tf


# [ batch, max_len_pos]
arg_max = tf.constant([[1,2,3,4,5,0],
                       [6,7,8,0,0,0],
                       [13,14,15,16,17,18]
                      ])


position_hot = tf.constant([[0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,1,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0],
                [0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0]])

position_list = tf.constant([[6,13,15,16, -1, -1],
                             [3, 9,13, -1, -1, -1],
                             [1, 4, 5,10,15,18]])


c = tf.one_hot(position_list, 21) # [batch, max_len_pos, seq_len]
c = tf.cast(c, dtype=tf.int32)
arg_max = tf.expand_dims(arg_max, 2) #  [batch, max_len_pos, 1]
# arg_max = tf.tile(arg_max, [1,1, 21])
print("tf.expand_dims(arg_max)",arg_max)
print("tf.one_hot(position_list)", c)
result = arg_max * c
print(result)
print(tf.reduce_sum(result, axis=-2))
