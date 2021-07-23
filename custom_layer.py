import re
import tensorflow as tf


def matmul(a, b):
    output = tf.linalg.matmul(
        a, b, transpose_a=False, transpose_b=True, adjoint_a=False, adjoint_b=False,
        a_is_sparse=False, b_is_sparse=False, name=None
    )
    
    
    return output
def tf_reduce_sum(x):
    output = tf.reduce_sum(x, 1)
    return output
def tf_expand_dims(x):
    output = tf.expand_dims(x, -1)
    return output
    
a = tf.random.uniform(
    (128, 3000, 1), minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None, name=None
)

b = tf.random.uniform(
    (128, 3000, 1), minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None, name=None
)

out = matmul(a, b)
print(out.shape)