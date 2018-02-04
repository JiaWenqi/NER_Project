# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
'''
If we have a vocabulary file "test.txt"
with the following content:
```
emerson
lake
palmer
```
```
'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #sess.run(iterator.initializer)
    tf.tables_initializer().run()
    vocabulary_list = tf.constant(["emerson", "lake", "palmer"])
    print(vocabulary_list)
    indices = tf.constant([0, 2], tf.int64)
    table = tf.contrib.lookup.index_to_string_table_from_tensor(vocabulary_list, default_value="UNKNOWN")
    values = table.lookup(indices)
    ...
    tf.tables_initializer().run()
    values.eval()# == > ["lake", "UNKNOWN"]
    print(values.eval())
    print(type(values.eval()))
    print(type(values))


