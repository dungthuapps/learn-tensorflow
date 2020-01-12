"""Concept 3 - RNN
(Exercise of Section 8).

Data:
    https://keras.io/examples/cifar10_cnn/ (small)
    https://www.tensorflow.org/datasets/overview


# Procedures
    1. Load data with one-hot encoding
    2. First Insight of data
    3. Build Models
        1. PlaceHolders of Input X and output y_true
            # ! Notice: X ~ [None, number_pixels]
            - None means to tell TF that, we do not how many images yet, will be done in run time
            - same to y_true
        2. Layer 1 - resize x -> [batch_size, h, w, c]
        3. Layer 2: CNN block 1 (Variable of Weights)
            - neurons: tf.nn.conv2d
            - activations: tf.nn.relu
        4. Flatten of CNN -> DNN (reshape)
        5. DNN layer:
            - neurons: y = w * x + b
            - activations: tf.nn.relu
        6. Dropout Layer:
            - create placeholder to get input from feed_dict
            - tf.nn.dropout
        7. Similar we have flow:
            X -> resize X -> CNN-1 -> MaxPool-1 -> CNN-2 ->
                -> MaxPool-2 -> Flatten (reshape) -> DNN -> Droptout -> Softmax
        8. Cost Function
            - tf.nn.softmax_cross_entropy_with_logits
        9. Lost Function
            - tf.reduce_mean
        10. Optimization (Gradient Descent, back ward update)
            - tf.train.AdamOptimizer
            - optimizer.minimize(.)
    4. Run model (Train + Validation for each epoch)
        1. Specify epochs and batch size
        2. Init global variables (graphs)
        3. Create a session
        4. Inside the session, create a feed_dict to place holders
            2. for each epoch in the session,
                1. feed_dict = random_batch_of (x, y_true, and prob_of_drop_out )
                2. train = run/re-run session with feed_dict
                4. validation for each run
                    1. get number of matches
                    2. get accuracy
                        - tf.reduce_mean of matches
                        - run session with eval_feed_dict

"""