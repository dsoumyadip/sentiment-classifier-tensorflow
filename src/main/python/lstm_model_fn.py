# Main model function

import tensorflow as tf


def lstm_model_fn(features, labels, mode, params):
    print(features['x'])
    print(features['len'])
    # [batch_size x sentence_size x embedding_size]
    inputs = tf.contrib.layers.embed_sequence(
        features['x'], params['vocab_size'], params['embedding_size'],
        initializer=params['embedding_initializer'])

    # create an LSTM cell of size 200
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units = 200)

    # create the complete LSTM
    lstm_outputs, final_states = tf.nn.dynamic_rnn(
        lstm_cell, inputs, sequence_length=features['len'], dtype=tf.float32)

    # get the final hidden states of dimensionality [batch_size x sentence_size]
    rnn_outputs = final_states.h

    logits = tf.layers.dense(inputs=rnn_outputs, units=2)

    predictions = {

        "classes": tf.argmax(logits, 1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions,
                                          export_outputs={
                                              'predict': tf.estimator.export.PredictOutput(predictions)
                                          }
        )

    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(params['learning_rate'])
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



