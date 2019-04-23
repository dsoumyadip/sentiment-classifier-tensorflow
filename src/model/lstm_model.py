__author__ = "Soumyadip"

import tensorflow as tf

from src.utils.utils import embedding_initializer

NUM_CLASSES = 2


def lstm_with_attention(features, labels, mode, params):

    with tf.name_scope("embedding"):
        init_embeddings = tf.constant(
                                embedding_initializer(params['reverse_dict'], params['root_path'],
                                                      params['resources_path']), dtype=tf.float32)

        embeddings = tf.get_variable('embeddings', initializer=init_embeddings, trainable=False)
        x_embeddings = tf.nn.embedding_lookup(embeddings, features['x'])

    with tf.name_scope("bidirectional_lstm"):
        fw_cells = [tf.nn.rnn_cell.BasicLSTMCell(params['sequence_length']) for _ in range(params['num_layers'])]
        fw_cells = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=params['keep_prob']) for cell in fw_cells]
        bw_cells = [tf.nn.rnn_cell.BasicLSTMCell(params['sequence_length']) for _ in range(params['num_layers'])]
        bw_cells = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=params['keep_prob']) for cell in bw_cells]

        rnn_outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                                                fw_cells, bw_cells, inputs=x_embeddings,
                                                                sequence_length=features['len'],
                                                                dtype=tf.float32
                                                            )
    with tf.name_scope("attention"):
        attention_score = tf.nn.softmax(tf.contrib.slim.fully_connected(rnn_outputs, 1))
        attention_out = tf.squeeze(
            tf.matmul(tf.transpose(rnn_outputs, perm=[0, 2, 1]), attention_score),
            axis=-1)

    with tf.name_scope("output"):
        logits = tf.contrib.slim.fully_connected(attention_out, NUM_CLASSES, activation_fn=None)
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

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(params['learning_rate'])
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    with tf.name_scope("accuracy"):

        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=predictions["classes"])}
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)




