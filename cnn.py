# -*- coding: utf-8 -*
from tensorflow.contrib.rnn import DropoutWrapper
from utils import *


BATCH_SIZE = config.FLAGS.batch_size
unit_num = embeddings_size         # 默认词向量的大小等于RNN(每个time step) 和 CNN(列) 中神经单元的个数, 为了避免混淆model中全部用unit_num表示。
time_step = max_sequence      # 每个句子的最大长度和time_step一样,为了避免混淆model中全部用time_step表示。
filter_sizes = [3, 5, 7]
filter_num = 128
DROPOUT_RATE = config.FLAGS.dropout
EPOCH = config.FLAGS.epoch
TAGS_NUM = get_class_size()


class NER_net:
    def __init__(self, scope_name, iterator, embedding, batch_size):
        '''
        :param scope_name:
        :param iterator: 调用tensorflow DataSet API把数据feed进来。
        :param embedding: 提前训练好的word embedding
        :param batch_size:
        '''
        self.batch_size = batch_size
        self.embedding = embedding
        self.iterator = iterator
        with tf.variable_scope(scope_name) as scope:
            self._build_net()

    def _build_net(self):
        self.global_step = tf.Variable(0, trainable=False)
        source = self.iterator.source
        tgt = self.iterator.target_input
        # 得到当前batch的长度（如果长度不足的会被padding填充）
        max_sequence_in_batch = self.iterator.source_sequence_length
        max_sequence_in_batch = tf.reduce_max(max_sequence_in_batch)
        max_sequence_in_batch = tf.to_int32(max_sequence_in_batch)

        # x: [batch_size, time_step, embedding_size], float32
        self.x = tf.nn.embedding_lookup(self.embedding, source)
        # y: [batch_size, time_step]
        self.y = tgt

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            # conv: [batch_size, time_step, 128]
            conv = tf.layers.conv1d(self.x, filters=filter_num, kernel_size=filter_size, activation=tf.nn.relu, padding="same")
            pooled = tf.layers.max_pooling1d(conv, pool_size=2, strides=1, padding="same")
            pooled_outputs.append(pooled)
        # [batch_size, time_step, 384]
        pooled = tf.concat(pooled_outputs, 2)
        # [batch_size, 384*time_step]
        #flat1 = tf.layers.flatten(pooled)
        # [batch_size*time_step, 384]
        x_reshape = tf.reshape(pooled, [-1, len(filter_sizes)*filter_num])
        drop1 = tf.layers.dropout(x_reshape)
        # projection: [None, TAGS_NUM]
        projection = tf.layers.dense(drop1, units=TAGS_NUM, activation=tf.nn.relu, kernel_regularizer=tf.nn.l2_loss, bias_regularizer=tf.nn.l2_loss)

        # 前面只是
        # -1 to time step
        self.outputs = tf.reshape(projection, [self.batch_size, time_step, TAGS_NUM])

        self.seq_length = tf.convert_to_tensor(self.batch_size * [max_sequence_in_batch], dtype=tf.int32)
        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.outputs, self.y, self.seq_length)

        # Add a training op to tune the parameters.
        self.loss = tf.reduce_mean(-self.log_likelihood)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)



def train(net, iterator, sess):
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt is not None:
        path = ckpt.model_checkpoint_path
        print 'loading pre-trained model from %s.....' % path
        saver.restore(sess, path)

    current_epoch = sess.run(net.global_step)
    while True:
        if current_epoch > EPOCH: break
        try:
            tf_unary_scores, tf_transition_params, _, losses = sess.run(
                [net.outputs, net.transition_params, net.train_op, net.loss])

            if current_epoch % 100 == 0:
                print '*' * 100
                print current_epoch, 'loss', losses
                print '*' * 100

            # 每隔10%的进度则save一次。
            if current_epoch % (EPOCH / 10) == 0 and current_epoch != 0:
                sess.run(tf.assign(net.global_step, current_epoch))
                saver.save(sess, model_path+'points', global_step=current_epoch)

            current_epoch += 1

        except tf.errors.OutOfRangeError:
            sess.run(iterator.initializer)
        except tf.errors.InvalidArgumentError:
            # iterator.next() cannot get enough data to a batch, initialize it.
            # 正常初始化流程
            sess.run(iterator.initializer)
    print 'training finished!'


def predict(net, tag_table, sess):
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt is not None:
        path = ckpt.model_checkpoint_path
        print 'loading pre-trained model from %s.....' % path
        saver.restore(sess, path)
    else:
        print 'Model not found, please train your model first'
        return

    # 获取原文本的iterator
    file_iter = file_content_iterator(pred_file)

    while True:
        # batch等于1的时候本来就没有padding，如果批量预测的话，记得这里需要做长度的截取。
        try:
            tf_unary_scores, tf_transition_params = sess.run(
                [net.outputs, net.transition_params])
        except tf.errors.OutOfRangeError:
            print 'Prediction finished!'
            break

        # 把batch那个维度去掉
        tf_unary_scores = np.squeeze(tf_unary_scores)

        viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
            tf_unary_scores, tf_transition_params)
        tags = []
        '''
        for id in viterbi_sequence:
            tags.append(sess.run(tag_table.lookup(tf.constant(id, dtype=tf.int64))))
        '''
        tags = sess.run(tag_table.lookup(tf.constant(viterbi_sequence, dtype=tf.int64)))
        write_result_to_file(file_iter, tags)


if __name__ == '__main__':

    action = config.FLAGS.action
    # 获取词的总数。
    vocab_size = get_src_vocab_size()
    src_unknown_id = tgt_unknown_id = vocab_size
    src_padding = vocab_size + 1

    src_vocab_table, tgt_vocab_table = create_vocab_tables(src_vocab_file, tgt_vocab_file, src_unknown_id,
                                                           tgt_unknown_id)
    embedding = load_word2vec_embedding(vocab_size, False, True)

    if action == 'train':
        iterator = get_iterator(src_vocab_table, tgt_vocab_table, vocab_size, BATCH_SIZE)
    elif action == 'predict':
        BATCH_SIZE = 1
        DROPOUT_RATE = 1.0
        iterator = get_predict_iterator(src_vocab_table, vocab_size, BATCH_SIZE)
    else:
        print 'Only support train and predict actions.'
        exit(0)

    tag_table = tag_to_id_table()
    net = NER_net("ner", iterator, embedding, BATCH_SIZE)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        tf.tables_initializer().run()

        if action == 'train':
            train(net, iterator, sess)
        elif action == 'predict':
            predict(net, tag_table, sess)