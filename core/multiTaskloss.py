import tensorflow.compat.v1 as tf



class MultiLossLayer():
    def __init__(self, loss_list):
        self._loss_list = loss_list
        self._sigmas_sq = []
        for i in range(len(self._loss_list)):
            self._sigmas_sq.append(tf.get_variable('Sigma_sq_' + str(i), dtype=tf.float32, shape=[],
                                                 initializer=tf.initializers.random_uniform(minval=0.2, maxval=1)))

    def get_loss(self):
        factor = tf.div(1.0, tf.multiply(2.0, self._sigmas_sq[0]))
        loss = tf.add(tf.multiply(factor, self._loss_list[0]), tf.log(self._sigmas_sq[0]))
        for i in range(1, len(self._sigmas_sq)):
            factor = tf.div(1.0, tf.multiply(2.0, self._sigmas_sq[i]))
            loss = tf.add(loss, tf.add(tf.multiply(factor, self._loss_list[i]), tf.log(self._sigmas_sq[i])))
        return loss