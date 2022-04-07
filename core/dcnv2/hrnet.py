import tensorflow.compat.v1 as tf

from core.dcnv2.dcnv2 import DCNv2


def basic_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    x = cbr(input, out_filters, 3, strides)
    x = cbr(x, out_filters, 3, 1, activate=False)

    if with_conv_shortcut:
        residual = cbr(input, out_filters, 1, strides, activate=False)
        x = tf.add(x, residual)
    else:
        x = tf.add(x, input)
    x = tf.nn.relu(x)
    return x


def bottleneck_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    expansion = 4
    de_filters = int(out_filters / expansion)
    x = cbr(input, de_filters, 1, 1)
    x = cbr(x, de_filters, 3, strides)
    x = cbr(x, out_filters, 1, 1, activate=False)
    if with_conv_shortcut:
        residual = cbr(input, out_filters, 1, strides, activate=False)
        x = tf.add(x, residual)
    else:
        x = tf.add(x, input)
    x = tf.nn.relu(x)
    return x


def cbr(input, filters, kernel_size, strides, activate=True, use_dcn=False):
    if use_dcn:
        pad2 = (kernel_size - 1) // 2
        x = DCNv2(input.shape[-1], filters, filter_size=kernel_size, stride=strides, padding=pad2,
                  bias_attr=False,
                  )(input)
    else:
        x = tf.layers.conv2d(input, filters, kernel_size, strides=strides, padding='SAME')
    x = tf.layers.batch_normalization(x, beta_initializer=tf.zeros_initializer(),
                                      gamma_initializer=tf.ones_initializer(),
                                      moving_mean_initializer=tf.zeros_initializer(),
                                      moving_variance_initializer=tf.ones_initializer())
    if activate:
        x = tf.nn.relu(x)
    return x


def stem_net(input):
    x = cbr(input, 36, 3, (2, 2))
    x = cbr(x, 36, 3, (2, 2))
    x = bottleneck_Block(x, 144, with_conv_shortcut=True)
    for i in range(3):
        x = bottleneck_Block(x, 144, with_conv_shortcut=False)
    return x


def transition_layer1(x, out_filters_list=[18, 36]):
    x0 = cbr(x, out_filters_list[0], 3, 1)
    x1 = cbr(x, out_filters_list[1], 3, (2, 2))
    return [x0, x1]


def make_branch(x, out_filters, branch=4):
    for i in range(branch):
        x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def upsample(input_data, kernel_size=4, strides=(2, 2)):
    numm_filter = input_data.shape.as_list()[-1]
    output = cbr(input_data, numm_filter, 3, 1, use_dcn=True)
    output = tf.layers.conv2d_transpose(output, numm_filter, kernel_size=kernel_size, padding='same',
                                        strides=strides, kernel_initializer=tf.random_normal_initializer())
    output = cbr(output, numm_filter, 3, 1, use_dcn=True)
    return output


def fuse_layer1(x):
    x0_0 = x[0]
    x0_1 = cbr(x[1], 18, 1, 1, activate=False)
    x0_1 = upsample(x0_1)
    x0 = tf.add(x0_0, x0_1)
    x1_0 = cbr(x[0], 36, 3, (2, 2), activate=False)
    x1_1 = x[1]
    x1 = tf.add(x1_0, x1_1)
    return [x0, x1]


def transition_layer2(x, out_filters_list=[18, 36, 72]):
    x0 = cbr(x[0], out_filters_list[0], 3, 1)
    x1 = cbr(x[1], out_filters_list[1], 3, 1)
    x2 = cbr(x[1], out_filters_list[2], 3, (2, 2))
    return [x0, x1, x2]


def fuse_layer2(x):
    x0_0 = x[0]
    x0_1 = cbr(x[1], 18, 1, 1, activate=False)
    x0_1 = upsample(x0_1)
    x0_2 = cbr(x[2], 18, 1, 1, activate=False)
    x0_2 = upsample(x0_2, kernel_size=8, strides=(4, 4))
    x0 = tf.concat([x0_0, x0_1, x0_2], axis=-1)
    return x0


def seg_hrnet(inputs):
    x = stem_net(inputs)
    x = transition_layer1(x)
    x0 = make_branch(x[0], 18)
    x1 = make_branch(x[1], 36)
    x = fuse_layer1([x0, x1])
    #
    x = transition_layer2(x)
    x0 = make_branch(x[0], 18)
    x1 = make_branch(x[1], 36)
    x2 = make_branch(x[2], 72)
    x = fuse_layer2([x0, x1, x2])
    x = cbr(x, 128, 3, 1)
    return x

if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, shape=(None, 400, 400, 3))
    model = seg_hrnet(inputs)
    print(model)
