import mxnet as mx

import config
import utils

def dice_loss(softmax, label, smooth=1.0, name='', include_bg=False):
    """
    mean Dice loss function

    :param softmax: input softmax tensor with shape (N,C,D,H,W)
    :param label: input label tensor with shape (N,C,D,H,W)
    :param smooth: smoothing constant for Laplace smoothing
    :param name: network name
    :param include_bg: whether to include background in loss computation or not
    """
    pred_size = mx.sym.square(softmax).sum(axis=[2, 3, 4])
    label_size = label.sum(axis=[2, 3, 4])
    intersection_size = (softmax * label).sum(axis=[2, 3, 4])
    error = (2.0 * intersection_size+smooth) / (pred_size + label_size + smooth)

    if not include_bg:
        error = mx.sym.slice_axis(error, axis=1, begin=1, end=None)
    error = error.mean()
    
    return mx.symbol.MakeLoss(1.0 - error, name=name+'smooth_dice')



def conv(data, kernel, pad, stride, num_filter, net_name, name, num_group):
    """
    convolution wrapper

    :param data: input tensor with shape (N,C,D,H,W)
    :param kernel: shape of the convolving kernel (X,Y,Z)
    :param pad: padding (X,Y,Z)
    :param stride: convolution stride (X,Y,Z)
    :param num_filter: number of filters
    :param net_name: name of the network
    :param name: name of the filter
    :param num_group: number of convolutional groups
    """

    return mx.sym.Convolution(data=data, kernel=kernel, pad=pad, stride=stride, num_filter=num_filter, num_group=num_group,
                               name=net_name + name + 'sep_conv')


def activation(data, type, name):
    """
    activation function wrapper

    :param data: input tensor with shape (N,C,D,H,W)
    :param type: one of ['relu','lrelu','prelu','elu','softplus']
    :param name: activation layer name
    """
    assert(type in ['relu','lrelu','prelu','elu','softplus'])

    act = mx.sym.Activation(data=data, act_type='relu', name=name)

    if type == 'lrelu':
        act = mx.sym.LeakyReLU(data=data, act_type='leaky', slope=config.slope, name=name)
    elif type == 'prelu':
        act = mx.sym.LeakyReLU(data=data, act_type='prelu', slope=config.slope,name=name)
    elif type == 'elu':
        act = mx.sym.LeakyReLU(data=data, act_type='elu', slope=config.slope, name=name)
    elif type=='softplus':
        act = mx.sym.Activation(data=data, act_type='softrelu', name=name)

    return act


def res_net_pre_activation(data, num_filter, net_name, name, normalize=True, num_group=1):
    """
    pre-activation residual block

    :param data: input tensor with shape (N,C,D,H,W)
    :param num_filter: number of filters
    :param net_name: name of the network
    :param name: name of the residiual block
    :param normalize: normalize feature maps or not
    :param num_group: number of groups in convolutions
    """
    norm_data = data
    if normalize:
        norm_data = mx.sym.InstanceNorm(data, name = net_name+name+'norm1')
    relu1 = activation(data=norm_data, type=config.activation, name=name + 'relu1')
    conv1 = conv(data=relu1, kernel=(3, 3, 3), pad=(1, 1, 1), stride=(1, 1, 1), num_filter=num_filter, num_group=num_group, net_name=net_name, name=name + 'conv1')

    if normalize:
        conv1 = mx.sym.InstanceNorm(conv1, name = net_name+name+'norm2')
    relu2 = activation(data=conv1, type=config.activation, name=name + 'relu2')
    conv2 = conv(data=relu2, kernel=(3, 3, 3), pad=(1, 1, 1), stride=(1, 1, 1), num_filter=num_filter, num_group=num_group, net_name = net_name, name=name + 'conv2')

    s = mx.sym.elemwise_add(data,conv2,name=name+'sum')

    return s



def transform_encoders_feature_map(feature_maps):
    """
    transformation of multiple encoders outputs

    :param feature_maps: concatenated encoders' feature maps
    :return joint feature map

    """
    maps = mx.sym.split(feature_maps,axis=1, num_outputs=config.encoder_groups)

    new_stack = mx.sym.stack(*maps,axis=5)
    return mx.sym.max_axis(new_stack,axis=5)



def get_unet_symbol(data, features_number, outputs_number, net_name, stack_conn_0, stack_conn_1, stack_conn_2, return_feature_map = False):
    """
    get base network symbol

    :param data: input tensor with shape (N,C,D,H,W)
    :param features_number: base number of filters (features)
    :param outputs_number: number of output classes
    :param net_name: name of the network
    :param stack_conn_0, stack_conn_1, stack_conn_2: connections with cascaded networks (None or corresponding symbol)
    :param return_feature_map: return softmax or softmax with feature maps from deeper layers of network
    :param training: produce symbol for training or testing phase

    :return symbol

    """

    conv1 = conv(data=data, kernel=(3, 3, 3), pad=(1, 1, 1), stride=(1, 1, 1), num_filter=features_number * 2 * config.encoder_channel_multiplier,
                 num_group=config.encoder_groups, net_name=net_name, name='conv1')

    
    rb1 = res_net_pre_activation(conv1, features_number * 2 * config.encoder_channel_multiplier, net_name, 'rb1', True, config.encoder_groups)
    pool1 = mx.sym.Convolution(data=rb1, kernel=(3, 3, 3), pad=(1, 1, 1), stride=(2,2,2), num_filter=features_number * 4 * config.encoder_channel_multiplier,
                               num_group=config.encoder_groups , name=net_name + 'pool1')
    block_0 = mx.sym.Dropout(pool1, p=0.1, name=net_name + 'do1')


    rb2 = res_net_pre_activation(block_0, features_number * 4 * config.encoder_channel_multiplier, net_name, 'rb2', True, config.encoder_groups)
    pool2 = mx.sym.Convolution(data=rb2, kernel=(3, 3, 3), pad=(1, 1, 1), stride=(2,2,2), num_filter=features_number * 8 * config.encoder_channel_multiplier,
                               num_group=config.encoder_groups, name=net_name + 'pool2')
    block_1 = mx.sym.Dropout(pool2, p=0.1, name=net_name + 'do2')


    rb3 = res_net_pre_activation(block_1, features_number * 8 * config.encoder_channel_multiplier, net_name, 'rb3', True, config.encoder_groups)
    pool3 = mx.sym.Convolution(data=rb3, kernel=(3, 3, 3), pad=(1, 1, 1), stride=(2,2,2), num_filter=features_number * 16 * config.encoder_channel_multiplier,
                               num_group=config.encoder_groups, name=net_name + 'pool3')
    block_2 = mx.sym.Dropout(pool3, p=0.1, name=net_name + 'do3')


    rb4 = res_net_pre_activation(block_2, features_number * 16 * config.encoder_channel_multiplier, net_name, 'rb4', True, config.encoder_groups)
    rb4 = transform_encoders_feature_map(rb4)
    up_conv4_1 = mx.sym.Deconvolution(rb4, kernel=(2, 2, 2), pad=(0, 0, 0), stride=(2, 2, 2),
                                      num_filter=features_number * 8, name=net_name + 'up_conv4_1')


    rb3 = transform_encoders_feature_map(rb3)
    connection0 = mx.symbol.concat(up_conv4_1, rb3, dim=1)
    if stack_conn_0 is not None:
        connection0 = mx.symbol.concat(up_conv4_1, rb3, stack_conn_0, dim=1, name=net_name+'sconc0')
    connection0 = mx.sym.Convolution(data=connection0, kernel=(1, 1, 1), pad=(0, 0, 0), num_filter=features_number * 8)
    rb5 = res_net_pre_activation(connection0, features_number * 8, net_name, 'rb5', True)
    up_conv3_1 = mx.sym.Deconvolution(rb5, kernel=(2, 2, 2), pad=(0, 0, 0), stride=(2, 2, 2),
                                      num_filter=features_number * 4, name=net_name + 'up_conv3_1')


    rb2 = transform_encoders_feature_map(rb2)
    connection1 = mx.symbol.concat(up_conv3_1, rb2, dim=1)
    if stack_conn_1 is not None:
        connection1 = mx.symbol.concat(up_conv3_1, rb2, stack_conn_1, dim=1, name=net_name+'sconc1')
    connection1 = mx.sym.Convolution(data=connection1, kernel=(1, 1, 1), pad=(0, 0, 0), num_filter=features_number * 4)
    rb6 = res_net_pre_activation(connection1, features_number * 4, net_name, 'rb6', True)
    up_conv2_1 = mx.sym.Deconvolution(rb6, kernel=(2, 2, 2), pad=(0, 0, 0), stride=(2, 2, 2),
                                      num_filter=features_number * 2, name=net_name + 'up_conv2_1')


    rb1 = transform_encoders_feature_map(rb1)
    connection2 = mx.symbol.concat(up_conv2_1, rb1, dim=1)
    if stack_conn_2 is not None:
        connection2 = mx.symbol.concat(up_conv2_1, rb1, mx.sym.BlockGrad(stack_conn_2), dim=1, name=net_name+'sconc2')
    
    connection2 = mx.sym.Convolution(data=connection2, kernel=(1, 1, 1), pad=(0, 0, 0), num_filter=features_number * 2)
    rb7 = res_net_pre_activation(connection2, features_number * 2, net_name, 'rb7', True)

    fconv3 = mx.sym.Convolution(data=rb7, kernel=(1, 1, 1), pad=(0, 0, 0), num_filter=outputs_number)
    #if stack_conn_2 is not None:
    #   fconv3 = mx.symbol.elemwise_add(fconv3, stack_conn_2, dim=1, name=net_name+'fconn')

    if return_feature_map:
        return fconv3, rb7

    return fconv3



def softmax(net, label):
    """
    apply softmax and loss to the inputs

    :param net: network output tensor with shape (N,C,D,H,W)
    :param label: ground truth tensor with shape (N,C,D,H,W)

    :return symbol, loss

    """
    softmax = mx.sym.softmax(data=net, name='softmax_lbl', axis=1)
    loss_dice = dice_loss(softmax, label, 1.0, 'seg')

    return softmax, loss_dice



def get_segmentation_model(data, label, filters_number=config.seg_filters_number, n_outputs = config.seg_output_features, name_prefix = '', training=True):
    """
    get segmentation model symbol and corresponding loss

    :param data: input tensor with shape (N,C,D,H,W)
    :param label: ground truth tensor with shape (N,C,D,H,W)
    :param filters_number: base number of filters
    :param n_outputs: number of output classes
    :param name_prefix: model prefix

    :return symbol, loss
    """
    data128 = data

    data64 = mx.sym.Pooling(data=data128, pool_type="avg", kernel=(2,2,2), stride=(2,2,2))
    data32 = mx.sym.Pooling(data=data128, pool_type="avg", kernel=(4,4,4), stride=(4,4,4))
    data16 = mx.sym.Pooling(data=data128, pool_type="avg", kernel=(8,8,8), stride=(8,8,8))
    label128 = label
    label64 = mx.sym.Pooling(data=label128, pool_type="avg", kernel=(2,2,2), stride=(2,2,2))
    label32 = mx.sym.Pooling(data=label128, pool_type="avg", kernel=(4,4,4), stride=(4,4,4))
    label16 = mx.sym.Pooling(data=label128, pool_type="avg", kernel=(8,8,8), stride=(8,8,8))

    #net16, fm16 = get_unet_symbol(data = data16, features_number=filters_number * 8, outputs_number=n_outputs, net_name=name_prefix + 'net16',
    #                              stack_conn_0=None, stack_conn_1=None, stack_conn_2=None, return_feature_map=True, training=True)
    #net16_sm, loss_dice16 = softmax(net16, label16)


    net32, fm32 = get_unet_symbol(data=data32, features_number=filters_number * 4, outputs_number=n_outputs,
                                  net_name=name_prefix + 'net32',
                                  stack_conn_0=None, stack_conn_1=None, stack_conn_2=None, return_feature_map=True)
    net32_sm, loss_dice32 = softmax(net32, label32)



    net64, fm64 = get_unet_symbol(data=data64, features_number=filters_number * 2, outputs_number=n_outputs,
                                  net_name=name_prefix + 'net64',
                                  stack_conn_0=None, stack_conn_1=net32_sm, stack_conn_2=None, return_feature_map=True)
    net64_sm, loss_dice64  = softmax(net64, label64)


    net128 = get_unet_symbol(data=data, features_number=filters_number, outputs_number=n_outputs,
                             net_name=name_prefix + 'net128',
                             stack_conn_0=net32_sm, stack_conn_1=net64_sm, stack_conn_2=None, return_feature_map=False)
    net128_sm, loss_dice128 = softmax(net128, label128)

    loss = 0.4*loss_dice128+0.3*loss_dice64+0.2*loss_dice32#+0.1*loss_dice16

    if not training:
        return mx.sym.BlockGrad(net128_sm), None

    return mx.sym.BlockGrad(net128_sm), loss