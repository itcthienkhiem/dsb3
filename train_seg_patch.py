#!/usr/bin/env python
# encoding: utf-8
import cPickle as pickle
import string
import sys
import time
from itertools import izip
import lasagne as nn
import numpy as np
import theano
from datetime import datetime, timedelta
import utils
import logger
import theano.tensor as T
import buffering
from configuration import config, set_configuration
import pathfinder

# theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:#接受命令行参数,0表示代码本身文件路径，要从1开始取参数
    sys.exit("Usage: train.py <configuration_name>")

config_name = sys.argv[1]#取出参数
set_configuration('configs_seg_patch', config_name)#动态导入模块
expid = utils.generate_expid(config_name)#返回带时间戳的模块名字
print
print "Experiment ID: %s" % expid
print

# metadata
metadata_dir = utils.get_dir_path('models', pathfinder.TIANCHI_METADATA_PATH)# "METADATA_PATH_1": "/mnt/storage/metadata/dsb3/"
metadata_path = metadata_dir + '/%s.pkl' % expid#放置结果的位置【注意，数据是分批次执行的，因此每个批次都要保存】

# logs
logs_dir = utils.get_dir_path('logs', pathfinder.TIANCHI_METADATA_PATH)#获取文件路径
sys.stdout = logger.Logger(logs_dir + '/%s.log' % expid)#获取Logger类一个对象
sys.stderr = sys.stdout

print 'Build model'
model = config().build_model() #返回一个Model，包含(l_in, l_out, l_target)
all_layers = nn.layers.get_all_layers(model.l_out) #返回所有层实例的一个列表
all_params = nn.layers.get_all_params(model.l_out) #返回所有参数的一个列表
num_params = nn.layers.count_params(model.l_out) #返回参数的数量
print '  number of parameters: %d' % num_params
print string.ljust('  layer output shapes:', 36),
print string.ljust('#params:', 10),
print 'output shape:'
for layer in all_layers:
    name = string.ljust(layer.__class__.__name__, 32) #若长度不够32，则用空格填充至32，layer.__class__.__name__ ，获取已知对象的类名。如返回  'contr_1_1'
    num_param = sum([np.prod(p.get_value().shape) for p in layer.get_params()]) #.shape返回矩阵的形状。np.prod：返回数组元素的乘积。get_params():获得每一层的参数;get_value()查看参数值
    num_param = string.ljust(num_param.__str__(), 10) #.__str__()用来返回对象的字符串表达式
    print '    %s %s %s' % (name, num_param, layer.output_shape)

train_loss = config().build_objective(model, deterministic=False) #训练损失
valid_loss = config().build_objective(model, deterministic=True) #验证损失

learning_rate_schedule = config().learning_rate_schedule
learning_rate = theano.shared(np.float32(learning_rate_schedule[0])) #学习率的设置
updates = config().build_updates(train_loss, model, learning_rate) #返回每个参数的更新表达式

x_shared = nn.utils.shared_empty(dim=len(model.l_in.shape)) #创建一个指定维度的Theano共享变量
y_shared = nn.utils.shared_empty(dim=len(model.l_target.shape)) #如上

idx = T.lscalar('idx') #变量声明
givens_train = {}
givens_train[model.l_in.input_var] = x_shared[idx * config().batch_size:(idx + 1) * config().batch_size] #获取训练输入数据。batch_size = 4
givens_train[model.l_target.input_var] = y_shared[idx * config().batch_size:(idx + 1) * config().batch_size] #获取训练目标数据

givens_valid = {}
givens_valid[model.l_in.input_var] = x_shared#Q：为什么验证集是这个size？【A：因为验证时不分批】
givens_valid[model.l_target.input_var] = y_shared

# theano functions
iter_get_predictions = theano.function([idx], nn.layers.get_output(model.l_out), givens=givens_train,on_unused_input='ignore')

iter_train = theano.function([idx], train_loss, givens=givens_train, updates=updates)

iter_get_targets = theano.function([idx], nn.layers.get_output(model.l_target), givens=givens_train, #get_output()计算输出
                                   on_unused_input='ignore')
iter_get_inputs = theano.function([idx], nn.layers.get_output(model.l_in), givens=givens_train,
                                  on_unused_input='ignore')
iter_validate = theano.function([], valid_loss, givens=givens_valid)

if config().restart_from_save:#分批次训练，中断后从上一次的位置开始
    print 'Load model parameters for resuming'
    resume_metadata = utils.load_pkl(config().restart_from_save)
    nn.layers.set_all_param_values(model.l_out, resume_metadata['param_values'])#设置所有层的参数值
    start_chunk_idx = resume_metadata['chunks_since_start'] + 1#从上一次终止的下一个位置开始
    chunk_idxs = range(start_chunk_idx, config().max_nchunks)#剩下的为训练模块的范围

    lr = np.float32(utils.current_learning_rate(learning_rate_schedule, start_chunk_idx))
    print '  setting learning rate to %.7f' % lr
    learning_rate.set_value(lr)
    losses_eval_train = resume_metadata['losses_eval_train']
    losses_eval_valid = resume_metadata['losses_eval_valid']
else:#从0开始
    chunk_idxs = range(config().max_nchunks)
    losses_eval_train = []
    losses_eval_valid = []
    start_chunk_idx = 0

train_data_iterator = config().train_data_iterator#训练迭代器
valid_data_iterator = config().valid_data_iterator#验证迭代器

print
print 'Data'
print 'n train: %d' % train_data_iterator.nsamples
print 'n validation: %d' % valid_data_iterator.nsamples
print 'n chunks per epoch', config().nchunks_per_epoch

print
print 'Train model'
chunk_idx = 0
start_time = time.time()
prev_time = start_time
tmp_losses_train = []

# use buffering.buffered_gen_threaded()#开始获取数据训练，训练数据的来源：train_data_iterator.generate()
for chunk_idx, (x_chunk_train, y_chunk_train, id_train) in izip(chunk_idxs, buffering.buffered_gen_threaded(
        train_data_iterator.generate())):
    if chunk_idx in learning_rate_schedule:
        lr = np.float32(learning_rate_schedule[chunk_idx])
        print '  setting learning rate to %.7f' % lr
        print
        learning_rate.set_value(lr)

    # load chunk to GPU
    x_shared.set_value(x_chunk_train)
    y_shared.set_value(y_chunk_train)

    # make nbatches_chunk iterations
    chunk_train_losses = []
    for b in xrange(config().nbatches_chunk):
        loss = iter_train(b)
        chunk_train_losses.append(loss)
        tmp_losses_train.append(loss)
    print chunk_idx, np.mean(chunk_train_losses)

    if ((chunk_idx + 1) % config().validate_every) == 0:
        print
        print 'Chunk %d/%d' % (chunk_idx + 1, config().max_nchunks)
        # calculate mean train loss since the last validation phase
        mean_train_loss = np.mean(tmp_losses_train)
        print 'Mean train loss: %7f' % mean_train_loss
        losses_eval_train.append(mean_train_loss)
        tmp_losses_train = []

        # load validation data to GPU
        tmp_losses_valid = []
        for i, (x_chunk_valid, y_chunk_valid, ids_batch) in enumerate(
                buffering.buffered_gen_threaded(valid_data_iterator.generate(),
                                                buffer_size=2)):
            x_shared.set_value(x_chunk_valid)
            y_shared.set_value(y_chunk_valid)
            l_valid = iter_validate()
            print i, l_valid
            tmp_losses_valid.append(l_valid)

        # calculate validation loss across validation set
        valid_loss = np.mean(tmp_losses_valid)
        # TODO: taking mean is not correct if chunks have different sizes!!!
        print 'Validation loss: ', valid_loss
        losses_eval_valid.append(valid_loss)

        now = time.time()
        time_since_start = now - start_time
        time_since_prev = now - prev_time
        prev_time = now
        est_time_left = time_since_start * (config().max_nchunks - chunk_idx + 1.) / (chunk_idx + 1. - start_chunk_idx)
        eta = datetime.now() + timedelta(seconds=est_time_left)
        eta_str = eta.strftime("%c")
        print "  %s since start (%.2f s)" % (utils.hms(time_since_start), time_since_prev)
        print "  estimated %s to go (ETA: %s)" % (utils.hms(est_time_left), eta_str)
        print

    if ((chunk_idx + 1) % config().save_every) == 0:
        print
        print 'Chunk %d/%d' % (chunk_idx + 1, config().max_nchunks)
        print 'Saving metadata, parameters'

        with open(metadata_path, 'w') as f:
            pickle.dump({
                'configuration_file': config_name,
                'git_revision_hash': utils.get_git_revision_hash(),
                'experiment_id': expid,
                'chunks_since_start': chunk_idx,
                'losses_eval_train': losses_eval_train,
                'losses_eval_valid': losses_eval_valid,
                'param_values': nn.layers.get_all_param_values(model.l_out)
            }, f, pickle.HIGHEST_PROTOCOL)
            print '  saved to %s' % metadata_path
            print
