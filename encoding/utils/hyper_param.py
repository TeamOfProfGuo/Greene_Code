# encoding:utf-8

__all__ = ['get_model_args', 'split_train_args', 'get_train_args']

# 第一位 与 第二位
module_dict = {'1':'CA6', '2':'CA6|act_fn=tanh', '3':'CA2a|act_fn=sigmoid', '4':'CA2a|act_fn=softmax',
               '5':'CA2b|act_fn=sigmoid', '6':'CA2b|act_fn=softmax',
               '7':'PA0|act_fn=sigmoid', '8':'PA0|act_fn=tanh', # PA0的其他参数已经写到模块里了
               '9':'CA3', 'a':'PA2|act_fn=sigmoid', 'b':'PA2|act_fn=tanh',
               'c': 'PA3a|act_fn=sigmoid', 'd': 'PA3a|act_fn=tanh',
               'e': 'GF|att=pdl|gcf=merge-gc', 'f': 'GF1|att=pdl|gcf=merge-gc',   # 备注 gca=None
               '0':None, None:None
               }
# 第三位【2】
center_dict = {'0':None, None:None,  '1': 'apn'}

# 第四位【3】 321af: y1与y2, y3融合，  'aaf': stage-wise融合
dan_dict = {'0':None, None:None, '1':'21af', '2': '321af', '3': 'aaf'}  # 321af: y1与y2, y3融合，

# 第五位 【4】
aux_dict = {'0':None, None:None, '1':'1', '2':'2', '3':'3', '4':'4', '5':'32', '6':'21', '7':'43', '8':'321', '9':'432'
            }

def get_args(s, i):
    try:
        return s[i]
    except:
        return None


def get_model_args(exp_args):
    if '_' in exp_args:
        exp_args = exp_args.split('_')[0]
    mmf_args = module_dict[get_args(exp_args,0)]
    mmfs = 'mmf=' + mmf_args if mmf_args is not None else None

    mrf_args = module_dict[get_args(exp_args,1)]
    mrfs = 'mrf=' + mrf_args if mrf_args is not None else None

    ctr = center_dict[get_args(exp_args, 2)]
    dan = dan_dict[get_args(exp_args, 3)]
    aux = aux_dict[get_args(exp_args, 4)]

    model_args = dict(mmfs=mmfs, mrfs=mrfs, ctr=ctr, dan=dan, aux=aux)
    return model_args


def split_train_args(exp_args):
    if '_' in exp_args:
        train_args = exp_args.split('_')[1]
    else:
        train_args = None
    return train_args


scheduler_dict = {None: None, 'p': 'poly', 'c': 'cos', 's':'step'}
lr_dict = {None: None, '0':None, '1': 0.001, '2': 0.002, '3':0.003, '4':0.004, '5':0.005 }
aux_weight_dict = {None: None, '2':0.2, '3': 0.3, '5':0.5, '8': 0.8, 'a':1.0, 'b':1.2, 'c':1.5}


def get_train_args(train_args):
    scheduler = scheduler_dict[get_args(train_args, 0)]
    lr = lr_dict[get_args(train_args, 1)]
    aux_weight = aux_weight_dict[get_args(train_args, 2)]

    train_args = dict(lr_scheduler = scheduler, lr =lr, aux_weight=aux_weight)
    return train_args


