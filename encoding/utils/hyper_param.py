# encoding:utf-8

__all__ = ['get_model_args', 'split_train_args', 'get_train_args']

# 第一位 与 第二位
module_dict = {'1':'CA6', '2':'CA6|act_fn=tanh', '3':'CA2a|act_fn=sigmoid', '4':'CA2a|act_fn=softmax',
               '5':'CA2b|act_fn=sigmoid', '6':'CA2b|act_fn=softmax',
               'v':'CA2b|act_fn=softmax|ppl=1',
               'w':'CA2b|act_fn=softmax|ppl=2',
               'x':'CA2b|act_fn=softmax|ppl=3',
               'y':'CA2b|act_fn=softmax|ppl=4',




               '7':'PA0|act_fn=sigmoid', '8':'PA0|act_fn=tanh', # PA0的其他参数已经写到模块里了
               '9':'CA3', 'a':'PA2|act_fn=sigmoid', 'b':'PA2|act_fn=tanh',
               'c': 'PA3a|act_fn=sigmoid',

               'd': 'GF0',   # 只有irb
               'D': 'GF|att=idt|gcf=merge-gc',    # GF: 只有gc
               'E': 'GF|att=pdl|gcf=merge-add',   # GF: 只有pdl (没有irb, fusion直接用相加）

               'e': 'GF|att=pdl|gcf=merge-gc',    # GF: 没有irb, 有pdl 和 gc
               'f': 'GF1|att=pdl|gcf=merge-gc',   # GF: irb + pdl + gc
               'm': 'GF1|att=pdl|gcf=merge-add',  # GF: irb + pdl + add
               'n': 'GF1|att=pdl|gcf=merge-gc|gca=ppl-1',
               'o': 'GF1|att=pdl|gcf=merge-gc|gca=ppl-2',
               'p': 'GF1|att=pdl|gcf=merge-gc|gca=ppl-3',
               'q': 'GF1|att=pdl|gcf=merge-gc|gca=ppl-4',


               'g': 'PSK|act_fn=sig|pp=a', 'h': 'PSK|act_fn=sig|pp=b', 'i': 'PSK|act_fn=sig|pp=c',      # 注意 b: U=x+y, c:  U=x  a: U=y
               'j': 'PSK|act_fn=soft|pp=a', 'k': 'PSK|act_fn=soft|pp=b', 'l': 'PSK|act_fn=soft|pp=c',
               'r': 'PSK|act_fn=sig|pp=b|ppl=1',
               's': 'PSK|act_fn=sig|pp=b|ppl=2',
               't': 'PSK|act_fn=sig|pp=b|ppl=3',      # pp=b means U=x+y,  ppl=3 pyramid pooling of 3 levels

               'A': 'PSK|act_fn=sig|pp=b|ppl=3|r=16|dd=8',      # pp=b means U=x+y,  ppl=3 pyramid pooling of 3 levels
               'B': 'PSK|act_fn=sig|pp=b|ppl=3|r=16|dd=4',
               'C': 'PSK|act_fn=sig|pp=b|ppl=3|r=16|dd=16',

               'D': 'PSK|act_fn=sig|pp=b|ppl=3|r=8|dd=8',      # pp=b means U=x+y,  ppl=3 pyramid pooling of 3 levels
               'E': 'PSK|act_fn=sig|pp=b|ppl=3|r=16|dd=8',
               'F': 'PSK|act_fn=sig|pp=b|ppl=3|r=31|dd=8',

               'u': 'PSK|act_fn=sig|pp=b|ppl=4',

               '0': None, None:None


               }
# 第三位【2】
center_dict = {'0':None, None:None,  '1': 'apn'}

# 第四位【3】 321af: y1与y2, y3融合，  'aaf': stage-wise融合
dan_dict = {'0':None, None:None, '1':'21af', '2': '321af', '3': 'aaf'}  # 321af: y1与y2, y3融合，

# 第五位 【4】
aux_dict = {'0':None, None:None, '1':'1', '2':'2', '3':'3', '4':'4', '5':'32', '6':'21', '7':'43', '8':'321', '9':'432'
            }

# 第六位 【5】
out_dict = {'0':None, None:None, 'g':'g', 'h':'g2'}

def get_args(s, i):
    try:
        return s[i]
    except:
        return None


def get_model_args(exp_args):    # 有关模型结构的超参
    if '_' in exp_args:
        exp_args = exp_args.split('_')[0]

    # exp_args中有关模型结构超参 默认有两位， 第一位为 mmf, 第二位为 mrf，
    # 之后的依此为center piece, 模型最后的dan, aux loss, out:模型最后输出segmentation是否用LearnedUpUnit(default)
    mmf_args = module_dict[get_args(exp_args,0)]
    mmfs = 'mmf=' + mmf_args if mmf_args is not None else None

    mrf_args = module_dict[get_args(exp_args,1)]
    mrfs = 'mrf=' + mrf_args if mrf_args is not None else None

    ctr = center_dict[get_args(exp_args, 2)]
    dan = dan_dict[get_args(exp_args, 3)]
    aux = aux_dict[get_args(exp_args, 4)]
    out = out_dict[get_args(exp_args, 5)]

    model_args = dict(mmfs=mmfs, mrfs=mrfs, ctr=ctr, dan=dan, aux=aux, out=out)
    return model_args


def split_train_args(exp_args):  # 有关训练过程的超参
    if '_' in exp_args:
        train_args = exp_args.split('_')[1]
    else:
        train_args = None
    return train_args


scheduler_dict = {None: None, '0': None, 'p': 'poly', 'c': 'cos', 's':'step'}
lr_dict = {None: None, '0':None, '1': 0.001, '2': 0.002, '3':0.003, '4':0.004, '5':0.005 }
aux_weight_dict = { None: None, '0': None, '2': 0.2, '3': 0.3, '5':0.5, '8': 0.8, 'a':1.0, 'b':1.2, 'c':1.5, 'd': 2}
class_weight_dict = {None: None, '0': None,   '1': 'wt1', '2':'wt2', '3':'wt3',
                                              'a':'median_frequency', 'b': 'logarithmic'
                    }    # a is for 0.5


def get_train_args(train_args):
    scheduler = scheduler_dict[get_args(train_args, 0)]
    lr = lr_dict[get_args(train_args, 1)]
    aux_weight = aux_weight_dict[get_args(train_args, 2)]
    class_weight = class_weight_dict[get_args(train_args,3)]

    train_args = dict(lr_scheduler = scheduler, lr =lr, aux_weight=aux_weight, class_weight=class_weight)
    return train_args


