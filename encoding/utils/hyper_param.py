# encoding:utf-8

__all__ = ['get_model_args']

module_dict = {'1':'CA6', '2':'CA6|act_fn=tanh', '3':'CA2a|act_fn=sigmoid', '4':'CA2a|act_fn=softmax',
               '5':'CA2b|act_fn=sigmoid', '6':'CA2b|act_fn=softmax',
               '7':'PA0|act_fn=sigmoid', '8':'PA0|act_fn=tanh', # PA0的其他参数已经写到模块里了
               '0':None, None:None
               }

center_dict = {'0':None, None:None,  '1': 'apn'}

# 321af: y1与y2, y3融合，  'aaf': stage-wise融合
dan_dict = {'0':None, None:None, '1':'21af', '2': '321af', '3': 'aaf'}  # 321af: y1与y2, y3融合，


def get_args(s, i):
    try:
        return s[i]
    except:
        return None


def get_model_args(exp_args):
    mmf_args = module_dict[get_args(exp_args,0)]
    mmfs = 'mmf=' + mmf_args if mmf_args is not None else None

    mrf_args = module_dict[get_args(exp_args,1)]
    mrfs = 'mrf=' + mrf_args if mrf_args is not None else None

    ctr = center_dict[get_args(exp_args, 2)]
    dan = dan_dict[get_args(exp_args, 3)]

    model_args = dict(mmfs=mmfs, mrfs=mrfs, ctr=ctr, dan=dan)
    return model_args

