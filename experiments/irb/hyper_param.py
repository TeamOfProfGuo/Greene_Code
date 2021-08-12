


module_dict = {'1':'CA6', '2':'CA6|act_fn=tanh', '3':'CA2a|act_fn=sigmoid', '4':'CA2a|act_fn=softmax',
               '5':'CA2b|act_fn=sigmoid', '6':'CA2b|act_fn=softmax',
               '7':'PA0|act_fn=sigmoid|conv=conv|fuse=cat', '8':'PA0|act_fn=tanh|conv=conv|fuse=add',
               '0':None, None:None
               }

center_dict = {'0':None, None:None,  '1': 'apn'}

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
    ctr = center_dict[get_args(exp_args,2)]
    model_args = dict(mmfs=mmfs, mrfs=mrfs, ctr=ctr)
    return model_args


