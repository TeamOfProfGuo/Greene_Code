# encoding:utf-8

__all__ = ['parse_setting', 'get_param_ids']

def parse_setting(s, sep_out='|', sep_in='='):
    def parse_kv(e):
        k, v = e.split(sep_in)
        if v.isdigit():
            v = int(v)
        elif v in ['True', 'False']:
            v = bool(v)
        return k, v

    if s=='' or s is None:
        return {}
    s_list = s.split(sep_out)
    s_dict = dict([ tuple(parse_kv(e)) for e in s_list ])
    return s_dict


def get_param_ids(module_list):
    param_ids = []
    for mo in module_list:
        ids = list(map(id, mo.parameters()))
        param_ids = param_ids + ids
    return param_ids