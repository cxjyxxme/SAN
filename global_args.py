def _init():
    global _global_dict
    _global_dict = {}

def set_args(value):
    _global_dict['args'] = value

def get_args():
    return _global_dict['args']