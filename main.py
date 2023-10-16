import importlib
import shutil
import time
import hashlib
import logging
import io
import time
import zipfile
import yaml
import argparse
import random
import os
import torch
import pandas as pd
import numpy as np

def set_seed(seed=2333):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def myhash(data):
    data = data + str(time.time())
    return hashlib.md5(data.encode(encoding='UTF-8')).hexdigest()

def check_result_existing(log_file):
    # 检查结果是否已经存在，如果存在就返回True，否则返回False
    if os.path.isfile(log_file):
        return True
    return False

def smart_convert(value):
    assert isinstance(value, str)
    if value.count('.') > 0:
        try:
            return float(value)
        except:
            pass
    try:  # 检查整数
        return int(value)
    except:
        return value

def need_import(value):
    if isinstance(value, str) and len(value) > 3 and value[0] == value[-1] == '_' and not value == "__init__":
        return True
    else:
        return False

def create_obj_from_json(js):
    if isinstance(js, dict): #传入是字典
        rtn_dict = {}
        for key, values in js.items():
            if need_import(key):
                assert values is None or isinstance(values,
                                                    dict), f"拟导入的对象{key}的值必须为dict或None，用于初始化该对象"
                assert len(js) == 1, f"{js} 中包含了需要导入的{key}对象，不能再包含其他键值对"
                key = key[1:-1]  # 去掉 key的前后 `_`
                cls = my_import(key)
                if "__init__" in values:  # 如果该类被设置了初始化函数，则需要读取初始化函数的值
                    assert isinstance(values, dict), f"__init__ 关键字，放入字典对象，作为父类{key}的初始化函数"
                    init_params = create_obj_from_json(values['__init__'])  # 获取初始化的参数，并返回。
                    if isinstance(init_params, dict):
                        obj = cls(**init_params)  # 清空"__init__"相关的值，方便后续处理。
                    else:
                        obj = cls(init_params)
                    values.pop("__init__")
                else:
                    obj = cls()
                # 此处已经不包含 "__init__"的key，value对
                for k, v in values.items():
                    setattr(obj, k, create_obj_from_json(v))
                return obj
            rtn_dict[key] = create_obj_from_json(values)
        return rtn_dict
    elif isinstance(js, (set, list)):
        return [create_obj_from_json(x) for x in js]
    elif isinstance(js,str):
        if need_import(js):
            cls_name = js[1:-1]
            return my_import(cls_name)()
        else:
            return js
    else: # 其他对象直接返回
        return js

def my_import(name):  # name指定的算法类名，eg：algorithm.pmf.MF
    components = name.split('.')
    model_name = '.'.join(components[:-1])
    class_name = components[-1]
    mod = importlib.import_module(model_name)
    cls = getattr(mod, class_name)  # 在py文件中获取在pmf.py文件下获取模型类 <classes 'algorithm.pmf.MF'>
    return cls  # 返回的是类

def myloads(jstr):
    if hasattr(yaml, 'full_load'):
        js = yaml.full_load(io.StringIO(jstr))
    else:
        js = yaml.load(io.StringIO(jstr))
    if isinstance(js, str):
        return {js: {}}
    else:
        return js

start_time = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
parser = argparse.ArgumentParser(description='算法测试程序')
parser.add_argument('-f', dest='argFile', type=str, required=True,
                    default=None,
                    help='通过YAML文件指定试验参数文件。')
parser.add_argument('-w', dest='over_write', type=bool,
                    help='强制覆盖已经存在的结果')
parser.add_argument('-o', dest='out_path', type=str, required=False,
                    help='指定结果存放路径')
parser.add_argument('-m', dest='metrics', type=str, required=False,
                    help='验证指标列表，逗号分割')
parser.add_argument('-a', dest='alg', type=str, required=False,
                    help='需要验证的算法')
parser.add_argument('-d', dest='data_path', type=str, required=False,
                    help='数据目录或者数据集压缩包')
parser.add_argument('-r', dest='params', type=myloads, required=False,
                    # default="{}",
                    help='''算法参数，是一个json，例如"{d: 20,lr: 0.1,n_itr: 1000}" ''')

def update_parameters(param: dict, to_update: dict) -> dict:
    for k, v in param.items():
        if k in to_update:
            if to_update[k] is not None:
                if isinstance(param[k], (dict,)):
                    param[k].update(to_update[k])
                else:
                    param[k] = to_update[k]
            to_update.pop(k)
    param.update(to_update)
    return param

def str_obj_js(mystr):
    # 将字符串转化为 描述对象的json
    if ':' in mystr:
        return myloads(mystr)
    else:
        return {mystr: {}}

def enclose_class_name(value):
    if isinstance(value,dict): # 如果vale是一个字典,{apt.scenario.ScenarioAbstract:{...}}
        assert len(value)==1, "只能有一个类"
        for k,v in value.items():
            if k[0]==k[-1]=="_":
                return {k:v}
            else:
                return {f"_{k}_":v}
    elif isinstance(value,str): # 如果vale是一个字符串,apt.scenario.ScenarioAbstract
        #如果发现value一定是类名，则自动给其添加前后`_`
        if value[0]==value[-1]=="_":
            return value
        else:
            return f"_{value}_"
    else:
        return value

def parse_objects(filedict):
    algorithm = create_obj_from_json(enclose_class_name({filedict['algorithm']:filedict['algorithm_parameters']}))
    metrics = []
    for m in filedict['metrics']:
        metrics.append(create_obj_from_json(enclose_class_name(m)))
    return algorithm, metrics

def _split(data):
    n_train = 0.8
    flag_train = 0.8
    n_test = 0.1
    n_valid = 0.1
    random_seed = 1
    n = len(data)  # 数据总量
    index = np.arange(n)
    np.random.seed(random_seed)
    np.random.shuffle(index)
    train = data[index[: int(n * n_train)]]
    valid = data[index[int(n * flag_train): int(n * (flag_train + n_valid))]]
    test = data[index[int(n * (flag_train + n_valid)):]]
    train_x, train_y = train[:, :-1], train[:, -1]
    valid_x, valid_y = valid[:, :-1], valid[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def split(data):
    data1, data2 = data
    train_x1, train_y, valid_x1, valid_y, test_x1, test_y = _split(data1)
    train_x2, _, valid_x2, _, test_x2, _ = _split(data2)
    return (train_x1, train_x2, train_y), (valid_x1, valid_x2, valid_y), (test_x1, test_x2, test_y)

def parse_data(data_dir):
    data = None
    if os.path.isfile(data_dir): # 如果是zip文件，则按照zip方式读取数据
        z = zipfile.ZipFile(data_dir, "r")
        for filename in z.namelist():
            if filename == 'data1.csv':
                data1 = pd.read_csv(z.open(filename), header=None)
            if filename == 'data2.csv':
                data2 = pd.read_csv(z.open(filename), header=None)
        assert data1 is not None, "压缩包中找不到data1.csv文件"
        assert data2 is not None, "压缩包中找不到data2.csv文件"
    else:
        data1 = pd.read_csv(data_dir + "/data1.csv", header=None)
        data2 = pd.read_csv(data_dir + "/data2.csv", header=None)
    data1 = data1.astype(np.float32).values
    data2 = data2.astype(np.float32).values
    data = data1,data2
    return data

if __name__ == '__main__':
    args = parser.parse_args()
    filedict = {}
    if args.argFile is not None:
        filelist = args.argFile.split(',')
        for fname in filelist:
            with open(fname.strip(), 'rb') as infile:
                fd = yaml.safe_load(infile)
                update_parameters(filedict, fd)

    # metrics 默认必须是数组,如果是字符串，则自动转化为数组
    if 'metrics' in filedict and isinstance(filedict['metrics'],str):
        filedict['metrics'] = [filedict['metrics']]

    m = None if args.metrics is None else args.metrics.split(',')
    arg_dict = {
        "algorithm": args.alg,
        "algorithm_parameters": args.params,
        "data_path": args.data_path,
        "out_path": args.out_path,
        "over_write": args.over_write,
        "metrics": m}

    update_parameters(filedict, arg_dict)  # 将命令行传入的参数更新到YAML配置文件指定的参数中
    # 当配置文件和命令行有相同的参数时，以命令行的参数为准，命令行参数优先级最高

    algorithm, metrics = parse_objects(filedict)

    data_path = filedict['data_path']
    out_path = filedict['out_path']
    over_write = filedict['over_write']

    assert data_path, "必须给定数据目录"
    if not out_path: out_path='log' # 默认输出目录为 log 目录
    set_seed(7)
    ds = parse_data(data_path)
    data_name = os.path.splitext(data_path)[0]  # 去除路径包含的文件后缀名
    data_name = os.path.split(data_name)[-1]  # 去掉文件的父路径，只保留文件名
    # 记录实验配置到log文件
    cache_dir = os.path.join(out_path,algorithm.class_name() + '-' + data_name, 'cache')
    out_path = os.path.join(out_path,algorithm.class_name() + '-' + data_name, str(myhash(str(algorithm))))

    if os.path.isdir(out_path):
        if over_write:
            shutil.rmtree(out_path)  # 删除所有内容
        else:
            print(f"结果目录已经存在{out_path}")
            exit(-1)
    os.makedirs(out_path)

    log_path = os.path.join(out_path, 'result.log')
    start_time = str(time.strftime("%Y%m%d%H%M%S", time.localtime()))
    algorithm.checkpoint_path = os.path.join(out_path, 'checkpoint')
    if not hasattr(algorithm, "tensorboard_path") or not algorithm.tensorboard_path:
        algorithm.tensorboard_path = os.path.join(out_path, 'log_tensorboard')

    # 设置缓存目录，用于存放模型的中间结果。
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    if not hasattr(algorithm, 'cache_dir') or not algorithm.cache_dir:
        algorithm.cache_dir = cache_dir

    logging.basicConfig(filename=log_path,  # 配置日志保存路径
                        level=logging.INFO,
                        format="%(message)s")
    logging.info('Train Start Time: ' + start_time)  # 程序启动时间
    logging.info("ds" + ":" + data_path)
    logging.info("algorithm-parameter:")
    logging.info(algorithm)  # 保存模型参数

    train_set, valid_set, test_set = split(ds)
    valid_fun = metrics[0]
    algorithm.train(train_set, valid_set, test_set, valid_fun)  # 调用需要训练的模型的train函数，因此必须实现train函数
    out = algorithm.predict(test_set)
    pred = out  # 调用预测函数，获得预测结果
    true_label = test_set[2]
    results = [m(true_label, pred) for m in metrics]  # 调用验证指标类中的evaluate函数，
    headers = [str(m) for m in metrics]
    results = dict(zip(headers, results))
    print("Final Results is :", results, filedict)
