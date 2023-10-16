import io
import numpy as np
import sys
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model.earlystopping import EarlyStopping

class TrainerBase:
    def __init__(self, epochs, evaluate_steps=1,valid_on_train_set = True,valid_on_test_set = True,verbose=True):
        self.epochs = epochs
        self.verbose = verbose
        self.evaluate_steps = evaluate_steps
        self.valid_on_train_set = valid_on_train_set
        self.valid_on_test_set = valid_on_test_set

    def _train_epoch(self, model, train_loader, epch):
        iter_data = (
            tqdm(
                train_loader,
                total=len(train_loader),
                ncols=100,
                desc=f"Train {epch}:>5"
            )
            if self.verbose
            else train_loader
        )
        headers = None
        data = []
        for batch_data in iter_data:
            # batch是train_loader的一个batch
            result = model.opt_one_batch(batch_data)
            assert isinstance(result, dict),   "opt_one_batch 返回数据必须是dict类型"
            assert 'loss' in result, "opt_one_batch 返回的字典必须包含loss关键字"
            if headers is None :
                headers = result.keys()
            data.append(list(result.values()))
        data = np.mean(np.array(data), axis=0)  # 按照数据的最后一个维度求均值，data中必须不能是tensor数据，
        return dict(zip(headers,data)) # 返回一个字典

    def _eval_data(self, model, dataloader, valid_fun):
        # 进行一次测试，数据可能来自于训练集，也可能来自于测试集
        return model.eval_data(dataloader, valid_fun)

    def create_EarlyStopping(self, model):
        """
        根据模型中的参数，创建早停函数
        """
        patience = 200
        delta = 0
        trace_func = print
        checkpoint = io.BytesIO()
        if hasattr(model, 'checkpoint_path'):
            checkpoint = open(model.checkpoint_path,'+bx')
        if hasattr(model, 'es_patience'):
            patience = model.es_patience
        if hasattr(model, 'es_delta'):
            delta = model.es_delta
        return EarlyStopping(patience, self.verbose, delta, trace_func, checkpoint)

    def train(self, model, train_loader, valid_loader,test_loader, valid_func, loger=sys.stdout):
        assert hasattr(model,
                       'opt_one_batch'), f"模型 {model} 必须实现 opt_one_batch 函数才能使用 TrainerBase"
        assert hasattr(model,
                       'eval_data'), f"模型 {model} 必须实现 opt_one_batch 函数才能使用 eval_data"
        assert hasattr(model,
                       'save'), f"模型 {model} 必须实现 opt_one_batch 函数才能使用 save"
        assert hasattr(model,
                       'load'), f"模型 {model} 必须实现 opt_one_batch 函数才能使用 load"
        # 将第epch个的score打印出来
        def printf(key, value, epch):
            if isinstance(loger, SummaryWriter):
                loger.add_scalar(key, value, global_step=epch)
            elif loger == sys.stdout:
                line = f"{key}={value}\t\t  epcho={epch}"
                loger.writelines(line)
        metric_name = str(valid_func)  # 获得评价指标名字
        early_stoping = self.create_EarlyStopping(model)  # 根据model中的参数初始化早停
        # 开始迭代
        for epch in range(1, self.epochs + 1):
            results = self._train_epoch(model, train_loader, epch)  # 一次轮训练，并返回损失值
            if self.verbose:
                for key, value in results.items(): #打印一次训练返回的所有结果，必须包含loss值
                    printf(key, value, epch)
            # 不进行验证，只做训练。
            if self.evaluate_steps <=0 :
                continue
            # 如果满足验证条件，则开始验证
            if (epch - 1) % self.evaluate_steps == 0:  # 当前训练需要 计算验证指标
                if self.valid_on_train_set:  # 是否需要在训练集上计算指标?
                    # 模型在训练集上的结果
                    train_score = self._eval_data(model, train_loader, valid_func)
                    printf(f"{metric_name}@train set", train_score, epch)
                    print("train_score:{},epoch:{}".format(train_score, epch))
                if self.valid_on_test_set and test_loader is not None:  # 是否需要在测试集上计算指标?
                    # 模型在测试集上的结果
                    test_score = self._eval_data(model, test_loader, valid_func)
                    printf(f"{metric_name}@test set", test_score, epch)
                    print("test_score:{},epoch:{}".format(test_score, epch))
                # 模型在验证集上的结果
                if valid_loader is not None:  # 如果验证集不为空，模型则在验证集上指标结果作为筛选模型的标准
                    val_score = self._eval_data(model, valid_loader, valid_func)
                    printf(f"{metric_name}@valid set", val_score, epch)
                    print("val_score:{},epoch:{}".format(val_score, epch))
                elif self.valid_on_train_set: # 如果开启了训练集的验证标志，优先使用该指标为筛选模型的标准
                    val_score=train_score
                else: # 既没有提供验证集，又不在训练集上计算指标时，就直接使用训练集上的损失值作为筛选模型的标准。
                    val_score = -results['loss'] # 如果没有给定验证集，则使用损失函数的负值估计最优模型。
            if valid_loader != None and  hasattr(valid_func, 'bigger') and valid_func.bigger == False:
                val_score = -val_score  # 分数越小越好，则取反进行判断
            early_stoping(val_score, model, epch)  # 保存当前最优模型，因为是在训练类里面实现的load和save。
            if early_stoping.early_stop:
                break
        if self.evaluate_steps > 0:  #模型有验证过程
            early_stoping.get_best(model)  # 加载保存的最优模型