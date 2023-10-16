import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizer_module
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
from torch.utils.data import DataLoader
from model.evaluation import EmbeddedDataset, build_matrix
from sklearn.preprocessing import MinMaxScaler
from model.train import TrainerBase
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression


def make_tuple(data):
    # data=[(x1,x2,x3,x4...),(y1,y2,y3,y4,...)]
    # 返回 [(x1,y1),(x2,y2),(x3,y3),(x4,y4),...]
    return list(zip(*data))

def init_optimizer(optimizer_name, params):
    assert hasattr(optimizer_module, optimizer_name)
    OptimizerClass = getattr(optimizer_module, optimizer_name)
    return OptimizerClass(params)

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to_dense()

class Encoder(nn.Module):
    def __init__(self, input_dim1=28, input_dim2=28, z_dim=64):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim1 * input_dim2, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, z_dim * 2),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        params = self.net(x)
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-5  # Make sigma always positive
        return Independent(Normal(loc=mu, scale=sigma), 1)  # Return a factorized Normal distribution

class Decoder(nn.Module):
    def __init__(self, z_dim=64, input_dim1=28, input_dim2=28, scale=0.39894):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, input_dim1*input_dim2)
        )
    def forward(self, z):
        x = self.net(z)
        return Independent(Normal(loc=x, scale=self.scale), 1)

class MIEstimator(nn.Module):
    def __init__(self, size1, size2):
        super(MIEstimator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
        )
    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2):
        x2 = x2.to(torch.float32)
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1

# Schedulers for beta
class Scheduler:
    def __call__(self, **kwargs):
        raise NotImplemented()

class LinearScheduler(Scheduler):
    def __init__(self, start_value, end_value, n_iterations, start_iteration=0):
        self.start_value = start_value
        self.end_value = end_value
        self.n_iterations = n_iterations
        self.start_iteration = start_iteration
        self.m = (end_value - start_value) / n_iterations

    def __call__(self, iteration):
        if iteration > self.start_iteration + self.n_iterations:
            return self.end_value
        elif iteration <= self.start_iteration:
            return self.start_value
        else:
            return (iteration - self.start_iteration) * self.m + self.start_value

class ExponentialScheduler(LinearScheduler):
    def __init__(self, start_value, end_value, n_iterations, start_iteration=0, base=10):
        self.base = base
        super(ExponentialScheduler, self).__init__(start_value=math.log(start_value, base),
                                                   end_value=math.log(end_value, base),
                                                   n_iterations=n_iterations,
                                                   start_iteration=start_iteration)

    def __call__(self, iteration):
        linear_value = super(ExponentialScheduler, self).__call__(iteration)
        return self.base ** linear_value

class LMIB(nn.Module):
    def __init__(self, input_dim1, dim1, input_dim2,dim2, z_dim, n_class, ALPHA, BETA, GAMMA, beta_start_value, beta_end_value, beta_n_iterations, beta_start_iteration,
                 iterations, device, flag):
        super().__init__()
        self.encoder1 = Encoder(input_dim1, dim1, z_dim)
        self.encoder2 = Encoder(input_dim2, dim2, z_dim)
        self.encoderT1 = Encoder(input_dim1, dim1, z_dim)
        self.encoderT2 = Encoder(input_dim2, dim2, z_dim)
        self.decoder1 = Decoder(z_dim, input_dim1, dim1)
        self.decoder2 = Decoder(z_dim, input_dim1, dim1)
        self.decoderT1 = Decoder(z_dim, input_dim2, dim2)
        self.decoderT2 = Decoder(z_dim, input_dim1, dim1)
        self.decoder_v1 = self.decoder1
        self.decoder_v2 = self.decoder2
        self.encoder_v1 = self.encoder1
        self.encoder_v2 = self.encoder2
        self.encoder_vT1 = self.encoderT1
        self.encoder_vT2 = self.encoderT2
        self.decoder_vT1 = self.decoderT1
        self.decoder_vT2 = self.decoderT2
        self.iterations = iterations
        self.n_class = n_class
        self.z_dim = z_dim
        self.device = device
        self.flag = flag
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.GAMMA = GAMMA
        self.z_mi_estimator = MIEstimator(self.z_dim, self.z_dim)
        self.beta_scheduler = ExponentialScheduler(start_value=beta_start_value, end_value=beta_end_value,
                                                   n_iterations=beta_n_iterations,
                                                   start_iteration=beta_start_iteration)
        self.alpha_scheduler = ExponentialScheduler(start_value=beta_start_value, end_value=beta_end_value,
                                                   n_iterations=beta_n_iterations,
                                                   start_iteration=beta_start_iteration)

    def cal_mi(self, v, z):
        v = v.view(v.size(0), -1)
        if torch.cuda.is_available ():
            V_np = v.detach().cuda().data.cpu().numpy()
            Z_np = z.detach().cuda().data.cpu().numpy()
        else:
            V_np = v.detach().numpy()
            Z_np = z.detach().numpy()
        # 设置最近邻的数量
        k = 5
        # 使用最近邻算法查找最近的邻居
        v_nbrs = NearestNeighbors(n_neighbors=k).fit(V_np)
        z_nbrs = NearestNeighbors(n_neighbors=k).fit(Z_np)
        # 查找V和Z的最近邻
        v_distances, v_indices = v_nbrs.kneighbors(V_np)
        z_distances, z_indices = z_nbrs.kneighbors(Z_np)
        # 计算平均互信息
        mi_sum = 0.0
        for i in range(len(V_np)):
            v_neighbors = V_np[v_indices[i]]
            z_neighbors = Z_np[z_indices[i]]
            v_entropy = np.mean(
                [np.log(k / (np.sum(np.exp(-np.linalg.norm(v - v_neighbors, axis=1)))) + 1e-8) for v in v_neighbors])
            z_entropy = np.mean(
                [np.log(k / (np.sum(np.exp(-np.linalg.norm(z - z_neighbors, axis=1)))) + 1e-8) for z in z_neighbors])
            vz_entropy = np.mean([np.log(k / (np.sum(np.exp(-np.linalg.norm(v - v_neighbors, axis=1)) * np.exp(
                -np.linalg.norm(z - z_neighbors, axis=1)))) + 1e-8) for v, z in zip(v_neighbors, z_neighbors)])
            mi = v_entropy + z_entropy - vz_entropy
            mi_sum += mi
        # 计算平均互信息
        mi_estimate = mi_sum / len(V_np)
        return mi_estimate

    def cal_loss(self, v1, v2, y):
        # Encode a batch of data
        p_z1_given_v1 = self.encoder_v1(v1)
        p_z2_given_v2 = self.encoder_v2(v2)
        p_T1_given_v1 = self.encoder_vT1(v1)
        p_T2_given_v2 = self.encoder_vT2(v2)
        # Sample from the posteriors with reparametrization
        z1 = p_z1_given_v1.rsample()
        z2 = p_z2_given_v2.rsample()
        T1 = p_T1_given_v1.rsample()
        T2 = p_T2_given_v2.rsample()
        prob_x_given_z1 = self.decoder_v1(z1)
        distortion1 = -prob_x_given_z1.log_prob(v1.view(v1.shape[0], -1))
        distortion1 = distortion1.mean()
        prob_x_given_z2 = self.decoder_v2(z2)
        distortion2 = -prob_x_given_z2.log_prob(v1.view(v1.shape[0], -1))
        distortion2 = distortion2.mean()
        prob_x_given_T1 = self.decoder_vT1(T1)
        distortion3 = -prob_x_given_T1.log_prob(v2.view(v2.shape[0], -1))
        distortion3 = distortion3.mean()
        prob_x_given_T2 = self.decoder_vT2(T2)
        distortion4 = -prob_x_given_T2.log_prob(v1.view(v1.shape[0], -1))
        distortion4 = distortion4.mean()
        zT12_mi_gradient, zT12_mi_estimation = self.z_mi_estimator(T1, T2)
        zT12_mi_gradient = zT12_mi_gradient.mean()
        # Update the value of beta according to the policy
        beta = self.beta_scheduler(self.iterations)
        if self.iterations < self.flag:
            loss = beta * (distortion1 + distortion2 + distortion3 + distortion4)
        else:
            loss = - zT12_mi_gradient + self.cal_mi(v1, z1) + self.cal_mi(v2, z2)- self.cal_mi(v1, T2) - self.cal_mi(v2, T1)+ beta * (distortion1 + distortion2 + distortion3 + distortion4)
        self.iterations += 1
        return loss

class testLMIB_csv():
    def __init__(self):
        self.n_class = 10
        self.z_dim = 64
        self.nepochs = 1000
        self.input_dim1 = 256
        self.dim1 = 256
        self.input_dim2 = 256
        self.dim2 = 256
        self.encoder_lr = 0.0001
        self.decoder_lr = 0.0001
        self.miest_lr = 0.0001
        self.batch_size = 64
        self.ALPHA = 9e-5
        self.BETA = 1e-3
        self.GAMMA = 1e-3
        self.flag = 200
        self.beta_start_value = 1e-3
        self.beta_end_value = 1
        self.beta_n_iterations = 100000
        self.beta_start_iteration = 50000
        self.iterations = 0
        self.eval = None
        self.optimizer_name = 'Adam'
        self.model = None
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

    def opt_one_batch(self,batch) :
        x1, x2, y = batch
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        y = y.to(self.device)
        loss = self.model.cal_loss(x1, x2, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        Loss_dict={}
        Loss_dict['loss'] = float(loss.data.cpu().numpy()) #不管数据在gpu还是cpu都统一存入cpu
        return Loss_dict

    def eval_data(self, vloader, metric) -> float:
        embedded_data_t = EmbeddedDataset(self.batch_size, encoder1=self.model.encoder1, encoder2=self.model.encoder2,
                                        encoderT1=self.model.encoderT1, encoderT2=self.model.encoderT2,
                                        base_dataset=vloader, device=self.device)
        x, y = build_matrix(embedded_data_t)
        accuracy, _ = self._predict(x, y)
        return accuracy

    def train(self, ds,valid_ds= None, test_ds=None, valid_func=None,cb_progress=lambda x:None):
        ds = make_tuple(ds)
        train_loader = DataLoader(ds, self.batch_size, shuffle=False, num_workers=0)
        if valid_ds is not None:
            valid_ds = make_tuple(valid_ds)
            valid_loader = DataLoader(valid_ds, batch_size=self.batch_size)
        else:
            valid_loader = None
        if test_ds is not None:
            test_ds = make_tuple(test_ds)
            test_loader = DataLoader(test_ds, batch_size=self.batch_size)
        else:
            test_loader = None
        self.model = LMIB(self.input_dim1, self.dim1,self.input_dim2,self.dim2, self.z_dim, self.n_class, self.ALPHA, self.BETA, self.GAMMA, self.beta_start_value, self.beta_end_value, self.beta_n_iterations, self.beta_start_iteration, self.iterations, self.device, self.flag)
        # 把模型放到gpu或cpu上
        self.model.to(self.device)
        # 设置优化方法及相关参数
        self.opt = init_optimizer(self.optimizer_name, [
            {'params': self.model.encoder1.parameters(), 'lr': self.encoder_lr},
        ])
        self.opt.add_param_group(
            {'params': self.model.encoder2.parameters(), 'lr': self.encoder_lr}
        )
        self.opt.add_param_group(
            {'params': self.model.encoderT1.parameters(), 'lr': self.encoder_lr}
        )
        self.opt.add_param_group(
            {'params': self.model.encoderT2.parameters(), 'lr': self.encoder_lr}
        )
        self.opt.add_param_group(
            {'params': self.model.decoder1.parameters(), 'lr': self.decoder_lr}
        )
        self.opt.add_param_group(
            {'params': self.model.decoder2.parameters(), 'lr': self.decoder_lr}
        )
        self.opt.add_param_group(
            {'params': self.model.decoderT1.parameters(), 'lr': self.decoder_lr}
        )
        self.opt.add_param_group(
            {'params': self.model.decoderT2.parameters(), 'lr': self.decoder_lr}
        )
        self.opt.add_param_group(
            {'params': self.model.z_mi_estimator.parameters(), 'lr': self.miest_lr}
        )
        trainer = TrainerBase(self.nepochs)
        trainer.train(self, train_loader, valid_loader, test_loader, valid_func)
        # report new status
        cb_progress(1)

    def _predict(self, x, y):
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)
        Logist = LogisticRegression(solver='saga', multi_class='multinomial', tol=.1, C=self.n_class)
        Logist.fit(x, y)
        acc = Logist.score(x, y)
        pred = Logist.predict(x)
        return acc, pred

    def predict(self, valid_ds):
        valid_ds = make_tuple(valid_ds)
        valid_loader = DataLoader(valid_ds, batch_size=self.batch_size)
        embedded_data = EmbeddedDataset(self.batch_size, encoder1=self.model.encoder1, encoder2=self.model.encoder2,
                                        encoderT1=self.model.encoderT1, encoderT2=self.model.encoderT2,
                                        base_dataset=valid_loader, device=self.device)
        x, y = build_matrix(embedded_data)
        _, pred = self._predict(x, y)
        return pred

    def save(self, fio):
        fio.seek(0)
        torch.save(self.model.state_dict(), fio)

    def load(self, fio):
        fio.seek(0)
        self.model.load_state_dict(torch.load(fio))

    def class_name(self):
        # 模型名字
        return str(self.__class__)[8:-2].split('.')[-1].lower()

