import numpy as np
import torch
from torch.utils.data import Subset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

class EmbeddedDataset:
    def __init__(self, batch_size, base_dataset, encoder1, encoder2=None, encoderT1=None, encoderT2=None, device='cpu'):
        self.BLOCK_SIZE = batch_size
        if encoderT1 == None:
            encoder1 = encoder1.to(device)
        else:
            encoder1 = encoder1.to(device)
            encoder2 = encoder2.to(device)
            encoderT1 = encoderT1.to(device)
            encoderT2 = encoderT2.to(device)
        self.means, self.target = self._embed(encoder1, encoder2, encoderT1, encoderT2, base_dataset, device)

    def _embed(self, encoder1, encoder2, encoderT1, encoderT2,  dataset, device):
        if encoderT1 == None:
            encoder1.eval()
        else:
            encoder1.eval()
            encoder2.eval()
            encoderT1.eval()
            encoderT2.eval()
        ys = []
        reps = []
        with torch.no_grad():
            for a in dataset:
                if len(a)==3:
                    x = a[0]
                    x2 = a[1]
                    x2 = x2.to(device)
                    y = a[2]
                else:
                    x = a[0]
                    y = a[1]
                x = x.to(device)
                y = y.to(device)
                if encoderT1 == None:
                    p_z_given_x = encoder1(x)
                    reps.append(p_z_given_x.mean.detach())
                else:
                    p_z_given_x = encoder1(x)
                    p_T1_given_x = encoderT1(x)
                    p_z2_given_x2 = encoder2(x2)
                    p_T2_given_x = encoderT2(x2)
                    reps.append((p_z_given_x.mean+p_T1_given_x.mean+p_T2_given_x.mean+p_z2_given_x2.mean).detach())
                ys.append(y)
            ys = torch.cat(ys, 0)
        return reps, ys

    def __getitem__(self, index):
        y = self.target[index]
        x = self.means[index // self.BLOCK_SIZE][index % self.BLOCK_SIZE]
        return x, y

    def __len__(self):
        return self.target.size(0)

def split(dataset, size, split_type):
    if split_type == 'Random':
        data_split, _ = torch.utils.data.random_split(dataset, [size, len(dataset) - size])
    elif split_type == 'Balanced':
        class_ids = {}
        for idx, (_, y) in enumerate(dataset):
            if isinstance(y, torch.Tensor):
                y = y.item()
            if y not in class_ids:
                class_ids[y] = []
            class_ids[y].append(idx)
        ids_per_class = size // len(class_ids)
        selected_ids = []
        for ids in class_ids.values():
            seed = 1
            np.random.seed(seed)
            selected_ids += list(np.random.choice(ids, min(ids_per_class, len(ids)), replace=False))
        data_split = Subset(dataset, selected_ids)
    return data_split

def build_matrix(dataset):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    xs = []
    ys = []
    for x, y in data_loader:
        xs.append(x)
        ys.append(y)
    xs = torch.cat(xs, 0)
    ys = torch.cat(ys, 0)
    if xs.is_cuda:
        xs = xs.cpu()
    if ys.is_cuda:
        ys = ys.cpu()
    return xs.data.numpy(), ys.data.numpy()

def evaluate(batch_size, encoder1, data, class_num, device):
    embedded_data = EmbeddedDataset(batch_size, data, encoder1, device=device)
    return train_and_evaluate_linear_model(embedded_data,class_num)

def evaluate_test(batch_size, encoder1, encoder2, encoderT1, encoderT2, data, class_num, device):
    embedded_data = EmbeddedDataset(batch_size, data, encoder1, encoder2, encoderT1, encoderT2, device=device)
    return train_and_evaluate_linear_model(embedded_data,class_num)

def train_and_evaluate_linear_model_from_matrices(x_train, y_train, solver='saga', multi_class='multinomial', tol=.1, C=10):
    model = LogisticRegression(solver=solver, multi_class=multi_class, tol=tol, C=C)
    model.fit(x_train, y_train)
    return model

def train_and_evaluate_linear_model(data, C, solver='saga', multi_class='multinomial', tol=.1):
    x, y = build_matrix(data)
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    model = LogisticRegression(solver=solver, multi_class=multi_class, tol=tol, C=C)
    model.fit(x, y)
    accuracy = model.score(x, y)
    return model,scaler, accuracy
