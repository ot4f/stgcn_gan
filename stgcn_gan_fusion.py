import torch
import torch.nn as nn
import numpy as np
from scipy.sparse.linalg import eigs
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import device
from stgcn_eval import multi_pred, stgcn_evaluate

DATA_PATHS = {
    "pems": {"feat": "data/PeMSD7_V_228.csv", "adj": "data/PeMSD7_W_228.csv"}
}

FLOAT_DTYPE = torch.float32

def load_pems_data(dataset):
    pems_adj = pd.read_csv(dataset['adj'],header=None)
    W = pems_adj.values
    n = pems_adj.shape[0]
    W = W/10000.
    W2 = W * W
    W_mask = np.ones([n, n]) - np.identity(n)
    w = np.exp(-W2/0.1)
    w = (w >= 0.5) * W_mask
    # w = np.mat(w)
    pems_feat = pd.read_csv(dataset['feat'],header=None)
    pems_feat = np.array(pems_feat)
    return pems_feat, w

############################### math graph ################
def scaled_laplacian(W):
    """
    Normalized graph Laplacian function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.matrix, [n_route, n_route].
    """
    # import pdb; pdb.set_trace()
    n = np.shape(W)[0]
    d = np.sum(W, axis=1)
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    return np.mat(2*L/lambda_max - np.identity(n))

def cheb_poly_approx(L, Ks):
    """
    Chebyshev polynomials approximation function. 
    :param L: np.matrix, [n_route, n_route], graph Laplacian.
    :param Ks: int, kernel size of spatial convolution.
    :return: np.ndarray, [n_route, Ks*n_route].
    """
    n = np.shape(L)[0]
    L0 = np.mat(np.identity(n))
    L1 = np.mat(L)

    if Ks > 1:
        L_list = [L0, L1]
        for i in range(Ks-2):
            Ln = np.mat(2 * L * L1 - L0)
            L_list.append(Ln)
            L0, L1 = L1, Ln
        return np.concatenate(L_list, axis=-1)
    elif Ks == 1:
        return np.asarray(L0)
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{Ks}".')

def first_approx(W):
    """
    1st-order approximation function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.ndarray, [n_route, n_route].
    """
    n = np.shape(W)[0]
    A = W + np.identity(n)
    d = np.sum(A, axis=1)
    sinvD = np.sqrt(np.mat(np.diag(d)).I)
    return np.mat(sinvD * A * sinvD)

# def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
#     """
#     Load weight matrix function.
#     :param file_path: str, the path of saved weight matrix file.
#     :param sigma2: float, scalar of matrix W.
#     :param epsilon: float, thresholds to control the sparsity of matrix W.
#     :param scaling: bool, whether applies numerical scaling on W.
#     :return: np.ndarray, [n_route, n_route].
#     """
#     try:
#         W = pd.read_csv(file_path, header=None).values
#     except FileNotFoundError:
#         print(f'ERROR: input file was not found in {file_path}.')

#     # check whether W is a 0/1 matrix.
#     if set(np.unique(W)) == {0.0, 1.0}:
#         print('The input graph is a 0/1 matrix; set "scaling" to False.')
#         scaling = False

#     if scaling:
#         n = W.shape[0]
#         W = W / 10000.
#         W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
#         # refer to Eq.10
#         return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
#     else:
#         return W

@torch.jit.script
def fused_sigmoid(x):
    return torch.sigmoid(x)

@torch.jit.script
def fused_add(x,y):
    return torch.add(x, y)

@torch.jit.script
def fused_mul(x, y):
    return torch.mul(x,y)

@torch.jit.script
def fused_relu(x):
    return F.relu(x)


########################
class Gconv(nn.Module):
    def __init__(self, Ks, c_in, c_out, seq_len=12):
        """
        :param adj: adj weight matrix, [n_route, n_route]
        :param Ks: int, kernel size of graph convolution.
        :param c_in: int, size of input channel.
        :param c_out: int, size of output channel.
        :return: tensor, [batch_size, n_route, c_out].
        """
        super(Gconv, self).__init__()
        # L = scaled_laplacian(adj)
        # Lk = cheb_poly_approx(L, Ks)
        # self.register_buffer(
        #     "laplacian", torch.FloatTensor(Lk)
        # )
        self.weights = nn.Parameter(
            torch.FloatTensor(seq_len, c_in, c_out)
        )
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.seq_len = seq_len
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("relu"))

    def forward(self, x, adj):
        """
        :param x: tensor, [batch, seq_len, n_route, c_in].
        :param adj: ndarray, shape [batch, seq_len, n_route, n_route]
        :return: tensor, [batch, seq_len, n_route, c_out].
        """
        # import pdb; pdb.set_trace()
        batch_size, seq_len, n_route, _ = x.shape
        # print(x.shape)
        # adj = adj.reshape(-1, n_route, n_route)
        # laps = []
        # for i in range (adj.shape[0]):
        #     L = scaled_laplacian(adj[i])
        #     lap = cheb_poly_approx(L, self.Ks)
        #     laps.append(lap)
        # import pdb; pdb.set_trace()
        # laplacian = torch.FloatTensor(np.stack(laps)).to(device)
        # laplacian = laplacian.reshape(batch_size, seq_len, n_route, -1)
        eye = torch.eye(n_route,device=device,dtype=FLOAT_DTYPE).unsqueeze(0).unsqueeze(0).\
            expand(batch_size, seq_len, n_route, n_route)
        # print(adj.shape, eye.shape)
        adjacency = fused_add(adj, eye)
        diag = fused_mul(adjacency.sum(dim=-1, keepdim=True).pow(-0.5).expand(adjacency.size()), eye)
        adjacency = diag.matmul(adjacency).matmul(diag)

        weights = self.weights.unsqueeze(0).expand(batch_size, seq_len, self.c_in, self.c_out)
        # x = x.permute(0, 1, 3, 2)  # [..., cin, n_route]
        # x = torch.matmul(x, laplacian).reshape(batch_size, seq_len, self.c_in, self.Ks , n_route)
        # x = x.reshape(0, 1, -1, n_route).permute(0, 1, 3, 2)
        # x = torch.matmul(x, weights)
        # x = x.permute(0, 2, 1).reshape(-1, n) # [batch, c_in, n] -> [batch*c_in, n]
        # x = torch.matmul(x, laplacian).reshape(-1, self.c_in, self.Ks, n) 
        # x = x.permute(0, 3, 1, 2).reshape(-1, self.c_in * self.Ks)
        # x = torch.matmul(x, self.weights).reshape(-1, n, self.c_out)
        output = torch.matmul(adjacency, x).matmul(weights)
        return output


class TemporalConv(nn.Module):
    def __init__(self, Kt, c_in, c_out, act_func='relu'):
        super(TemporalConv, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.act_func = act_func
        
        self.conv1 = nn.Conv2d(c_in, c_out, 1, 1)
        if act_func == 'GLU':
            self.conv2 = nn.Conv2d(c_in, 2 * c_out, (Kt, 1), 1, padding=((Kt-1)//2,0))
        else:
            self.conv2 = nn.Conv2d(c_in, c_out, (Kt, 1), 1, padding=((Kt-1)//2,0))
            
    def forward(self, x):
        """
        :param x: tensor, [batch_size, time_step, n_route, c_in].
        :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out].
        """
        _, T, n, _ = x.shape
        
        x = x.permute(0, 3, 1, 2).contiguous() # [batch_size, c_in, time_step, n_route]

        # print(x.dtype)

        if self.c_in > self.c_out:
            x_input = self.conv1(x)
        elif self.c_in < self.c_out:
            x_input = torch.cat([x, torch.zeros(x.shape[0], self.c_out - self.c_in, T, n, device=device, dtype=FLOAT_DTYPE)], dim=1)
        else:
            x_input = x
        
        # keep the original input for residual connection
        x_input = x_input[:, :, :T, :]

        x_conv = self.conv2(x)
        torch.cuda.nvtx.range_push("before_GLU")
        if self.act_func == 'GLU':
            x_output = fused_mul(fused_add(x_conv[:, 0:self.c_out, :, :], x_input), fused_sigmoid(x_conv[:, self.c_out:, :, :]))
        else:
            if self.act_func == 'linear':
                x_output = x_conv
            elif self.act_func == 'sigmoid':
                x_output = fused_sigmoid(x_conv)
            elif self.act_func == 'relu':
                x_output = fused_relu(fused_add(x_conv, x_input))
            else:
                raise ValueError(f'ERROR: activation function "{self.act_func}" is not defined.')
        x_output = x_output.permute(0, 2, 3, 1)
        torch.cuda.nvtx.range_pop()

        return x_output

class TemporalConv1(nn.Module):
    def __init__(self, Kt, c_in, c_out, act_func='relu'):
        super(TemporalConv1, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.act_func = act_func
        
        self.conv1 = nn.Conv2d(c_in, c_out, 1, 1)
        if act_func == 'GLU':
            self.conv2 = nn.Conv2d(c_in, 2 * c_out, (Kt, 1), 1)
        else:
            self.conv2 = nn.Conv2d(c_in, c_out, (Kt, 1), 1)
            
    def forward(self, x):
        """
        :param x: tensor, [batch_size, time_step, n_route, c_in].
        :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out].
        """
        _, T, n, _ = x.shape
        
        x = x.permute(0, 3, 1, 2).contiguous() # [batch_size, c_in, time_step, n_route]

        if self.c_in > self.c_out:
            x_input = self.conv1(x)
        elif self.c_in < self.c_out:
            x_input = torch.cat([x, torch.zeros(x.shape[0], self.c_out - self.c_in, T, n, device=device,dtype=FLOAT_DTYPE)], dim=1)
        else:
            x_input = x
        
        # keep the original input for residual connection
        x_input = x_input[:, :, self.Kt-1:T, :]

        x_conv = self.conv2(x)
        if self.act_func == 'GLU':
            x_output = fused_mul(fused_add(x_conv[:, 0:self.c_out, :, :], x_input), fused_sigmoid(x_conv[:, self.c_out:, :, :]))
        else:
            if self.act_func == 'linear':
                x_output = x_conv
            elif self.act_func == 'sigmoid':
                x_output = fused_sigmoid(x_conv)
            elif self.act_func == 'relu':
                x_output = fused_relu(fused_add(x_conv,x_input))
            else:
                raise ValueError(f'ERROR: activation function "{self.act_func}" is not defined.')
        x_output = x_output.permute(0, 2, 3, 1)

        return x_output
        

class SpatioConv(nn.Module):
    def __init__(self, Ks, c_in, c_out):
        super(SpatioConv, self).__init__()
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        
        self.conv1 = nn.Conv2d(c_in, c_out, 1, 1)
        self.gconv = Gconv(Ks, c_in, c_out)

    def forward(self, x, adj):
        """
        :param x: tensor, [batch_size, time_step, n_route, c_in].
        :return: tensor, [batch_size, time_step, n_route, c_out].
        """
        _, T, n, _ = x.shape

        if self.c_in > self.c_out:
            x_input = self.conv1(x.permute(0,3,1,2)).permute(0,2,3,1)
        elif self.c_in < self.c_out:
            x_input = torch.cat([x, torch.zeros(x.shape[0], T, n, self.c_out - self.c_in, device=device, dtype=FLOAT_DTYPE)], dim=1)
        else:
            x_input = x
        
        # x = x.reshape(-1, n, self.c_in)
        x = self.gconv(x, adj)
        # x = x.reshape(-1, T, n, self.c_out)
        x = fused_relu(fused_add(x[:, :, :, 0:self.c_out],x_input))
        return x


class STConvBlock(nn.Module):
    def __init__(self, n_route, Ks, Kt, channels, keep_prob, act_func='GLU', layer_norm=True):
        """
        :param Ks: int, kernel size of spatial convolution.
        :param Kt: int, kernel size of temporal convolution.
        :param channels: list, channel configs of a single st_conv block.
        :param keep_prob: placeholder, prob of dropout.
        :param act_func: str, activation function.
        """
        super(STConvBlock, self).__init__()
        self.Ks = Ks
        self.Kt = Kt
        
        c_si, c_t, c_oo = channels
        self.tconv1 = TemporalConv(Kt, c_si, c_t, act_func=act_func)
        self.sconv = SpatioConv(Ks, c_t, c_t)
        self.tconv2 = TemporalConv(Kt, c_t, c_oo, act_func=act_func)
        self.layer_norm = nn.LayerNorm([n_route, c_oo]) if layer_norm else None
        self.dropout = nn.Dropout(keep_prob)

    def forward(self, x, adj):
        x = self.tconv1(x)
        x = self.sconv(x, adj)
        x = self.tconv2(x)
        if self.layer_norm:
            x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class STGCN(nn.Module):
    def __init__(self, n_route, n_his, Ks, Kt, blocks, keep_prob, act_func='GLU'):
        """
        :param adj: adj weight matrix, [n_route, n_route]
        :param n_his: int, size of historical records for training.
        :param Ks: int, kernel size of spatial convolution.
        :param Kt: int, kernel size of temporal convolution.
        :param blocks: list, channel configs of st_conv blocks.
        :param keep_prob: placeholder.
        """
        super(STGCN, self).__init__()
        self.Ks = Ks
        self.Kt = Kt
        # self.n_his = n_his
        self.n_blocks = len(blocks)
        # self.Ko = n_his - 2 * (Kt-1) * len(blocks)
        self.Ko = n_his
        assert self.Ko > 1

        self.n_route = n_route
        self.c_out = blocks[-1][-1]

        self.st_conv_blocks = nn.ModuleList()
        for _, channels in enumerate(blocks):
            self.st_conv_blocks.append(STConvBlock(n_route, Ks,Kt,channels,keep_prob,act_func=act_func))
        # self.st_convs = nn.Sequential(*st_conv_blocks)
        self.output_layer = nn.Sequential(
            TemporalConv1(self.Ko, self.c_out, self.c_out, act_func=act_func),
            nn.LayerNorm([self.n_route, self.c_out]),
            TemporalConv1(1, self.c_out, self.c_out, act_func='sigmoid'),
        )
        self.conv_out = nn.Conv2d(self.c_out, 1, 1, 1)

    def forward(self, x, adj):
        """
        :param x: tensor, [batch_size, time_step, n_route, c_in].
        :return: tensor, [batch_size, n_route, c_out]
        """
        for layer in self.st_conv_blocks:
            x = layer(x, adj)
        # x = self.st_convs(x, adj)
        x = self.output_layer(x)  # shape [batch_size, 1, n_route, c_out]
        # x = x.reshape(-1, self.n_route, self.c_out)
        x = x.permute(0, 3, 1, 2) 
        x = self.conv_out(x)
        x = x.permute(0, 2, 3, 1)  # shape [batch_size, 1, n_route, 1]
        return x



class Generator(nn.Module):
    def __init__(self, n_route, n_his, Ks, Kt, blocks, keep_prob, act_func='GLU'):
        super(Generator, self).__init__()
        self.stgcn = STGCN(n_route, n_his, Ks, Kt, blocks, keep_prob, act_func=act_func)

    def forward(self, x, adj):
        """
        :param x: tensor, shape (batch_size, time_step, n_route, c_in)
        :param adj: shape (n_route, n_route)
        """
        # print('In Generator', x.shape)
        # import pdb; pdb.set_trace()
        torch.cuda.nvtx.range_push("before_generator")
        batch_size, seq_len, n_route, _ = x.shape
        noise_weight = (0.99*torch.rand(batch_size, seq_len, n_route, n_route, device=device, dtype=FLOAT_DTYPE)+0.01)
        noise_weight = (noise_weight+noise_weight.transpose(2,3))/2.0
        w = noise_weight * adj
        output = self.stgcn(x, w)
        torch.cuda.nvtx.range_pop()

        return output


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(Discriminator, self).__init__() 

        self.ffn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        :param x: shape (batch_size, 1, n_route, 1)
        """
        x = x.view(x.shape[0], -1)
        h = self.ffn1(x)
        output = self.ffn2(h)
        output = output.squeeze(-1)
        return output


def test(model, test_loader, step=0):
    with torch.no_grad():
        pred_list = []
        gt_list = []
        for _, (test_x, test_y) in enumerate(test_loader):
            test_x = test_x.to(device)
            test_y = test_y.to(device)

            pred = multi_pred1(model, test_x, test_y, seq_len, pred_len, A)
            pred_list.append(pred)
            gt_list.append(test_y)
        
        pred = torch.cat(pred_list, dim=1)
        gt = torch.cat(gt_list, dim=0)

        loss = torch.mean((pred[0:1, ...].squeeze(0)-gt[:,0:1,...].squeeze(1))**2)
        metrics = stgcn_evaluate(model, test_x, gt, seq_len, pred_len, x_stats, pred_len-1, pred)
        rmse, mae, acc, r2, var, mape = [metrics[0][k] for k in ['rmse', 'mae', 'acc', 'r2', 'var', 'mape']]

        info = f'Test - loss: {loss:.4f}, rmse: {rmse:.4f}, mae: {mae:.4f}, acc: {acc:.4f}, r2: {r2:.4f}, var: {var:.4f}, mape: {mape:.4%}'
        print(info)
        print('%s,%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f%%,%.4f' % ('stgcn_gan','pems',pred_len,n_epoch,rmse,mae,acc,r2,var,mape*100,loss))
        if step > 0:
            vis.plot_one(step, loss.item(), 'val_loss')
            vis.plot_many_stack(step, {'rmse': rmse, 'mae': mae, 'mape(%)': mape*100})
            vis.plot_many_stack(step, {'acc': acc, 'r2': r2, 'var': var})


def train():
    step = 0
    since = time.time()
    for epoch in range(n_epoch):
        for j, data in enumerate(train_loader):
            step += 1
            train_x, train_y = data
            train_x = train_x.type(FLOAT_DTYPE).to(device)
            train_y = train_y.type(FLOAT_DTYPE).to(device)

            # print(train_x.dtype, train_y.dtype)
            
            for _ in range(n_critic):
                fake_y = generator(train_x, A)
                mse_loss = criterion(fake_y, train_y[:,0:1,:,:])
                critic_real = discriminator(train_y[:, 0:1, :, :]).mean()
                critic_fake = discriminator(fake_y).mean()

                loss_critic = -critic_real + critic_fake 
                opt_critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()
            
                for p in discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)
            
            gen_fake = discriminator(fake_y).mean()
            loss_gen = -gen_fake
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            print('[epoch: %d, iter: %d] loss: %.4f' % (epoch, j+1, mse_loss.item()))
            vis.plot_one(step, mse_loss.item(), 'train_loss')
            vis.plot_one(step, gen_fake.item(), 'g_fake')
            vis.plot_many_stack(step, {'d_loss': loss_critic.item(), 'g_loss': loss_gen.item()})
            vis.plot_many_stack(step, {'d_real': critic_real.item(), 'd_fake': critic_fake.item()})
           
            if (step) % 20 == 0:
                generator.eval()
                discriminator.eval()

                test(generator, val_loader, step)

                generator.train()
                discriminator.train()
        print('Epoch: %d, spend: %.4f' % (epoch, time.time()-since))
        if lr_scheduler_on:
            vis.plot_many_stack(epoch+1, {'g_lr': opt_gen.param_groups[0]['lr'], 
            'd_lr': opt_critic.param_groups[0]['lr']}, xlabel='epoch')
            # if (epoch+1) % gen_step == 0:
            gen_lr_scheduler.step()
            # if (epoch+1) % critic_step == 0:
            critic_lr_scheduler.step()
        

if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    from utils import torch_dataset
    from visualization import Visualizer
    from stgcn_eval import multi_pred1, stgcn_evaluate
    import argparse
    import time

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--test', action='store_true')
    argparser.add_argument('--epoch', type=int, default=50)
    argparser.add_argument('--lr_scheduler', action='store_true')
    argparser.add_argument('--vis', default='',help='vis name')

    args = argparser.parse_args()
    print(args)

    testing = args.test

    vis = None
    if not testing:
        vis_name = args.vis
        if vis_name == "":
            vis_name = str(args.epoch)
        vis = Visualizer(env='z_stgcn_gan_%s' % vis_name)
    
    feat, A = load_pems_data(DATA_PATHS['pems'])
    # plt.imshow(w, cmap=plt.cm.YlGnBu)
    # plt.colorbar()
    # plt.show()
    blocks = [[1,32,64],[64,32,128]]
    n_route_ = A.shape[0]
    generator = Generator(n_route_, 12, 3, 3, blocks, 0.5, 'GLU')
    discriminator = Discriminator(n_route_*1, 512)
    
    if FLOAT_DTYPE == torch.float16:
        generator.half()
        discriminator.half()
    generator.to(device)
    discriminator.to(device)

    time_len = feat.shape[0]
    test_rate = 0.1
    seq_len = 12
    pred_len = 3
    normalize = 'zscore'
    C_0 = 1
    batch_size = 64
    train_dataset, test_dataset, val_dataset, x_stats = \
        torch_dataset(feat, time_len, test_rate, seq_len, pred_len, normalize=normalize, n_day=44, day_slot=288, C_0=C_0)
    # scale = x_stats['std']
    # bias = x_stats['mean']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    initial_lr = 1e-3
    n_epoch = args.epoch
    criterion = nn.MSELoss(reduction='sum')

    opt_gen = torch.optim.RMSprop(generator.parameters(), lr=initial_lr, weight_decay=0)
    opt_critic = torch.optim.RMSprop(discriminator.parameters(), lr=initial_lr, weight_decay=0)
    
    gen_lr_scheduler = torch.optim.lr_scheduler.StepLR(opt_gen, step_size=10, gamma=0.7, last_epoch=-1)
    critic_lr_scheduler = torch.optim.lr_scheduler.StepLR(opt_critic, step_size=10, gamma=0.7, last_epoch=-1)

    lr_scheduler_on = args.lr_scheduler
    n_critic = 5
    # gen_step = 10
    # critic_step = 10

    A = torch.from_numpy(A).type(FLOAT_DTYPE).to(device)

    if testing:
        generator.load_state_dict(torch.load('output/stgcn_gan_%d.pth' % n_epoch))
        test(generator, test_loader)
    else:
        train()
        torch.save(generator.state_dict(), 
            'output/stgcn_gan_%d.pth' % (n_epoch))
        test(generator, test_loader)
            






