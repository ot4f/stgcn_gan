import numpy as np
import torch

def multi_pred(model, x_test, y_test, n_his, n_pred):
    """
    :param model
    :param y_pred, shape [batch_size, 1, n_route, 1]
    :param x_test, shape [bacth_size, n_his, n_route, 1]
    :param y_test, shape [batch_size, n_pred, n_route, 1]
    """
    model.eval()

    # y_pred = pred.data.cpu().numpy()
    # x_test = x_test.data.cpu().numpy()
    # y_test = y_test.data.cpu().numpy()

    test_seq = torch.clone(torch.cat([x_test, y_test[:, 0:1, :, :]], axis=1))

    step_list = []
    with torch.no_grad():
        for _ in range(n_pred):
            pred = model(test_seq[:, 0:n_his, :, :])
            pred = pred[:, 0, :, :]
            # print(pred.shape)
            test_seq[:, 0:n_his-1, : ,:] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his-1, :, :] = pred
            step_list.append(pred)
    return torch.stack(step_list)  # shape [n_pred, batch_size, n_route, C]


def multi_pred1(model, x_test, y_test, n_his, n_pred, adj):
    """
    :param model
    :param y_pred, shape [batch_size, 1, n_route, 1]
    :param x_test, shape [bacth_size, n_his, n_route, 1]
    :param y_test, shape [batch_size, n_pred, n_route, 1]
    """
    model.eval()

    # y_pred = pred.data.cpu().numpy()
    # x_test = x_test.data.cpu().numpy()
    # y_test = y_test.data.cpu().numpy()

    test_seq = torch.clone(torch.cat([x_test, y_test[:, 0:1, :, :]], axis=1))

    step_list = []
    with torch.no_grad():
        for _ in range(n_pred):
            pred = model(test_seq[:, 0:n_his, :, :], adj)
            pred = pred[:, 0, :, :]
            # print(pred.shape)
            test_seq[:, 0:n_his-1, : ,:] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his-1, :, :] = pred
            step_list.append(pred)
    return torch.stack(step_list)  #

def get_metric(name=None):
    def _MAPE(v, v_): 
        # print(v.shape, v_.shape)
        # mask = (v!=0)
        return np.mean(np.abs(v_-v)/(v+1e-5))
    def _RMSE(v, v_):
        return np.sqrt(np.mean((v_-v)**2))
    def _MAE(v, v_):
        return np.mean(np.abs(v_-v))
    def _ACC(v, v_):
        _, node_num, _ = v.shape
        v = v.reshape(-1, node_num)
        v_ = v_.reshape(-1, node_num)
        return 1 - np.linalg.norm(v-v_,'fro')/np.linalg.norm(v,'fro')
    def _R2(v, v_):
        return 1-np.sum((v-v_)**2)/np.sum((v-np.mean(v))**2)
    def _VAR(v, v_):
        return 1-np.var(v-v_)/np.var(v)
    
    metrics_funcs = {'mape': _MAPE, 'rmse': _RMSE, 'mae': _MAE, 
        'acc': _ACC, 'r2': _R2, 'var': _VAR}
    if name is None:
        return list(metrics_funcs.keys())
    else:
        if name in metrics_funcs:
            return metrics_funcs[name]
        else:
            raise ValueError(f"No metrics {name} found")

def stgcn_evaluate(model, x_test, y_test, n_his, n_pred, x_stats, step_idx, y_pred=None):
    if y_pred is None:
        y_pred = multi_pred(model, x_test, y_test, n_his, n_pred)
    y_test = y_test.transpose(0,1)  # [n_pred, batch_size, n_route,C] 
    # print(y_test.shape, y_pred.shape)
    mean = x_stats['mean']
    std = x_stats['std']

    metric_keys = get_metric()
    step_metrics = []
    if isinstance(step_idx, int):
        step_idx = [step_idx]    
    for step in step_idx:
        v = y_test[step] * std + mean
        v_ = y_pred[step] * std + mean
        metrics = {}
        for key in metric_keys:
            metric = get_metric(key)(v.data.cpu().numpy(), v_.data.cpu().numpy())
            metrics[key] = metric
        step_metrics.append(metrics)
    return step_metrics
