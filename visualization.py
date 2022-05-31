#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import visdom
import time

def plot_result(test_result,test_label1,path,pre_len=3):
    ##all test result visualization
    fig1 = plt.figure(figsize=(7,1.5))
#    ax1 = fig1.add_subplot(1,1,1)
    a_pred = test_result[:,0]
    a_true = test_label1[:,0]
    plt.plot(a_pred,'r-',label='prediction')
    plt.plot(a_true,'b-',label='true')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_all.jpg')
    plt.show()
    ## oneday test result visualization
    fig1 = plt.figure(figsize=(7,1.5))
#    ax1 = fig1.add_subplot(1,1,1)

    a_pred = test_result[-1-(276-pre_len)*pre_len::pre_len,0]
    a_true = test_label1[-1-(276-pre_len)*pre_len::pre_len,0]
    plt.plot(a_pred,'r-',label="prediction")
    plt.plot(a_true,'b-',label="true")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_oneday.jpg')
    plt.show()

def plot_result_real(vis, test_result,test_label1,pre_len=3):
    a_pred = test_result[-1-(276-pre_len)*pre_len::pre_len,0]
    a_true = test_label1[-1-(276-pre_len)*pre_len::pre_len,0]
    vis.plot_many_stack_static(range(0, len(a_pred)), 
        [a_pred, a_true], ['a_pred', 'a_true'])

def plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,path):
    ###train_rmse & test_rmse 
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_rmse, 'r-', label="train_rmse")
    plt.plot(test_rmse, 'b-', label="test_rmse")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/rmse.jpg')
    plt.show()
    #### train_loss & train_rmse
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_loss,'b-', label='train_loss')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/train_loss.jpg')
    plt.show()

    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_rmse,'b-', label='train_rmse')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/train_rmse.jpg')
    plt.show()

    ### accuracy
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_acc, 'b-', label="test_acc")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_acc.jpg')
    plt.show()
    ### rmse
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_rmse, 'b-', label="test_rmse")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_rmse.jpg')
    plt.show()
    ### mae
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_mae, 'b-', label="test_mae")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_mae.jpg')
    plt.show()


class Visualizer(object):
    def __init__(self, env='main', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        # self.index = {}
        self.env = env
        self.log_text = ''

    def plot_one(self, x, y, name, xlabel='iter', ylabel=''):
        # x = self.index.get(name, 1)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=str(name),
                      opts=dict(title=name, xlabel=xlabel, ylabel=ylabel),
                      update=None if x == 0 else 'append'
                      )
        # self.index[name] = x + step

    def plot_many_stack(self, x, d, xlabel='iter',  ylabel=''):
        name = list(d.keys())
        name_total = " ".join(name)
        # x = self.index.get(name_total, 1)
        val = list(d.values())
        if len(val) == 1:
            y = np.array(val)
        else:
            y = np.array(val).reshape(-1, len(val))
        self.vis.line(Y=y, X=np.ones(y.shape)*x,
                      win=str(name_total),
                      opts=dict(legend=name, title=name_total, xlabel=xlabel, ylabel=ylabel),
                      update=None if x == 0 else 'append'
                      )
        # self.index[name_total] = x+1
    
    def plot_many_stack_static(self, x, y, name, xlabel='iter',  ylabel=''):
        self.vis.line(X=np.column_stack([x]*len(y)),Y=np.column_stack(y),win=" ".join(name),
            opts=dict(legend=name))
    
    def plot_images(self, images, nrow=1, win='images'):
        self.vis.images(images, nrow=nrow, win=win)

    def plot_image(self, image, win='image'):
        self.vis.image(image, win=win)


    def log(self, info, win='log_text'):
        self.log_text += ('[{}] {} <br>'.format(time.strftime('%m/%d_%H:%M:%S'),info))
        self.vis.text(self.log_text, win)


# if __name__ == '__main__':
#     vis = Visualizer(env='test2')
#     since = time.time()
#     while True:
#         for i in range(10):
#             loss1 = np.random.rand()
#             vis.plot_one(loss1, 'train loss', 5)
#             time.sleep(1)
#         loss2 = np.random.rand()
#         vis.plot_many_stack({'train':loss1, 'val':loss2})
#         time.sleep(0.2)
#         # vis.log(time.time()-since)

#         if time.time()-since > 30:
#             print('over')
#             break