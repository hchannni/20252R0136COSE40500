import sys
import os
import matplotlib.pyplot as plt
import numpy as np

class nn_linear_layer:
    def __init__(self, input_size, output_size, std=1):
        self.W = np.random.normal(0,std,(output_size,input_size))
        self.b = np.random.normal(0,std,(output_size,1))
    
    def forward(self,x):
        return np.dot(x, self.W.T) + self.b.T
    
    def backprop(self,x,dLdy):
        dLdW = np.dot(dLdy.T, x)
        dLdb = np.sum(dLdy, axis=0, keepdims=True)
        dLdx = np.dot(dLdy, self.W)
        return dLdW,dLdb,dLdx

    def update_weights(self,dLdW,dLdb):

        # parameter update
        self.W=self.W+dLdW
        self.b=self.b+dLdb

class nn_activation_layer:
    
    def __init__(self):
        pass
    
    def forward(self,x):
        return 1 / (1 + np.exp(-x)) # sigmoid
    
    def backprop(self,x,dLdy):
        sig_x = 1 / (1 + np.exp(-x))
        dydx = sig_x * (1 - sig_x)
        return dLdy * dydx


class nn_softmax_layer:
    def __init__(self):
        pass
    def forward(self,x):
        x_stable = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(x_stable)
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def backprop(self,x,dLdy):
        return dLdy

class nn_cross_entropy_layer:
    def __init__(self):
        pass
        
    def forward(self,x,y):
        n = x.shape[0]
        t = np.hstack((1-y, y))
        log_p = np.log(x + 1e-9)
        loss = -np.sum(t * log_p)
        return loss / n
        
    def backprop(self,x,y):
        n = x.shape[0]
        t = np.hstack((1 - y, y))
        dLds = x - t
        return dLds / n

if __name__ == '__main__':
    num_d=5

    num_test=40

    lr=0.1
    num_gd_step=10000

    batch_size=4*num_d

    num_class=2

    accuracy=0

    show_train_data=True

    show_loss=True

    m_d1 = (0, 0)
    m_d2 = (1, 1)
    m_d3 = (0, 1)
    m_d4 = (1, 0)

    sig = 0.05
    s_d1 = sig ** 2 * np.eye(2)

    d1 = np.random.multivariate_normal(m_d1, s_d1, num_d)
    d2 = np.random.multivariate_normal(m_d2, s_d1, num_d)
    d3 = np.random.multivariate_normal(m_d3, s_d1, num_d)
    d4 = np.random.multivariate_normal(m_d4, s_d1, num_d)

    x_train_d = np.vstack((d1, d2, d3, d4))
    y_train_d = np.vstack((np.zeros((2 * num_d, 1), dtype='uint8'), np.ones((2 * num_d, 1), dtype='uint8')))

    if (show_train_data):
        plt.grid()
        plt.scatter(x_train_d[range(2 * num_d), 0], x_train_d[range(2 * num_d), 1], color='b', marker='o')
        plt.scatter(x_train_d[range(2 * num_d, 4 * num_d), 0], x_train_d[range(2 * num_d, 4 * num_d), 1], color='r',
                    marker='x')
        plt.show()

    layer1 = nn_linear_layer(input_size=2, output_size=4, )
    act = nn_activation_layer()

    layer2 = nn_linear_layer(input_size=4, output_size=2, )
    smax = nn_softmax_layer()
    cent = nn_cross_entropy_layer()

    loss_out = np.zeros((num_gd_step))

    for i in range(num_gd_step):
        x_train = x_train_d
        y_train = y_train_d

        l1_out = layer1.forward(x_train)
        a1_out = act.forward(l1_out)
        l2_out = layer2.forward(a1_out)
        smax_out = smax.forward(l2_out)
        loss_out[i] = cent.forward(smax_out, y_train)

        b_cent_out = cent.backprop(smax_out, y_train)
        b_nce_smax_out = smax.backprop(l2_out, b_cent_out)
        b_dLdW_2, b_dLdb_2, b_dLdx_2 = layer2.backprop(x=a1_out, dLdy=b_nce_smax_out)
        b_act_out = act.backprop(x=l1_out, dLdy=b_dLdx_2)
        b_dLdW_1, b_dLdb_1, b_dLdx_1 = layer1.backprop(x=x_train, dLdy=b_act_out)

        layer2.update_weights(dLdW=-b_dLdW_2 * lr, dLdb=-b_dLdb_2.T * lr)
        layer1.update_weights(dLdW=-b_dLdW_1 * lr, dLdb=-b_dLdb_1.T * lr)
        
        if (i + 1) % 2000 == 0:
            print('gradient descent iteration:', i + 1)

    if (show_loss):
        plt.figure(1)
        plt.grid()
        plt.plot(range(num_gd_step), loss_out)
        plt.xlabel('number of gradient descent steps')
        plt.ylabel('cross entropy loss')
        plt.show()

    num_test = 100

    for j in range(num_test):
        
        predicted = np.ones((4,))

        sig_t = 1e-2

        t11 = np.random.multivariate_normal((1,1), sig_t**2*np.eye(2), 1)
        t00 = np.random.multivariate_normal((0,0), sig_t**2*np.eye(2), 1)
        t10 = np.random.multivariate_normal((1,0), sig_t**2*np.eye(2), 1)
        t01 = np.random.multivariate_normal((0,1), sig_t**2*np.eye(2), 1)
        
        l1_out = layer1.forward(t11)
        a1_out = act.forward(l1_out)
        l2_out = layer2.forward(a1_out)
        smax_out = smax.forward(l2_out)
        predicted[0] = np.argmax(smax_out)
        print('softmax out for (1,1)', smax_out, 'predicted label:', int(predicted[0]))
        
        l1_out = layer1.forward(t00)
        a1_out = act.forward(l1_out)
        l2_out = layer2.forward(a1_out)
        smax_out = smax.forward(l2_out)
        predicted[1] = np.argmax(smax_out)
        print('softmax out for (0,0)', smax_out, 'predicted label:', int(predicted[1]))
        
        l1_out = layer1.forward(t10)
        a1_out = act.forward(l1_out)
        l2_out = layer2.forward(a1_out)
        smax_out = smax.forward(l2_out)
        predicted[2] = np.argmax(smax_out)
        print('softmax out for (1,0)', smax_out, 'predicted label:', int(predicted[2]))
        
        l1_out = layer1.forward(t01)
        a1_out = act.forward(l1_out)
        l2_out = layer2.forward(a1_out)
        smax_out = smax.forward(l2_out)
        predicted[3] = np.argmax(smax_out)
        print('softmax out for (0,1)', smax_out, 'predicted label:', int(predicted[3]))
        
        print('total predicted labels:', predicted.astype('uint8'))
        
        accuracy += (predicted[0] == 0) & (predicted[1] == 0) & (predicted[2] == 1) & (predicted[3] == 1)
        
        if (j + 1) % 10 == 0:
            print('test iteration:', j + 1)

    print('accuracy:', accuracy / num_test * 100, '%')





