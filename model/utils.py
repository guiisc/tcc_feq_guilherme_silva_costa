import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse_f, r2_score


def plot_identity_graphic(data_true, data_pred):
    plt.figure(figsize=(13, 6))
    m, b = np.polyfit(data_true, data_pred, 1)
    limit_high = np.max([data_pred.max(), data_true.max()])
    limit_low = np.min([data_pred.min(), data_true.min()])
    limits = np.array([limit_low, limit_high])
    r2 = r2_score(data_true, data_pred)
    mse = mse_f(data_true, data_pred)
    plt.plot(data_true, data_pred, 'ko', fillstyle='none')
    plt.plot(limits, limits, 'gray', label='identity')
    plt.plot(limits, m*limits+b, 'r:', linewidth=3, label=f'{m:.2f} x+{b:.2f}')
    plt.legend()
    plt.xlabel('true value')
    plt.ylabel('predicted value')
    plt.title(f'$R^2$={r2:.2f}, RMSE={np.sqrt(mse):.2f}, RMSRE: {RMSRE(data_true, data_pred):.2f} %', fontsize=10)
    plt.grid(color='gray', linestyle='--', linewidth=0.25)


def plot_boxplot_input(data, data_labels, name=''):
    plt.figure(figsize=(6, 4))
    plt.boxplot(data)
    plt.xticks(np.arange(len(data_labels))+1, data_labels, rotation='45')
    plt.grid(':', linewidth=0.25)
    plt.title('Boxplot of the ' + name + ' variables')
    
def plot_loss(model):
    plt.figure(figsize=(13, 6))
    loss = np.array(model.loss_curve_)
    y_text = loss.max()/100
    plt.plot(loss)
    plt.axhline(np.min(loss), color="red")
    plt.text(0, np.min(loss), f'{np.min(loss):.2f}')
    
    loss_lim = loss[loss > 1.05*np.min(loss)]
    plt.axvline(loss_lim.shape[0], 0, loss_lim[-1], color='red')
    plt.text(loss_lim.shape[0], y_text, f'{loss_lim.shape[0]}')
    plt.grid()


def plot_weight(weight, data, data_labels):
    plt.figure(figsize=(13, 6))
    plt.imshow(np.abs(weight).T, cmap='Greys',
               aspect='auto', interpolation='bicubic', origin='lower')
    cbar = plt.colorbar()
    cbar.set_label('Weight values (absolute)', rotation=270)
    plt.xticks(np.arange(len(data_labels)),
               data_labels, rotation='45', fontsize=8)
    plt.yticks(np.arange(weight.shape[1]),
               np.arange(weight.shape[1])+1, fontsize=8)
    plt.ylabel('Neuron index')
    plt.title('Weight matrix of the first hidden layer')
    plt.grid()


def plot_parameters_scores(g_s):
    plt.plot(figsize=(6, 5))
    score_array = g_s.cv_results_['mean_test_score']*-1
    params = g_s.cv_results_['params']
    keys = [*params[0].keys()]
    h_l_parm = []
    for param in params:
        aux = []
        for key in keys:
            aux += [str(param[key])]
        h_l_parm += ['-'.join(aux)]
    h_l_idx = np.arange(len(h_l_parm))
    plt.figure(figsize=(6, 4))
    plt.bar(h_l_idx, score_array, color='gray')
    plt.bar(np.argmin(score_array), np.min(score_array), color='blue')
    plt.bar(np.argmax(score_array), np.max(score_array), color='red')
    plt.xticks(h_l_idx, h_l_parm, rotation=90, fontsize=7)
    plt.ylabel('MSE')
    plt.xlabel('Hidden layer configurations')
    plt.title('MSE for the different hidden layer configurations')
    plt.grid(linewidth=0.25)


def plot_mlp(reg, inputs):
    plt.plot(figsize=(10, 8))
    plt.axis('off')
    for layer in range(len(reg.coefs_)):
        w = reg.coefs_[layer]
        y = [np.linspace(-w.shape[i]/2, w.shape[i]/2, w.shape[i])
             for i in range(2)] # y pos of nodes
        x = [layer+i+np.ones(w.shape[i]) for i in range(2)] # x pos of nodes
        for i in range(2):
            plt.plot(x[i], y[i], 'ko', fillstyle='none', markersize=15)
        for i in range(len(x[0])):
            for j in range(len(x[1])):
                plt.plot([x[0][i], x[1][j]], [y[0][i], y[1][j]],
                         'k--')#, linewidth=0.75*(abs(w)/w.max())[i, j])
        if layer < len(reg.coefs_):
            if layer == 0:
                I = inputs
            else:
                I = f
            aux = (I@w+reg.intercepts_[layer]).copy()
            if reg.get_params()['activation'] == 'identity':
                f = aux
            if reg.get_params()['activation'] == 'logistic':
                f = 1 / (1 + np.exp(-aux))
            if reg.get_params()['activation'] == 'tanh':
                f = np.tanh(aux)
            if reg.get_params()['activation'] == 'relu':
                f = aux
                f[aux <= 0] = 0
        if layer < len(reg.coefs_)-1:
            plt.scatter(x[1], y[1], c=np.abs(f), s=100, cmap='Greys')
        if layer == 0:
            for i in range(len(x[0])):
                plt.text(x[0][i]-0.20, y[0][i],
                         '{:0.2f}'.format(inputs[i]), fontsize=8)
        if layer == len(reg.coefs_)-1:
            outs = reg.predict(inputs.reshape(1, -1))[0]
            try:
                for i in range(len(x[1])):
                    out = outs[i]
                    txt = '{:.2f}'.format(out)
                    plt.text(x[1][i]+0.15, y[1][i], txt, fontsize=8)
            except:
                out = outs
                txt = '{:.2f}'.format(out)
                plt.text(x[1][0]+0.15, y[1][0], txt, fontsize=8)

    return

def RMSRE(true_values, predict_values):
    """
    Evaluate metric
    """
    relativeSE = ((predict_values-true_values)*100/true_values)**2
    MrelativeSE = np.mean( relativeSE )
    return np.sqrt(MrelativeSE)



def plot_first_layer_keras(model, label=None):
    plt.figure(figsize=(10,6))
    first_layer = model.get_weights()[0]
    plt.imshow(abs(first_layer).T, cmap='Greys', aspect='auto', interpolation='bicubic', origin='lower')
    cbar = plt.colorbar()
    cbar.set_label('Absolute Weight Values', rotation=270, labelpad=12, fontsize=10)
    if label is None:
        label = np.array([f'Param {i+1}' for i in range(first_layer.shape[0])])
    else:
        label = np.array(label)
    plt.xticks(np.arange(label.shape[0]), label, fontsize=10)
    plt.yticks(np.arange(first_layer.shape[1]), np.arange(first_layer.shape[1] )+1, fontsize=10)
    plt.ylabel('Neuron index', fontsize=10)
    plt.grid()
    plt.show()
    
    
    
def plot_rnn_keras(model):
    def nodes(model):
        lim_max = int(len(model)/4)
        lim_min = -int(len(model)/4)
        d = (lim_max - lim_min)/4

        x_aux = np.linspace(0, 1, num=int(len(model)/2)+1)
        y_axis = []
        x_axis = []
        for idx, i in enumerate(range(0, len(model), 2)):
            y_aux = np.linspace(0, 1, model[i].shape[0])
            y_axis.append(y_aux)

            x_axis.append( np.repeat(lim_min + idx*d, y_aux.shape[0]) )

        y_axis.append(np.array([0.5]))
        x_axis.append( np.array([lim_max]) )
        return np.array(x_axis, dtype=object), np.array(y_axis, dtype=object)

    x_axis, y_axis = nodes(model.get_weights())
    plt.figure(figsize=(10, 8))
    plt.axis('off')
    for i in range(x_axis.shape[0]):
        layer_idx = i*2
        plt.plot(x_axis[i], y_axis[i], 'ko', fillstyle='none', markersize=15)

        if layer_idx < len(model.get_weights()):
            layer = model.get_weights()[layer_idx]
            node_max = abs(layer).max()

            x_from = x_axis[i][0]
            x_to = x_axis[i+1][0]
            for i_from, y_from in enumerate(y_axis[i]):
                for i_to, y_to in enumerate(y_axis[i+1]):
                    pass
                    plt.plot([x_from, x_to], [y_from, y_to], color='black',
                             linestyle='dashed', alpha= abs(layer[i_from][i_to])/node_max)

    plt.show()