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
    plt.title(f'$R^2$={r2:.2f}, RMSE={np.sqrt(mse):.2f}, RMSRE: {RMSRE(data_true, data_pred, order=1):.2f} %,  RMSRE: {RMSRE(data_true, data_pred):.2f} %', fontsize=10)
    plt.grid(color='gray', linestyle='--', linewidth=0.25)
    
def plot_identity_keras(model, X_test, Y_test):
    """
    """
    Y_hat_test = model.predict(X_test)
    Y_hat_test = Y_hat_test.reshape(Y_hat_test.shape[0],)
    plot_identity_graphic(Y_test, Y_hat_test)


def plot_boxplot_input(data, data_labels, name=''):
    plt.figure(figsize=(6, 4))
    plt.boxplot(data)
    plt.xticks(np.arange(len(data_labels))+1, data_labels, rotation='45')
    plt.grid(':', linewidth=0.25)
    plt.title('Boxplot of the ' + name + ' variables')

def RMSRE(true_values, predict_values, order=2):
    """
    Evaluate metric
    """
    relativeSE = ((predict_values-true_values)*100/true_values)**order
    MrelativeSE = np.mean( relativeSE )
    if order == 1:
        return MrelativeSE
    elif order == 2:
        return np.sqrt(MrelativeSE)
    elif order == 3:
        return np.cbrt(MrelativeSE)
    return None



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
    
def evaluate(model, X_train, X_test, Y_train, Y_test):
    """
    
    """
    Y_hat_train = model.predict(X_train)
    Y_hat_train = Y_hat_train.reshape(Y_hat_train.shape[0],)
    Y_hat_test = model.predict(X_test)
    Y_hat_test = Y_hat_test.reshape(Y_hat_test.shape[0],)
    
    print(f'Train: {mse_f(Y_train, Y_hat_train):.4f}')
    print(f'Test: {mse_f(Y_test, Y_hat_test):.4f}')
    return

def load_model(path='500_dados.hdf5'):
    """
    
    """
    model = Sequential()
    model.add(Dense(25, input_dim=X_train.shape[1], activation='exponential'))
    model.add(Dense(11, activation='sigmoid', ))
    model.add(Dense(3, activation='exponential'))
    model.add(Dense(1))
    model.load_weights('500_dados.hdf5')
    return model