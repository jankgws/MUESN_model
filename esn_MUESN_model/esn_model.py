import numpy as np
import networkx as nx
import time


# 恒等写像
def identity(x):
    return x

def fm(x):
    x = np.piecewise(x, [x <= 0, x > 0], [-0.5, 0.5])
    return x

class Input:
    def __init__(self, N_u, N_x, input_scale, seed=0):


        np.random.seed(seed=seed)

        #別の入力重み生成法
        '''
        mask = np.random.rand(N_x, N_u)
        mask[mask < 0.7] = 0
        mat = np.random.randint(0, 2, (N_x, N_u)) * 2 - 1
        self.W_in = np.multiply(mat, mask)
        '''
        

        self.W_in = np.random.uniform(-input_scale, input_scale, (N_x, N_u))*input_scale

        #print("self.W_in.shape",self.W_in.shape)
        
    def __call__(self, u):
        #np.set_printoptions(threshold=np.inf)
        #print(u)
        #print(self.W_in)
        #print("self.W_in ,u", self.W_in.shape, u.shape)
        return np.dot(self.W_in, u)

class Mem:
    def __init__(self, N_x, N_m, fb_scale, seed=0):
        
        np.random.seed(seed=seed)
        self.Wmem = np.random.uniform(-fb_scale, fb_scale, (N_x, N_m))

    def __call__(self, m):
        return np.dot(self.Wmem, m)

    def setMemweight(self, Wmem_opt):
        self.setWmem = Wmem_opt
    
    def mem(self, u, x, m):
        return fm(np.dot(self.setWmem, np.concatenate([u, x, m], axis = 0)))
        

class Reservoir:
    def __init__(self, N_x, spectral_radius, density, activation_func, leaking_rate, seed=0):
        self.seed = seed
        self.W_r = self.make_weight_conection(N_x, density, spectral_radius)
        self.x = np.zeros(N_x)  # リザバー状態ベクトルの初期化
        self.activation_func = activation_func
        self.alpha = leaking_rate

    def make_weight_conection(self, N_x, density, spectral_radius):
        np.random.seed(self.seed)

        #'''
        m = int(N_x*(N_x-1)*density/2)
        G = nx.gnm_random_graph(N_x, m, self.seed)

        connection = nx.to_numpy_matrix(G)
        W_r = np.array(connection)

        rec_scale = 1.0
        W_r *= np.random.uniform(-rec_scale, rec_scale, (N_x, N_x))
        #'''

        #別のリザバー重み生成法
        '''
        mask = np.random.rand(N_x,N_x)
        mask[mask < 0.1] = 0
        mat = np.random.normal(0, 1, (N_x,N_x))
        W_r = np.multiply(mat, mask)
        '''

        eigv_list = np.linalg.eig(W_r)[0]
        sp_radius = np.max(np.abs(eigv_list))

        W_r *= spectral_radius / sp_radius

        #print(W_r.shape)

        return W_r

    def __call__(self, x_in):

        self.x = (1.0 - self.alpha) * self.x \
                 + self.alpha * self.activation_func(np.dot(self.W_r, self.x) \
                 + x_in)
        #print(self.x.shape)
        return self.x


class Output:
# 出力結合重み行列の初期化
    def __init__(self, N_x, N_y, seed=0):

        # 正規分布に従う乱数
        np.random.seed(seed=seed)
        self.Wout = np.random.normal(size=(N_y, N_x))

    # 出力結合重み行列による重みづけ
    def __call__(self,flag, u, x):

        if flag is not None:
            return np.dot(self.Wout, np.concatenate([u, x]))
        else:
            return np.dot(self.Wout, x)

    # 学習済みの出力結合重み行列を設定
    def setweight(self, Wout_opt):
        self.Wout = Wout_opt

class Feedback:
# フィードバック結合重み行列の初期化
    def __init__(self, N_y, N_x, fb_scale, seed=0):
        # 一様分布に従う乱数
        np.random.seed(seed=seed)
        self.W_fb = np.random.uniform(-fb_scale, fb_scale, (N_x, N_y))

    # フィードバック結合重み行列による重みづけ
    def __call__(self, y):
        return np.dot(self.W_fb, y)

class Wmem_optimizer:

    def get_Mem_opt(self, U, X, M):
        H = np.concatenate([U, X, M], axis = 1)
        Wmem_opt = np.dot(np.linalg.pinv(H),M).T
        #print(self.D_XT.shape, X_pseudo_inv.shape)
        return Wmem_opt

class Wout_optimizer:

    def get_Wout_opt(self, U, X, D):
        G = np.concatenate([U, X], axis = 1)
        Wout_opt = np.dot(np.linalg.pinv(G), D).T

        return Wout_opt


class Ridge:
    def __init__(self, N_x, N_y, ridge_rate):
        self.beta = ridge_rate
        self.X_XT = np.zeros((N_x, N_x))
        self.D_XT = np.zeros((N_y, N_x))
        self.N_x = N_x

    # 学習用の行列の更新
    def __call__(self, d, x):
        #print(d.shape,x.shape)
        x = np.reshape(x, (-1, 1))
        d = np.reshape(d, (-1, 1))
        self.X_XT += np.dot(x, x.T)
        self.D_XT += np.dot(d, x.T)

    # Woutの最適解（近似解）の導出
    def get_Wout_opt(self):
        X_pseudo_inv = np.linalg.inv(self.X_XT \
                                    + self.beta*np.identity(self.N_x))
        Wout_opt = np.dot(self.D_XT, X_pseudo_inv)
        #print(self.D_XT.shape, X_pseudo_inv.shape)
        return Wout_opt

class ESN:
    def __init__(self, N_u, N_y, N_x, N_m, density=0.05, input_scale=1.0,
                 spectral_radius=0.95, activation_func=np.tanh, mem_fb_scale = None, fb_scale = None,
                 seed = 0, fb_seed=0, noise_level = None, leaking_rate=1.0,
                 output_func=identity, inv_output_func=identity):

        self.Input = Input(N_u, N_x, input_scale, seed)
        self.Reservoir = Reservoir(N_x, density, spectral_radius, activation_func, 
                                    leaking_rate, seed)
        self.Output = Output(N_x, N_y)
        self.N_u = N_u
        self.N_y = N_y
        self.N_x = N_x
        self.N_m =N_m
        self.y_prev = np.zeros(N_y)
        self.output_func = output_func
        self.inv_output_func = inv_output_func

        # 出力層からのリザバーへのフィードバックの有無
        if fb_scale is None:
            self.Feedback = None
        else:
            self.Feedback = Feedback(N_y, N_x, fb_scale, fb_seed)
        
        if mem_fb_scale is None:
            self.Mem  = None
            self.Wb = None
        else:
            self.Mem = Mem(N_x, N_m, mem_fb_scale)
            self.Wb = np.random.uniform(-mem_fb_scale, mem_fb_scale, (N_x, N_m))

        # リザバーの状態更新おけるノイズの有無
        if noise_level is None:
            self.noise = None
        else:
            np.random.seed(seed=0)
            self.noise = np.random.uniform(-noise_level, noise_level, 
                                           (self.N_x, 1))

# バッチ学習
    def train_Wmem(self, U_set, M_set, optimizer):
        '''
        最初の学習ステージで、リザーバとWMユニットの重み（Wmem）が計算される。

        return
            WM x (K + N + WM) の行列: Wmem
        '''
        U = np.concatenate(U_set)
        M = np.concatenate(M_set)
        train_len = len(U)

        X = np.empty((train_len, self.N_x))

        for n in range(train_len):
            x_in = self.Input(U[n])

            m = M[n]
            m = self.inv_output_func(m)

            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                x_in += x_back
            
            if self.Mem is not None :
                x_mem = self.Mem(m)
                x_in += x_mem

            x = self.Reservoir(x_in)
            X[n] = x
            
        self.Mem.setMemweight(optimizer.get_Mem_opt(U, X, M))
            

    def train_Wout(self, U_set, M_set, D_set, optimizer, trans_len = None):
        '''
        U: 教師データの入力, データ長×N_u
        D: 教師データの出力, データ長×N_y
        optimizer: 学習器
        trans_len: 過渡期の長さ
        return: 学習前のモデル出力, データ長×N_y
        '''
        U = np.concatenate(U_set)
        M = np.concatenate(M_set)
        D = np.concatenate(D_set)
        #print(U.shape,D.shape)
        train_len = len(U)
        if trans_len is None:
            trans_len = 0
        Y = []
        x = np.zeros(self.N_x)
        x_mem = np.ones(self.N_m)
        X = np.empty((train_len, self.N_x))

        # 時間発展
        for n in range(train_len):
            #print(U[n])
            #print(train_len)
            x_in = self.Input(U[n])

            m = M[n]
            m = self.inv_output_func(m)

            # フィードバック結合
            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                x_in += x_back

            # ノイズ
            if self.noise is not None:
                x_in += self.noise
            
            if self.Mem is not None :
                x_mem = self.Mem.mem(U[n], x, x_mem)
                x_in += np.dot(self.Wb, x_mem)

            # リザバー状態ベクトル
            x = self.Reservoir(x_in)
            X[n] = x

            # 目標値
            d = D[n]
            d = self.inv_output_func(d)
            if self.Mem is None :
                if n > trans_len:  # 過渡期を過ぎたら
                    optimizer(d, x)

        # 学習済みの出力結合重み行列を設定
        '''
        if self.Mem is not None:
            self.Output.setweight(optimizer.get_Wout_opt(U, X, D))
        else:
            self.Output.setweight(optimizer.get_Wout_opt())
        '''
        self.Output.setweight(optimizer.get_Wout_opt())

# バッチ学習後の予測
    def predict(self, U):
        test_len = len(U)
        #np.set_printoptions(threshold=np.inf)
        #print(U)
        #print("test_len",test_len)
        #print(U.shape)
        X_pred = []
        Y_pred = []

        x = np.zeros(self.N_x)
        x_mem = np.ones(self.N_m)
        flag = self.Mem

        # 時間発展
        for n in range(test_len):
            x_in = self.Input(U[n])

            # フィードバック結合
            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                x_in += x_back

            if self.Mem is not None:
                x_mem = self.Mem.mem(U[n], x, x_mem)
                x_in += np.dot(self.Wb, x_mem)

            # リザバー状態ベクトル
            x = self.Reservoir(x_in)
            X_pred.append(x)

            y_pred = self.Output(flag, U[n],x)

            Y_pred.append(self.output_func(y_pred))
            self.y_prev = y_pred

        # モデル出力（学習後）
        return np.array(Y_pred), X_pred

if __name__ == '__main__':
    pass
        