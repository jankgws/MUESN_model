import numpy as np
import networkx as nx

def generate_reservoir_weights(N_x, spectral_radius=None, density=None, seed=None, randomize_seed_afterwards=False, verbose=False):

    if seed is not None:
        np.random.seed(seed)
    
    m = int(N_x*(N_x-1)*density/2)
    G = nx.gnm_random_graph(N_x, m, seed)

    connection = nx.to_numpy_matrix(G)
    W_r = np.array(connection)

    rec_scale = 1.0
    W_r *= np.random.uniform(-rec_scale, rec_scale, (N_x, N_x))

    '''
    mask = 1*(np.random.rand((N_x,N_x)))
    mask[mask < 0.1] = 0
    mat = np.random.normal(0, 1, (N_x,N_x))
    W_r = np.multiply(mat, mask)
    '''

    eigv_list = np.linalg.eig(W_r)[0]
    sp_radius = np.max(np.abs(eigv_list))

    if verbose:
        print ("Spectra radius of generated matrix before applying another spectral radius: "+sp_radius)
    if spectral_radius is not None:
        W_r *= spectral_radius / sp_radius

    if randomize_seed_afterwards:
        """ redifine randomly the seed in order to not fix the seed also for other methods that are using numpy.random methods.
            numpy.randomメソッドを使用している他のメソッドでもseedを固定しないように、ランダムにseedをredifineする。
        """
        import time
        np.random.seed(int(time.time()*10**6))
    return W_r

def generate_input_weights(N_x, N_u, input_scale=None, seed=None, randomize_seed_afterwards=False, verbose=False):
    """
    Reservoirの入力接続に使用されるウェイト行列を生成するメソッドです。
    """
    if seed is not None:
        np.random.seed(seed)
    '''
    mask = 1*(mdp.numx_rand.random((nbr_neuron, dim_input))<proba)
    mat = mdp.numx.random.randint(0, 2, (nbr_neuron, dim_input)) * 2 - 1
    W_in = mdp.numx.multiply(mat, mask)
    '''

    W_in = np.random.uniform(-input_scale, input_scale, (N_x, N_u))

    if input_scale is not None:
        W_in = input_scale * W_in
    if randomize_seed_afterwards:
        """ redifine randomly the seed in order to not fix the seed also for other methods that are using numpy.random methods.
        """
        import time
        np.random.seed(int(time.time()*10**6))
    return W_in