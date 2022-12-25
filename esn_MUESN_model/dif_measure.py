import numpy as np

def amount_of_change(states_out, verbose=False):
    #差分を含むベクトルを含む全リストの初期化（一段階前の比較）
    diff = [np.zeros_like(states_out[0])]*len(states_out)#np.zeros_likeは配列をすべて0に初期化する
    for idx_so in range(len(states_out)):
        so = states_out[idx_so]
        if verbose:
            print("so.shape", so.shape)
        so_diff_tmp = np.zeros_like(so) ## so_diff_tmp = mdp.numx.zeros(so.shape[0],so.shape[1])
        for current_time_step in range(1,so.shape[0]): # there is nothing to compute at step 0, so we skip it
            so_diff_tmp[current_time_step,:] = so[current_time_step,:] - so[current_time_step-1,:]
            if verbose:
                print("previous", so[current_time_step-1,:])
                print("actual", so[current_time_step,:])
                print("time step "+ str(current_time_step))
                print("diff", so_diff_tmp[current_time_step,:])
        diff[idx_so] = np.copy(so_diff_tmp) #so_diff_tmp[:] #np.copyはデータの要素、型をコピーする
    return diff

def sum_amount_of_change(diff, return_as_tuple=True, verbose=False):
    """
    入力:
        - diff: state_out の差分(n 個の時間ステップ後方)を格納したベクタのリスト.
            これは, amount_of_change() メソッドの出力である.
    """
    if return_as_tuple:
        sum_diff = [np.zeros_like(diff[0][:,0])]*len(diff)
        abs_sum_diff = [np.zeros_like(diff[0][:,0])]*len(diff)
        abs_max_diff = [np.zeros_like(diff[0][:,0])]*len(diff)
    else:
        s_diff = [np.zeros_like(diff[0][:,0:2])]*len(diff)
    for idx_diff in range(len(diff)):
        # summing the diff vector on the lines: each line has a single column afterwards
        if return_as_tuple:
            sum_diff[idx_diff] = diff[idx_diff].sum(axis=1)
            abs_sum_diff[idx_diff] = abs(diff[idx_diff]).sum(axis=1)
            abs_max_diff[idx_diff] = np.amax(abs(diff[idx_diff]), axis=1)
        else:
            s_diff[idx_diff] = np.concatenate((np.atleast_2d(diff[idx_diff].sum(axis=1)),
                np.atleast_2d(abs(diff[idx_diff]).sum(axis=1)),
                np.atleast_2d(np.amax(diff[idx_diff], axis=1))), axis=0).transpose()#np.atleast_2dは(要素)を少なくとも2次元で表示する
        if verbose:
            print("diff", diff)
            print("sum_diff", sum_diff)
            print("abs_sum_diff", abs_sum_diff)
            print("abs_max_diff", abs_max_diff)
    if return_as_tuple:
        return (sum_diff, abs_sum_diff, abs_max_diff)
    else:
        return s_diff

if __name__ == '__main__':
    pass   
