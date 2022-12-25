from operator import index
from sre_parse import State
import numpy as np
from sklearn.decomposition import PCA

import corpus_generator
import dif_measure
import plot_lib
from esn_model import ESN, Ridge, Wmem_optimizer, Wout_optimizer
from error import ThematicRoleError

def co_error(error_obj, _states_out, _curr_teachers, _subset, verbose=False):
    all_meaning_err = []
    all_sentence_err = []
    for i in range(len(_curr_teachers)):
        #print(_states_out[i].shape, _curr_teachers[i].shape)
        full_info_err = error_obj.compute_error(input_signal=_states_out[i],
                                            target_signal=_curr_teachers[i],
                                            verbose=verbose)
        (err_avg_on_signals, global_err_answer, nr_of_erroneous_SW, total_nr_of_pertinent_SW, NVa_correct, NVa_erroneous) = full_info_err
        all_meaning_err.append(err_avg_on_signals)
        all_sentence_err.append(global_err_answer)
    mean_me = np.mean(all_meaning_err)  #平均を求める
    std_me = np.std(all_meaning_err)  #標準偏差を求める
    median_me = np.median(all_meaning_err)  #中央値を求める
    mean_se = np.mean(all_sentence_err)
    std_se = np.std(all_sentence_err)
    median_se = np.median(all_sentence_err)
    return (mean_me, std_me, median_me, mean_se, std_se, median_se)


def simulation(N_x=100, spectral_radius=0.95, density=0.05, tau=6, act_time=20, subset=range(15,41),
               in_scal=0.75, ridge=10**-9, seed=0,
                Wmem_module = True, root_file_name=None, comp_diff_states=True, PCA_output = True, plot_output=True, verbose=False):

    sentence = {}
    sentence['subset'] = subset  
    sentence['act_time'] = act_time
    sentence['suppl_pause_at_the_end'] = 0
    sentence['initial_pause'] = True
    (sentence['l_input'], sentence['l_data']) = corpus_generator.get_corpus() 
    (sentence['l_output'], sentence['l_teacher']) = corpus_generator.get_coded_meaning()

    inputs = corpus_generator.generate_stim_input(d_io=sentence, verbose=verbose)
    MU_teacher = corpus_generator.generate_teacher_output(d_io=sentence, verbose=verbose)
    teacher = corpus_generator.generate_next_sentence(inputs)
    sentence['N_u'] = inputs[0].shape[1]

    if verbose: 
        print ("len(inputs)", len(inputs))
        print ("inputs[0].shape", inputs[0].shape)
        print ("len(teacher)", len(teacher))
        print ("sentence['subset']", sentence['subset'])

    N_u = sentence['N_u']
    N_m = MU_teacher[0].shape[1]
    N_y = teacher[0].shape[1]

    esn_model = ESN(N_u, N_y, N_x, N_m, density=density, input_scale=in_scal,
                 spectral_radius=spectral_radius, activation_func=np.tanh, mem_fb_scale= 0.4,fb_scale = None,
                 seed = seed, fb_seed=seed, noise_level = None, leaking_rate=(1./(tau*sentence['act_time'])))

    train_indices = [range(len(sentence['subset']))]  #subset=(15,41)であればlen(15,41)=26なので、[range(len(sentence['subset']))]=[range(0, 26)]となる
    test_indices = train_indices
    #print("train_indices",len(train_indices))

    err_obj = ThematicRoleError(d_io=sentence)


    for i in range(len(train_indices)):
        inputs_train = [inputs[x] for x in train_indices[i]]
        mem_teachers_train = [MU_teacher[x] for x in train_indices[i]]
        teachers_train = [teacher[x] for x in train_indices[i]]
        inputs_test = [inputs[x] for x in test_indices[i]]
        mem_teachers_test = [MU_teacher[x] for x in train_indices[i]]
        teachers_test = [teacher[x] for x in test_indices[i]]

        #np.set_printoptions(threshold=np.inf)
        #print(inputs_train)
        #print(teachers_train)
        #print(inputs_test)
        #print(teachers_test)
        #print(len(inputs_test))#3

        state_out_test = len(inputs_test)*[None]
        #U = len(inputs_test)*[None]
        #X = len(inputs_test)*[None]

        #for idx_train_out in range(len(inputs_test)):
        #    esn_model.train(inputs_train[idx_train_out], teachers_train[idx_train_out], Ridge(N_x, N_y, ridge))

        if Wmem_module:
            esn_model.train_Wmem(inputs_train, mem_teachers_train, Wmem_optimizer())
            esn_model.train_Wout(inputs_train, mem_teachers_train, teachers_train, Wout_optimizer())
        else:
            esn_model.train_Wout(inputs_train, mem_teachers_train, teachers_train, Ridge(N_x=N_x, N_y=N_y, ridge_rate=1e-4))
            
        for idx_test_out in range(len(inputs_test)):
            if idx_test_out == 0:
                (state_out_test[idx_test_out],X) = esn_model.predict(inputs_test[idx_test_out])
                U = inputs_test[idx_test_out]
            else:
                (state_out_test[idx_test_out],x) = esn_model.predict(inputs_test[idx_test_out])
                u = inputs_test[idx_test_out]
                U = np.append(U, u, axis = 0)
                X = np.append(X, x, axis = 0)
        
        #print(X)
        #print(X.shape)


        pca_X = PCA(n_components = 2)
        pca_X.fit(X)
        PC_X = pca_X.transform(X)

        #print(U)
        #print(U.shape)

        pca_U = PCA(n_components = 1)
        pca_U.fit(U)
        PC_U = pca_U.transform(U)

        #print(PC_U.shape)
        #print(PC_X.shape)

        np.set_printoptions(threshold=np.inf)
        #print(len(state_out_test))
        #print(state_out_test[0].shape)

        if PCA_output :
            plot_lib.plot_PCA(PC_X=PC_X, PC_U=PC_U, root_file_name=root_file_name,subtitle="PCA_test")

        if plot_output:
            plot_lib.plot_output(_outputs=state_out_test, d_io=sentence,
                save_pdf=True, root_file_name=root_file_name,subtitle="Predict_test")
        #'''
        

if __name__ == "__main__":

    import datetime

    date_now = datetime.datetime.now()
    file_name = date_now.strftime('%Y%m%d_%H%M%S')
    
    tau_list = list(range(1,10))
    act_time_list = list(range(10,30))
    N_x_list = list(range(50,2050,50))

    #for index in act_time_list:
    simulation(N_x=1000, spectral_radius=1.0, density=0.05, tau=4, act_time=20, subset=range(15,41),
                in_scal=0.75, ridge=10**-9, seed=0, root_file_name=file_name,PCA_output = False, plot_output= True)

    #print(mean_merr, std_me, mean_serr, std_se)
'''
    print(N_x_list,len(N_x_list))
    print(mean_merr,len(mean_merr))

    plt.bar(N_x_list, mean_merr, alpha=0.5)
    plt.ylabel("word mean err")
    plt.xlabel("N_x")
    plt.show()

'''