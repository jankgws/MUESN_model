import numpy as np

def check_signal_dimensions(input_signal, target_signal):
    if input_signal.shape != target_signal.shape:
        raise RuntimeError("Input shape (%s) and target_signal shape (%s) should be the same."% (input_signal.shape, target_signal.shape))
    
def keep_max_for_each_time_step_with_default(input_signal, default_min_value=-1.0):
    # get the maximum for each line (= each time step)
    m_arr = np.max(input_signal, axis=1)
    m_arr = np.atleast_2d(m_arr).T
    m_mat = np.concatenate([m_arr for _ in range(input_signal.shape[1])],axis=1)
    # keep only the maximum in each line / for each time step, rest is 0
    return (input_signal >= m_mat)*input_signal + (input_signal < m_mat)*default_min_value    
    
def threshold_and_take_max_before_error(input_signal, target_signal, error_measure, thresh, default_min_value=-1.0):
    """
    まず各行の最大値のみを保持する（つまり各タイムステップの最大値を保持する）。
    次に input_signal と target_signal に閾値を適用する。
    最後に error_measure 関数を用いて誤差を求める。
    閾値は、'thresh' が指定されない限り、target_signal の最大値と最小値の平均として推定される。
    """
    check_signal_dimensions(input_signal, target_signal)
    
    # check if default_min_value is coherent with the threshold
    if default_min_value >= thresh:
        raise Exception('the default value applied after the max is taken is equal or superior to the threshold.')
    
    if thresh == None:
        thresh = (max(target_signal) + min(target_signal)) / 2.
    
    input_signal_max = keep_max_for_each_time_step_with_default(input_signal, default_min_value=default_min_value)
    return error_measure(input_signal_max > thresh, target_signal > thresh)

def check_signal_dimensions(input_signal, target_signal):
    if input_signal.shape != target_signal.shape:
        raise RuntimeError("Input shape (%s) and target_signal shape (%s) should be the same."% (input_signal.shape, target_signal.shape))

def loss_01(input_signal, target_signal):
    '''
    ゼロワンロス関数を計算する 
    input_signalとtarget_signalが等しくないタイムステップの割合を返す。
    '''
    check_signal_dimensions(input_signal, target_signal)
    
    return np.mean(np.any(input_signal!= target_signal, 1))

def threshold_before_error(input_signal, target_signal, error_measure=loss_01, thresh=None):
    '''
    まず input_signal と target_signal に閾値を適用し、 error_measure 関数を用いて誤差を求める。
    閾値は、'thresh' が指定されない限り、 target_signal の最大値と最小値の平均値として推定される
    '''
    check_signal_dimensions(input_signal, target_signal)
    
    if thresh == None:
        thresh = (max(target_signal) + min(target_signal)) / 2
    return error_measure(input_signal > thresh, target_signal > thresh)

class ThematicRoleError():
    """
    Specific language error: measure defined for a special language task on thematic role assignment.
    テーマ別役割分担に関する特別な言語タスクのために定義されたメジャー
    """
    def __init__(self, d_io, error_measure=loss_01, threshold=0,
                 verbose=False):
        """
        入力です。
            - d_io: 入出力に関するパラメータと情報を収集する辞書
            - error_measure: self.time_step_sliceで定義された区間での誤差を計算するために使用されるメソッド
            - threshold: error_measureがバイナリ応答を識別するために使用する閾値。
        """
        self.error_measure = error_measure
        self.threshold = threshold
        self.lt_NVassoc = [('N1','V1'), ('N1','V2'), ('N2','V1'), ('N2','V2'),
                           ('N3','V1'), ('N3','V2'), ('N4','V1'), ('N4','V2')]
        self.d_io_current = d_io
        self.d_io = d_io.copy()
        self.verbose = verbose
        self._check_output_version()
        self.__initialize_error_algorithm()

    def __initialize_error_algorithm(self):
        """
        Definning:
            time_step_slice: start and stop slice for evaluate output signal
            max_answers: maximum number of active outputs
        定義する。
            time_step_slice: 出力信号の評価のための開始と停止のスライス
            max_answers: アクティブな出力の最大数
        """
        
        self.time_step_slice = slice(-1+self.d_io_current['full_time'],self.d_io_current['full_time'])#ここで範囲を決めるslice(start=339,stop=340)
        self.max_answers = self._get_max_answers()
    
    def _check_change_in_d_io(self, just_warning=False):
        if self.d_io_current != self.d_io:
            if just_warning:
                raise Warning("d_io (dictionary of input/output) has changed since the initialization of object 'thematic_role_error'.")
                self.__check_full_time()
                self.__check_output_version()
            else:
                raise Exception("d_io (dictionary of input/output) has changed since the initialization of object 'thematic_role_error'.")

    def _check_output_version(self):
        if not(('l_output') in self.d_io):
            print("!!! WARNING: io dictionary has no 'l_output' entry, version of output could not be checked. !!!")
        elif self.d_io['l_output'] != ['N1-A1','N1-O1','N1-R1','N1-A2','N1-O2','N1-R2','N2-A1','N2-O1','N2-R1','N2-A2','N2-O2','N2-R2','N3-A1','N3-O1','N3-R1','N3-A2','N3-O2','N3-R2','N4-A1','N4-O1','N4-R1','N4-A2','N4-O2','N4-R2']:
            raise Exception("Output coding is not the same as expected")

    def _get_max_answers(self):
        """
        Return the maximal number of answer for one sentence.
            It corresponds to the maximal length of elements in 'l_teacher'.
        !!! Warning: to be accurate, 'l_teacher' needs to corresponds to the subset selected, and not the full set of data.
        1つの文に対する答えの最大数を返す。
            これは'l_teacher'の要素の最大長に相当する。
        !!! 警告: 正確を期すために、'l_teacher' はデータのフルセットではなく、選択されたサブセットに対応する必要がある。
        """
        return max([len(x) for x in self.d_io['l_teacher']])

    def _get_NVassoc_sliced(self, input_signal, target_signal, verbose=False):
        """
        出力する。
            (NVassoc_admiting_anwser, NVassoc_not_present_in_sent)
            このタプルの各要素はリストである。リストの各要素は3つのタプルである。
                - 1番目：タプルのリスト 'self.lt_NVassoc' に含まれる名詞-動詞の関連付けのインデックス
                - 2番目: error_measureが使用するinput_signalのサブ行列(sub-numpyarray)
                - 3rd: error_measure で使用される teacher_signal の sub-matrix (sub-numpyarray)
                
        警告: このメソッドは、出力信号の特定のコーディング方法に依存しています。
            このコーディングが変更された場合、このメソッドの大部分を再コーディングしなければならないかもしれない。
                
        表記法。
            Nva。名詞-動詞連合
        
        NB: アルゴリズムが対処しなければならない問題が何であるかについて、いくつかの前提条件がある。
            文中にどのNVa-s（名詞-動詞連合）が存在するかを推論する必要がある。
            -- これは、Open Class Words (~Nouns) がいくつあって、いくつの意味があるのかを推測すること
             を推測することです。
            この方法では、target_signalから推論することで行う（入力コーパスがないため）。
        """
        ## 名詞-動詞の関連付け（すなわち、与えられた動詞に対する与えられた名詞の完全なAOR）。
        ## 名詞と動詞が同時に存在するNVassocである。
        ## 名詞と動詞を1つの教師として同時に持つNVassocです。
        ## 可能な異なるNVassocはself.lt_NVassocで与えられます。
        if verbose:
            print("<<< Beginning method _get_NVassoc_sliced():")
            print("self.time_step_slice", self.time_step_slice)
        
        ## このバージョンのエラー計算が、出力のコード化された方法で問題ないか確認する
        self._check_output_version()
        
        ## creating NVassoc
        NVassoc_admiting_anwser = []
        NVassoc_not_admiting_answer = []
        l_N_V_present_in_sentence = []
        ##文中に存在する名詞、動詞、名詞と動詞の結合を発見する。
        for idx in range(0,21+1,3):
            NVindex = int(idx/3)
            current_NVassoc = (NVindex, input_signal[self.time_step_slice, idx:idx+3], target_signal[self.time_step_slice, idx:idx+3])
#            if mdp.numx.any(target_signal[self.time_step_slice, idx:idx+3]):
            if np.any(target_signal[self.time_step_slice, idx:idx+3] > self.threshold): # the non-1 signal could be 0 or -1 (so np.any() is not sufficient)
                print("using mdp.numx.any")
                ## add the current NVassoc to the list
                NVassoc_admiting_anwser.append(current_NVassoc)
                ## 文中に存在する名詞と動詞のリストに名詞と動詞を追加する（後で使用されます）
                    # 重複が発生しますが、問題ではありません。
                l_N_V_present_in_sentence.extend(self.lt_NVassoc[NVindex])
            else:
                NVassoc_not_admiting_answer.append(current_NVassoc)
        if verbose:
            print("target_signal.shape", target_signal.shape)
            print("self.time_step_slice", self.time_step_slice)
            print("target_signal[self.time_step_slice, idx:idx+3].shape", target_signal[self.time_step_slice, idx:idx+3].shape)
            print(" NVassociations admiting anwser: ")
            for NVi in NVassoc_admiting_anwser:
                print("  - "+str(self.lt_NVassoc[NVi[0]])+" _ "+str(NVi))
            print(" NVassociations not admiting anwser: ")   
            for NVi in NVassoc_not_admiting_answer:
                print("  - "+str(self.lt_NVassoc[NVi[0]])+" _ "+str(NVi))
        
        ## NVassoc_not_admiting_answer が文中に存在するが、そのリストを作成する。
        NVassoc_not_admiting_answer_but_present = []
        NVassoc_not_present_in_sent = []
        # for each NVa in NVassoc_not_admiting_answer,
        for NVi in NVassoc_not_admiting_answer:
            # その名詞（つまりself.lt_NVassoc[NVi[0][0]）またはその動詞（つまりself.lt_NVassoc[NVi[0][1]）が文中に存在しないとき
            if l_N_V_present_in_sentence.count(self.lt_NVassoc[NVi[0]][0])==0 \
                or l_N_V_present_in_sentence.count(self.lt_NVassoc[NVi[0]][1])==0: 
                # put it in a new list containing the NVa not present in setence
                NVassoc_not_present_in_sent.append(NVi)
            #if N and V are present in the NVa
            else: 
                #add it to the new list
                NVassoc_not_admiting_answer_but_present.append(NVi)
        if verbose:
            print(" NVassociations not admiting anwser, but present: ")
            for NVi in NVassoc_not_admiting_answer_but_present:
                print(" - "+str(self.lt_NVassoc[NVi[0]])+" _ "+str(NVi))
            print(" NVassociations not admiting anwser and not present: ")   
            for NVi in NVassoc_not_present_in_sent:
                print("  - "+str(self.lt_NVassoc[NVi[0]])+" _ "+str(NVi))

        if (len(NVassoc_admiting_anwser)+len(NVassoc_not_admiting_answer_but_present)+len(NVassoc_not_present_in_sent)) != len(self.lt_NVassoc):
            raise Exception("The number of Noun-Verb association is not correct. Should be "+str(len(self.lt_NVassoc)))
        
        if verbose:
            print(">>> End of method _get_NVassoc_sliced():")
        
        return (NVassoc_admiting_anwser, NVassoc_not_admiting_answer_but_present, NVassoc_not_present_in_sent)

    def compute_error(self, input_signal, target_signal, verbose=False):
        """
        入力です。
            input_signal: 出力読み出しアクティビティ
            target_signal: 教師出力(教師学習用)
        出力。
            (意味の誤りの平均, 文の誤りの平均,
                誤った名詞・動作の数, 適切な名詞・動作の数, 正しい名詞のリスト, 間違った名詞のリスト)
        2行目は、デフォルトモードでは使用されない結果を収集します。この情報は、エラーについてより詳しく知るために使用します。
        """
        check_signal_dimensions(input_signal, target_signal)
        self._check_change_in_d_io()
        
        ## initialization
        perf_asso_adm_answ = [] #performance of NVa admiting answer
        (NVassoc_admiting_anwser, NVassoc_not_admiting_answer_but_present, NVassoc_not_present_in_sent) = \
            self._get_NVassoc_sliced(input_signal, target_signal, verbose=False)
        NVa_correct = []
        NVa_erroneous = []
        
        ## NVaが答えを認める際の計算エラーと不可能な状態
        for NVi in NVassoc_admiting_anwser:
            ## 3信号のAORで同時に正解が出た場合の割合の評価
            err_answer = threshold_and_take_max_before_error(input_signal=NVi[1],
                                                           target_signal=NVi[2],
                                                           error_measure=self.error_measure,
                                                           thresh=self.threshold)
            perf_asso_adm_answ.append(1 - err_answer)
            if err_answer > 0:
                NVa_erroneous.append(NVi[0])
            else:
                NVa_correct.append(NVi[0])
                
            if verbose:
                print("NVassoc_admiting_anwser: "+str(self.lt_NVassoc[NVi[0]])+" _ "+str(NVi))
                print("only max of NVa", keep_max_for_each_time_step_with_default(NVi[1]))
                print("err_answer="+str(err_answer))
        
        ## 答えを認めないが文中に存在するNVaの計算エラーと不可能な状態
        perf_asso_not_adm_answ_p = [] #NVaの性能は、答えを認めないが、存在する。
        for NVi in NVassoc_not_admiting_answer_but_present:
            err_answer = threshold_and_take_max_before_error(input_signal=NVi[1],
                                                       target_signal=NVi[2],
                                                       error_measure=self.error_measure,
                                                       thresh=self.threshold)
            perf_asso_not_adm_answ_p.append(1 - err_answer)
            if err_answer > 0:
                NVa_erroneous.append(NVi[0])
            else:
                NVa_correct.append(NVi[0])
            if verbose:
                print("NVassoc_not_admiting_answer_but_present: "+str(self.lt_NVassoc[NVi[0]])+" _ "+str(NVi))
                print("only max of NVa", keep_max_for_each_time_step_with_default(NVi[1]))                
                print("print err_answer="+str(err_answer))
        
        ## Compute means
        if perf_asso_adm_answ != []:
            if perf_asso_not_adm_answ_p != []:
                aa = perf_asso_adm_answ
                naap = perf_asso_not_adm_answ_p
                perf_asso_present = (len(aa)*np.mean(aa) + len(naap)*np.mean(naap)) / float((len(aa) + len(naap)))
            else:
                perf_asso_present = np.mean(perf_asso_adm_answ)
        else:
            raise Exception("There is no answer for this sentence.")

        # (文中に存在するNVaについて）該当するNVaがすべて正しい場合の割合を計算する。
        all_output_signal = []
        all_target_signal = []
        for NVi in NVassoc_admiting_anwser:
            all_output_signal.append(keep_max_for_each_time_step_with_default(NVi[1]))
            all_target_signal.append(NVi[2])
        for NVi in NVassoc_not_admiting_answer_but_present:
            all_output_signal.append(keep_max_for_each_time_step_with_default(NVi[1]))
            all_target_signal.append(NVi[2])
        global_out_arr = np.concatenate(all_output_signal, axis=1)#numpy行列を結合する。axis=1のときは横方向
        global_target_arr = np.concatenate(all_target_signal, axis=1)
        global_err_answer = threshold_before_error(input_signal=global_out_arr,
                                                   target_signal=global_target_arr,
                                                   error_measure=self.error_measure,
                                                   thresh=self.threshold)
        
        ##補足計算 (デフォルトのプログラムでは使用しない)
        ## 各動詞に関連する SW (意味語) 出力のうち、誤っているものの数を計算する
        # i.e. 誤ったNV-assocの数
        total_nr_of_pertinent_SW = len(NVassoc_admiting_anwser) + len(NVassoc_not_admiting_answer_but_present)
        nr_of_erroneous_SW = int(round(total_nr_of_pertinent_SW * (1-perf_asso_present)))
        if total_nr_of_pertinent_SW != (len(NVa_erroneous)+len(NVa_correct)):
            raise Exception("Incoherent total_nr_of_pertinent_SW. total_nr_of_pertinent_SW"+str(total_nr_of_pertinent_SW)+ \
                "\n NVa_correct="+str(NVa_correct)+ \
                "\n NVa_erroneous="+str(NVa_erroneous))
        if nr_of_erroneous_SW != len(NVa_erroneous):
            raise Exception("Incoherent nr_of_erroneous_SW." \
                +"\nnr_of_erroneous_SW="+str(nr_of_erroneous_SW)+ \
                "\n NVa_correct="+str(NVa_correct)+ \
                "\n len(NVa_erroneous)="+str(len(NVa_erroneous)))
        if verbose:
            print("all_output_signal", all_output_signal)
            print("global activity (only max was kept)", global_out_arr)
            print("global_out_arr.shape", global_out_arr.shape)
            print("global_err_answer", global_err_answer)
            print("total_nr_of_pertinent_SW", total_nr_of_pertinent_SW)
            print("nr_of_erroneous_SW", nr_of_erroneous_SW)
            print("SW level error: ", (1-perf_asso_present))
            
        
        return (1 - perf_asso_present, global_err_answer,
                nr_of_erroneous_SW, total_nr_of_pertinent_SW, NVa_correct, NVa_erroneous)

if __name__ == '__main__':
    pass