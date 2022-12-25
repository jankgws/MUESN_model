import numpy as np

def make_one_stim(l_input, sentence, act_time, suppl_pause_at_the_end, full_time, offset, initial_pause=True):
    """
    選択された文に対応するstim入力を返す。
    
    入力: 
        - l_input: 入力で与えられたすべての可能な単語のリスト. このリストの長さが入力の次元を与える.
        - full_time: 刺激のタイムステップの総数
        - offset: データ中の最大単語数と与えられた文の単語数との差を表す。
            オフセットを考慮する場合、オフセットにact_timeを乗じる（'pause'が偽か真かによって1倍か2倍になる）。
        
    変数
        - 文[i]: 現在の文のi+1番目の単語
        - l_input.index(sentence[i]): 入力刺激中の単語 'sentence[i]' のインデックスを表す。
    """
    # Initializations(初期設定)
    if initial_pause is True:
        j = 1
    else:
        j = 0
    stim = np.zeros((len(l_input), full_time)) # stimulus (returned value)
    # Generating the stimulus protocol while processing the sentence(文章を処理しながら刺激プロトコルを生成する)
    j = j + offset
    for i in range(len(sentence)):
        stim[l_input.index(sentence[i]), act_time*j:act_time*(j+1)] = np.ones((1,act_time)) 
        j = j + 1
    return stim.T

def make_one_teacher(l_output, AOR, act_time, full_time, suppl_pause_at_the_end, offset,
                     nr_words, initial_pause=True, start='end', verbose=False):
    """
    Returns the teacher outputs signal corresponding to the AOR (AOR: Agent-Object-Recipient) output selected corresponding to the same index sentence of get_info_stim.
        See method 'get_info_teacher_output()' for more details on the sentences.
    
    The output teacher is forced to one since a certain number of 'act_time' indicated by 'start'
    
    Modification of method _output_gen(l_output, AOR, act_time, full_time, pause, suppl_pause_at_the_end, initial_pause=True):
        in order to be able to set an arbitrary moment where to begin the output.
        if 'start' is set to 'end', this means that the output will be asked since the beginning of the last "element" of the sentence (an element could be a word or a sign of punctuation like a dot).
    
    Input:
        - AOR: output desired coded in the AOR fashion: it corresponds to the current line in 'l_teacher' obtained with the method get_info_teacher_output()
        - full_time: total number of time step of a teacher output signal
        - nr_words: indicates the number of words or word-like (comma, period, etc.)
        - start: if it's a number it indicates the position of the word where will start the teacher output signal (this number starts from 1, not from 0)
            if a decimal part is present, it indicates the relative position during the stimulus: e.g. 1.5 indicates that the signal will begin at the middle of the 1st word.
                If a fractional part exist it has to be taken into account, so it cannot be zero, that's why we take the upper closer integer.
            if the number is negative, this means that we consider the starting of the signal during the pause that is just after the word stimulus
                !!! -1.25 indicates that the signal will begin at the first quarter of the pause that is just after the 1st word:
                    the decimal part is interpreted separately from negativity (i.e. the fact that the number is negative)
        - offset: represents the difference between the maximum number of words in the data and the number of word of a given sentence.    
            when taking into account the offset, offset has to be multiplied by 'act_time'  
    """
    # Initializations
    if initial_pause is True:
        j = 1
    else:
        j = 0
    if start is None:
        st = 0
        fr = 0
    else:
        if start=='end': # checks if start is not a number
            st = int(nr_words)-1
            fr = 0
        elif -1<start<1: # if it is a number, check if it has a correct value
            raise Exception("argument 'start' cannot be between -1 and 1 (superior to -1 and inferior to 1). ")
        else:
            (fr, st) = np.modf(np.fabs(start)) # math.modf(x) returns the fractional and the integer part of x
            if verbose:
                print("nr_words:", nr_words)
                print("st:", st)
            if st > nr_words:
                raise Exception("The start point indicated for the output teacher is too large for the data: 'start' exceeded the total number of words. start="+str(start)+" ; nr_words="+str(nr_words))
            st = int(st-1) # start begins at 1 not at 0 like the index
            fr = int(np.ceil(act_time*fr)) # take the smallest integer value greater than or equal to (act_time*pc). If a fractional part exist it has to be taken into account, so it cannot be zero, that's why we take the upper closer integer.
    if start<0 and start is not None: 
        raise Warning("argument 'start' is negative. Information ignored, output teacher signal will start during the word.")
    teach = np.zeros((len(l_output), full_time)) # stimulus (returned value)
    off = offset
    if (act_time*(j+st+off)+fr) >= full_time:
        raise Warning("The output teacher is beginning to late: consequently the teacher output will be all zeros. act_time*(j+st+off)+fr)="+str(act_time*(j+st+off)+fr)+" ; full_time="+str(full_time))
    for i in range(len(AOR)):
        teach[l_output.index(AOR[i]), act_time*(j+st+off)+fr:full_time] = np.ones((1,full_time-(act_time*(j+st+off)+fr)))
    if verbose:
        print("nr_words:", nr_words, " _ start:",start, " _ offset:", offset, " _ st:",st , " _ fr:",fr ," _ j:", j, " _ off:", off)
        print("j+st+off=", str(j+st+off))
        print("act_time*(j+st+off)+fr: ", act_time*(j+st+off)+fr, " _ full_time:", full_time)
        print("ex of teacher output:", teach[l_output.index(AOR[i])], '\n')
    return teach.T

def generate_stim_input(d_io, verbose=False):
    # check if subset is too large for the data
    if len(d_io['subset']) > len(d_io['l_data']):
        s = "The length of the subset is too large. Input data has a lower size than the subset: the length of the subset is "+str(len(d_io['subset']))+" but the length of the input data is "+str(len(d_io['l_data']))+"."
        raise Exception(s)
    ## counting the number of words per line
    d_io['l_nr_word'] = [None]*len(d_io['l_data'])
    for i in range(len(d_io['l_data'])):
        d_io['l_nr_word'][i] = len(d_io['l_data'][i])
    d_io['mult'] = max(d_io['l_nr_word']) # max number of words in one sentence
    # compute full time of stimulus
    d_io['full_time'] = d_io['act_time']*d_io['mult'] + d_io['act_time']*(1*d_io['initial_pause']) + d_io['suppl_pause_at_the_end']# full time of stimulus
    if d_io['subset'] is None:
        d_io['subset'] = range(len(d_io['l_data']))
    if verbose:
        print("subset selected:"+str(d_io['subset']))
        print("l_input="+str(d_io['l_input']))
        print("full_time="+str(d_io['full_time']))
    stim_data = len(d_io['subset'])*[np.zeros((len(d_io['l_input']), d_io['full_time']))]
    l_offset = [d_io['mult']-x for x in d_io['l_nr_word']] #The offset represents the difference between the maximum number of words in the data and the number of word of a given sentence.
    idx_stim = 0
    for i in d_io['subset']:
        stim_data[idx_stim] = make_one_stim(l_input=d_io['l_input'], sentence=d_io['l_data'][i],
                                            act_time=d_io['act_time'], full_time=d_io['full_time'],
                                            suppl_pause_at_the_end=d_io['suppl_pause_at_the_end'],
                                            offset=l_offset[i], initial_pause=d_io['initial_pause'])
        idx_stim = idx_stim + 1
    d_io['l_offset'] = l_offset
    return stim_data

def generate_teacher_output(d_io, verbose=False):
    """ 
    Gives the hole teacher signal data set or a subset of the data set defined by the list 'subset' in dictionary d_io.
    The sentences are aligned to the right (so the beggining of the sentence depends on the length of maximal sentence)
    """
    teacher_output = len(d_io['subset'])*[np.zeros((len(d_io['l_output']),  d_io['full_time']))]
    idx_teach = 0
    for i in d_io['subset']:
        nr_words = d_io['l_nr_word'][i]
        teacher_output[idx_teach] = make_one_teacher(l_output=d_io['l_output'], AOR=d_io['l_teacher'][i],
                                    act_time=d_io['act_time'], full_time= d_io['full_time'],
                                    suppl_pause_at_the_end=d_io['suppl_pause_at_the_end'], nr_words=nr_words,
                                    start=1,  initial_pause=d_io['initial_pause'],
                                    offset=d_io['l_offset'][i])
        idx_teach = idx_teach + 1
    # Scale data from [0,1] to [-1,1]  
    for idx in range(len(teacher_output)):
            teacher_output[idx] = teacher_output[idx]*2 - 1
    return teacher_output

def generate_next_sentence(sentence):
    next_sentence = np.roll(sentence, -20, axis=1)

    return next_sentence
    
def get_corpus():
    l_input = ['by', 'from', 'he', 'him', 'it', 'on', 'that', 'the', 'then', 'to', 'was', '.', 'N', 'V']
    l_data = [['N', 'V', 'the', 'N', '.', 'then', 'he', 'V', 'it', '.'], #0
              ['the', 'N', 'was', 'V', 'by', 'N', '.', 'then', 'he', 'V', 'it', '.'], #1
              ['N', 'V', 'the', 'N', 'on', 'the', 'N', '.', 'then', 'he', 'V', 'it', '.'], #2
              ['the', 'N', 'was', 'V', 'on', 'the', 'N', 'by', 'N', '.', 'then', 'he', 'V', 'it', '.'], #3
              ['it', 'was', 'on', 'the', 'N', 'that', 'N', 'V', 'the', 'N', '.', 'then', 'he', 'V', 'it', '.'], #4
              ['N', 'V', 'the', 'N', '.'], #5
              ['the', 'N', 'was', 'V', 'by', 'N', '.'], #6
              ['N', 'V', 'the', 'N', 'on', 'the', 'N', '.'], #7
              ['the', 'N', 'was', 'V', 'on', 'the', 'N', 'by', 'N', '.'], #8
              ['it', 'was', 'on', 'the', 'N', 'that', 'N', 'V', 'the', 'N', '.'], #9
              ['N', 'V', 'the', 'N', '.', 'then', 'it', 'V', 'him', '.'], #10
              ['the', 'N', 'was', 'V', 'by', 'N', '.', 'then', 'it', 'V', 'him', '.'], #11
              ['N', 'V', 'the', 'N', 'on', 'the', 'N', '.', 'then', 'it', 'V', 'him', '.'], #12
              ['the', 'N', 'was', 'V', 'on', 'the', 'N', 'by', 'N', '.', 'then', 'it', 'V', 'him', '.'], #13
              ['it', 'was', 'on', 'the', 'N', 'that', 'N', 'V', 'the', 'N', '.', 'then', 'it', 'V', 'him', '.'], #14
              ['the', 'N', 'V', 'the', 'N', '.'], #15
              ['the', 'N', 'was', 'V', 'by', 'the', 'N', '.'], #16
              ['the', 'N', 'V', 'the', 'N', 'to', 'the', 'N', '.'], #17
              ['the', 'N', 'was', 'V', 'to', 'the', 'N', 'by', 'the', 'N', '.'], #18
              ['the', 'N', 'V', 'the', 'N', 'the', 'N', '.'], #19
              ['the', 'N', 'that', 'V', 'the', 'N', 'V', 'the', 'N', '.'], #20
              ['the', 'N', 'was', 'V', 'by', 'the', 'N', 'that', 'V', 'the', 'N', '.'], #21
              ['the', 'N', 'that', 'V', 'the', 'N', 'was', 'V', 'by', 'the', 'N', '.'], #22
              ['the', 'N', 'V', 'the', 'N', 'that', 'V', 'the', 'N', '.'], #23
              ['the', 'N', 'that', 'was', 'V', 'by', 'the', 'N', 'V', 'the', 'N', '.'], #24
              ['the', 'N', 'was', 'V', 'by', 'the', 'N', 'that', 'was', 'V', 'by', 'the', 'N', '.'], #25
              ['the', 'N', 'that', 'was', 'V', 'by', 'the', 'N', 'was', 'V', 'by', 'the', 'N', '.'], #26
              ['the', 'N', 'V', 'the', 'N', 'that', 'was', 'V', 'by', 'the', 'N', '.'], #27
              ['the', 'N', 'was', 'V', 'to', 'the', 'N', 'by', 'the', 'N', 'that', 'V', 'the', 'N', '.'], #28
              ['the', 'N', 'that', 'V', 'the', 'N', 'was', 'V', 'to', 'the', 'N', 'by', 'the', 'N', '.'], #29
              ['the', 'N', 'V', 'the', 'N', 'to', 'the', 'N', 'that', 'V', 'the', 'N', '.'], #30
              ['the', 'N', 'was', 'V', 'from', 'the', 'N', 'to', 'the', 'N', 'that', 'V', 'the', 'N', '.'], #31
              ['the', 'N', 'that', 'was', 'V', 'by', 'the', 'N', 'V', 'the', 'N', 'to', 'the', 'N', '.'], #32
              ['the', 'N', 'V', 'the', 'N', 'to', 'the', 'N', 'that', 'was', 'V', 'by', 'the', 'N', '.'], #33
              ['the', 'N', 'that', 'V', 'the', 'N', 'to', 'the', 'N', 'V', 'the', 'N', '.'], #34
              ['the', 'N', 'was', 'V', 'by', 'the', 'N', 'that', 'V', 'the', 'N', 'to', 'the', 'N', '.'], #35
              ['the', 'N', 'V', 'the', 'N', 'that', 'V', 'the', 'N', 'to', 'the', 'N', '.'], #36
              ['the', 'N', 'that', 'V', 'the', 'N', 'to', 'the', 'N', 'was', 'V', 'by', 'the', 'N', '.'], #37
              ['the', 'N', 'that', 'was', 'V', 'to', 'the', 'N', 'by', 'the', 'N', 'V', 'the', 'N', '.'], #38
              ['the', 'N', 'V', 'the', 'N', 'that', 'was', 'V', 'by', 'the', 'N', 'to', 'the', 'N', '.'], #39
              ['the', 'N', 'that', 'V', 'the', 'N', 'V', 'the', 'N', 'to', 'the', 'N', '.'], #40
              ['the', 'N', 'that', 'the', 'N', 'V', 'V', 'the', 'N', '.'], #41
              ['the', 'N', 'that', 'the', 'N', 'V', 'was', 'V', 'by', 'the', 'N', '.'], #42
              ['the', 'N', 'that', 'the', 'N', 'V', 'V', 'the', 'N', 'to', 'the', 'N', '.'], #43
              ['the', 'N', 'that', 'the', 'N', 'V', 'V', 'the', 'N', 'the', 'N', '.']] #44
    return (l_input, l_data)

def get_coded_meaning():
    l_output = ['N1-A1','N1-O1','N1-R1','N1-A2','N1-O2','N1-R2','N2-A1','N2-O1','N2-R1','N2-A2','N2-O2','N2-R2','N3-A1','N3-O1','N3-R1','N3-A2','N3-O2','N3-R2','N4-A1','N4-O1','N4-R1','N4-A2','N4-O2','N4-R2']
    l_teacher = [['N1-A1','N2-O1','N1-A2','N2-O2'],#0
         ['N2-A1','N1-O1','N2-A2','N1-O2'],#1
         ['N1-A1','N2-O1','N3-R1','N1-A2','N2-O2'], #2
         ['N3-A1','N1-O1','N2-R1','N3-A2','N1-O2'], #3
         ['N2-A1','N3-O1','N1-R1','N2-A2','N3-O2'], #4 
         ['N1-A1','N2-O1'],#5
         ['N2-A1','N1-O1'],#6
         ['N1-A1','N2-O1','N3-R1'], #7
         ['N3-A1','N1-O1','N2-R1'], #8
         ['N2-A1','N3-O1','N1-R1'], #9
         ['N1-A1','N2-O1','N2-A2','N1-O2'], #10
         ['N2-A1','N1-O1','N1-A2','N2-O2'], #11
         ['N1-A1','N2-O1','N3-R1','N2-A2','N1-O2'], #12
         ['N3-A1','N1-O1','N2-R1','N1-A2','N3-O2'], #13
         ['N2-A1','N3-O1','N1-R1','N3-A2','N2-O2'], #14
         ['N1-A1','N2-O1'], #15
         ['N2-A1','N1-O1'], #16
         ['N1-A1','N2-O1','N3-R1'], #17
         ['N3-A1','N1-O1','N2-R1'], #18
         ['N1-A1','N3-O1','N2-R1'], #19
         ['N1-A1','N2-O1','N1-A2','N3-O2'], #20
         ['N2-A1','N1-O1','N2-A2','N3-O2'], #21
         ['N1-A1','N2-O1','N3-A2','N1-O2'], #22
         ['N1-A1','N2-O1','N2-A2','N3-O2'], #23
         ['N2-A1','N1-O1','N1-A2','N3-O2'], #24
         ['N2-A1','N1-O1','N3-A2','N2-O2'], #25
         ['N2-A1','N1-O1','N3-A2','N1-O2'], #26
         ['N1-A1','N2-O1','N3-A2','N2-O2'], #27
         ['N3-A1','N1-O1','N2-R1','N3-A2','N4-O2'], #28 
         ['N1-A1','N2-O1','N4-A2','N1-O2','N3-R2'], #29
         ['N1-A1','N2-O1','N3-R1','N3-A2','N4-O2'], #30
         ['N2-A1','N1-O1','N3-R1','N3-A2','N4-O2'], #31
         ['N2-A1','N1-O1','N1-A2','N3-O2','N4-R2'], #32
         ['N1-A1','N2-O1','N3-R1','N4-A2','N3-O2'], #33
         ['N1-A1','N2-O1','N3-R1','N1-A2','N4-O2'], #34
         ['N2-A1','N1-O1','N2-A2','N3-O2','N4-R2'], #35
         ['N1-A1','N2-O1','N2-A2','N3-O2','N4-R2'], #36
         ['N1-A1','N2-O1','N3-R1','N4-A2','N1-O2'], #37
         ['N3-A1','N1-O1','N2-R1','N1-A2','N4-O2'], #38
         ['N1-A1','N2-O1','N3-A2','N2-O2','N4-R2'], #39
         ['N1-A1','N2-O1','N1-A2','N3-O2','N4-R2'], #40
         ['N1-O1','N1-A2','N2-A1','N3-O2'], #41
         ['N1-O1','N1-O2','N2-A1','N3-A2'], #42
         ['N1-O1','N1-A2','N2-A2','N3-O2','N4-R2'], #43
         ['N1-O1','N1-A2','N2-A2','N3-R2','N4-O2'], #44 
         ]
    return (l_output, l_teacher)

if __name__ == '__main__':
    pass
