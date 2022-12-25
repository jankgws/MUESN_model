import numpy as np
import pylab as pl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def lighten_color(color, amount = 0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def plot_PCA(PC_X, PC_U, root_file_name="",subtitle=""):
    pp = PdfPages(str(root_file_name)+'_'+str(subtitle)+'.pdf')
    colors = ['red', 'orange', 'yellow', 'green', 'turquoise', 'blue', 'purple']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    idx = 0
    patches = []
    print("Generating figures...\n")
    #for i in range(7):
    #index = np.random.permutation(np.arange(idx + 100, idx + length_x[i]))[:6000]
    #index = np.arange(idx + 100, idx + 100)
    ax.scatter(PC_X[0:,0], PC_X[0:,1], PC_U[0:,0], s = 0.1)
    ax.scatter(PC_X[0:,0], PC_X[0:,1], np.min(PC_U) - 0.5, s = 0.1)
    #idx += length_x[i]
    #patches.append(mpatches.Patch(color = colors[i], label = 'M = {:d}'.format(i)))

    ax.set_xlabel('Reservoir PC 1')
    ax.set_ylabel('Reservoir PC 2')
    ax.set_zlabel('Input PC 1')
    #ax.legend(handles = patches)
    #plt.savefig(dirname + "/fig.png")
    pl.draw()
    pp.savefig()
    pp.close()
    pl.show(block = True)

def get_labels(l_data, subset, l_offset, initial_pause=True, verbose=False):
    """
    データはこのように整理される。
        - l_data は文のリストである。
        - 各文章は単語のリスト
        
    ラベルの目盛りが生成される。プロットする場合、目盛りは最初の休止があるかのようにプロットされることに注意してください（目盛りはx=0で始まらないからです）。
        
    - offset: データ中の最大単語数と与えられた文の単語数との差を表す。
    - l_offset: l_dataの各文に対するオフセットのリストである。
    """
    if subset is None:
        labels = l_data
        offsets = l_offset
    else:
        labels = [l_data[x] for x in subset]
        offsets = [l_offset[x] for x in subset]
    lab_tick = len(labels)*[None]
    for i1 in range(len(labels)):
        lab_tick[i1] = [' ']*offsets[i1]
        for i2 in range(len(labels[i1])): 
            lab_tick[i1].append(labels[i1][i2])
        if initial_pause==False:
            # 最初の一時停止がない場合、最初のティックを削除する（ティックの問題を回避する簡単な方法）。
            lab_tick[i1] = lab_tick[i1][1:]
        if verbose:
            print("ticks :", lab_tick[i1])
    return (labels, lab_tick)

def plot_output(_outputs, d_io, save_pdf=True, root_file_name="",subtitle="", y_lim=[-0.1,1.5]):
    
    print(" *** Plotting outputs *** ")
    print(" * root_file_name="+root_file_name+" - subtitle="+subtitle+" * ")
    
    subset = d_io['subset']
    l_data = d_io['l_data']
    (labels, lab_tick) = get_labels(l_data=l_data, subset=subset,
                                    initial_pause=d_io['initial_pause'], l_offset=d_io['l_offset'])

    ## Plotting procedure
    if save_pdf:
        ## Initiate object PdfPages for saving figures
        pp = PdfPages(str(root_file_name)+'_'+str(subtitle)+'.pdf')
        pl.rcParams["axes.prop_cycle"] = pl.cycler("color", pl.get_cmap("tab20").colors)
    for i in range(len(_outputs)):
        pl.figure()
        pl.plot(_outputs[i])
        pl.legend(d_io['l_input'], loc='upper left')
        pl.suptitle("Testing sentence "+str(subset[i])+ ": '"+" ".join(labels[i])+"'"+"\n"+subtitle)
        pl.xticks(range(d_io['act_time'],_outputs[i].shape[0],d_io['act_time']))
        pl.ylabel('output')
        a = plt.gca()
        if y_lim!=None:
            a.set_ylim(y_lim)
        a.set_xticklabels(lab_tick[i], fontdict=None, minor=False)
            
        if save_pdf:
            # Save figure for each plot
            pp.savefig()
        pl.close()
    if save_pdf:
        ## Close object PdfPages
        pp.close()
    print(" * Plot finished * ")
    print(" *** ")
    

def plot_with_output_fashion(l_array, subset, d_io, root_file_name, subtitle="_output_fashion", legend=None, y_lim=None, verbose=False):
    print(" *** Plotting with output fashion *** ")
    print(" * root_file_name="+root_file_name+" - "+subtitle+" * ")
    
    (labels, lab_tick) = get_labels(l_data=d_io['l_data'], subset=subset,
                                    initial_pause=d_io['initial_pause'], l_offset=d_io['l_offset'])

    pp = PdfPages(str(root_file_name)+'_'+str(subtitle)+'.pdf')
    
    for i in range(len(l_array)):
        if verbose:
            print("idx_sentence", subset[i])
            print("i="+str(i))
            print("output[i]", l_array[i])
            print("label_sentence", labels[i])
            print("len(label_sentence)", len(labels[i]))
            print("words_tick", lab_tick[i])
        pl.figure()
        pl.plot(l_array[i])
        if legend is not None:
            pl.legend(legend)
        
        pl.suptitle("Testing sentence "+str(subset[i])+ ": '"+" ".join(labels[i])+"'"+"\n"+subtitle)
        pl.xticks(range(d_io['act_time'],l_array[i].shape[0],d_io['act_time']))
        a = plt.gca()
        if y_lim!=None:
            a.set_ylim(y_lim)
        a.set_xticklabels(lab_tick[i], fontdict=None, minor=False)
        
        pp.savefig()
        pl.close()
    pp.close()
    print(" * Plot finished * ")
    print(" *** ")

def plot_array_in_file(root_file_name, array_, data_subset=None, titles_subset=None, plot_slice=None, title="", subtitle="", legend_=None):
    """
    入力する。
        array_: プロットする配列または行列。
        data_subset: データ全体のうち、処理する部分集合に相当する。/
            array_ と subset は同じ長さでなければならない。
        titles_subset: サブタイトルのリスト
        plot_slice: array_ のうちプロットされる要素を決定するスライス。
    """
    if data_subset is None:
        data_subset = range(len(array_))
    if titles_subset is None:
        titles_subset = ['' for _ in range(len(data_subset))]
        nl_titles_sub = ''
    else:
        nl_titles_sub = '\n'
    if array_==[] or array_==np.array([]):
        import warnings
        warnings.warn("Warning: array empty. Could not be plotted. Title:"+str(title))
        return
    if plot_slice is None:
        plot_slice = slice(0,len(data_subset))
    else:
        if (plot_slice.stop-1) > len(data_subset):
            raise Exception("The last element of the slice is out of the subset.")
        subtitle = subtitle+"_slice-"+str(plot_slice.start)+"-"+str(plot_slice.stop)+"-"+str(plot_slice.step)
    ppIS = PdfPages(str(root_file_name)+str(title)+'.pdf')
    
    for i in range(plot_slice.stop)[plot_slice]:
        pl.figure()
        pl.suptitle(title+" "+str(titles_subset[i])+nl_titles_sub+" - seq "+str(data_subset[i])+"\n"+subtitle)
        pl.plot(array_[i])
        if legend_ is not None:
            pl.legend(legend_)
        ppIS.savefig()
        pl.close()
    ppIS.close()

if __name__ == '__main__':
    pass