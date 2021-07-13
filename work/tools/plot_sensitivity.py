import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import pickle
import argparse

my_linestyle = ['-','--','-.',':']
def plot_sensitivities(sensitivities,visualization,save_file):
    """Plot sensitivities from file.
    Args:
       sensitivities:  The sensitivities load from file using load_sensitivities().
       visualization:  Visualization of sensitivites.
       save_file:  Save visualization of sensitivites in save_file.
    """
    # matplotlib 中文
    from pylab import mpl

    mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
    print("Start to plot")
    # Plot
    figsize = (10,6)
    plt.figure(figsize=figsize)
    i = 0
    number = np.ceil(len(sensitivities)/2) if len(sensitivities) > 20 else len(sensitivities)
    count = 0
    # Have a look at the colormaps here and decide which one you'd like:
    # http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
    # plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.Accent(np.linspace(0, 1, number))))
    name_set = []
    for layer_name in sensitivities:
        name_set.append(layer_name)
        xy = [[sensitivity,0-sensitivities[layer_name][sensitivity]] for sensitivity in sensitivities[layer_name]]
        xy = np.array(xy)
        x,y = xy[:,0],xy[:,1]
        plt.plot(x,y,linestyle = my_linestyle[int(count/10)])
        if((i%number)==(number-1)) or (i+1==len(sensitivities)):
            plt.legend(name_set,bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)
            # plt.legend(name_set)
            name_set = []
            # plt.xlabel('剪枝率')
            # plt.ylabel('精度损失(%)')
            # plt.title('敏感性分析')
            plt.xlabel('Ratio')
            plt.ylabel('Accuracy Loss(%)')
            plt.title('Sensitivity analysis')
            plt.tight_layout()
            plt.savefig(save_file+str(int(np.ceil(i/number)))+'.png',dpi=300)
            if visualization:
                plt.show()
            plt.figure(figsize=figsize)
            count = 0
        i += 1
        count += 1
    print("Finished")

def load_sensitivities(sensitivities_file):
    """Load sensitivities from file.
    Args:
       sensitivities_file(str):  The file storing sensitivities.
    Returns:
       dict: A dict stroring sensitivities.
    """
    sensitivities = {}
    if sensitivities_file and os.path.exists(sensitivities_file):
        with open(sensitivities_file, 'rb') as f:
            if sys.version_info < (3, 0):
                sensitivities = pickle.load(f)
            else:
                sensitivities = pickle.load(f, encoding='bytes')
    return sensitivities

def main(args):
    
    sensitivities = load_sensitivities(args.file)
    sensitivities = sensitivities.sensitivies   # for dygraph
    if (len(sensitivities)<=0):
        raise Exception("Length of sensitivities is invalid.")
    if 'conv_last_weights' in sensitivities:
        conv_last_weights = sensitivities['conv_last_weights']
        del sensitivities['conv_last_weights']
        conv2d_layer = dict()
        for name in sensitivities.keys():
            if('conv2d' in name):
                conv2d_layer[name] = sensitivities[name]
        
        conv2d_layer = {name:conv2d_layer[name] for name in sorted(conv2d_layer.keys())}

        for name in conv2d_layer.keys():
            del sensitivities[name]
        sorted_sensitivities = {name:sensitivities[name] for name in sorted(sensitivities.keys(),key=lambda ele:int(ele[ele.find('v')+1:ele.find('_')]))}
        sorted_sensitivities.update(conv2d_layer)
        sorted_sensitivities['conv_last_weights'] = conv_last_weights
            
    else:
        sorted_sensitivities = {name:sensitivities[name] for name in sorted(sensitivities.keys())}
    plot_sensitivities(sorted_sensitivities,args.visualization,os.path.join(args.file.replace('.data','')))
    
    
    

if __name__ == "__main__":

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("-f",'--file',type=str,dest="file",help="Sensitivities file(.data)")
    parser.add_argument("-v",'--visualization',type=str2bool,dest="visualization",default=False,help="Visualize sensitivities")
    args = parser.parse_args()

    main(args)