import numpy as np
import torch
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('-a',  '--alpha', type = float, default = 0.1)
parser.add_argument('--dataset', type = str, default = 'CIFAR10')


args = parser.parse_args()


def SavePlot(args, path, x = None, y = None, row = None, col = None, hue = None, data = None, kind = None, legend = False, rotation = 45, font_scale = 1.5, xticks = None, ylim = None, Title = True):
    sns.set_style("whitegrid")

    font = 32

    sns.set(rc={'figure.figsize':(17.7,8.27)})

    s1 = sns.catplot(data = data, x = x, y = y, hue = hue, kind = kind, col = col, row = row, errorbar = 'sd', height = 6, aspect = 2, legend = legend, legend_out = False, palette = ['orange', 'magenta', 'blue', 'lime'])
    axes = s1.axes.flatten()
    if not Title:
        s1.set(title = '')

    if y == 'Marginal Coverage':
        for i, ax in enumerate(axes):
            s1.set_xticklabels(xticks, fontsize = font, rotation = rotation)

            ax.tick_params(axis = 'y', labelsize = font)

            ax.set_title(label = ax.get_title(), fontsize = font)
            ax.axhline(1-args.alpha, ls='--', c='red', linewidth = 6)
            RedLine = mlines.Line2D([], [], color='red', linestyle='--',
                            markersize=font, label='Target Coverage', linewidth = 6)


            if i == 0:
                ax.legend(handles=[RedLine], loc='upper right', frameon=True, fontsize=30, fancybox=True, framealpha=0.9, ncols = 1)
            ax.set_xticklabels(xticks, fontsize = font, rotation = rotation)
            ax.tick_params(labelbottom=True)

        s1.set_axis_labels('', "Marginal Coverage", fontsize = font)


        s1.set(ylim=(0.80, 1.0))

    if y == 'Prediction Size':
        for i, ax in enumerate(axes):
            s1.set_xticklabels(xticks, fontsize = font, rotation = rotation)
            ax.set_title(label = ax.get_title(), fontsize = font)
            ax.tick_params(axis = 'y', labelsize = font)
                                                   

            ax.set_xticklabels(xticks, fontsize = font, rotation = rotation)
            ax.tick_params(labelbottom=True)

        s1.set_axis_labels('', "Prediction Set Size", fontsize = font)

        s1.set(ylim=ylim)


    s1.set_xticklabels(xticks, fontsize = font, rotation = rotation)


    plt.tight_layout()
    s1.savefig(path, dpi = 100, bbox_inches='tight', pad_inches=.1)
    plt.close('all')


if args.dataset == 'CIFAR10':
    args.alpha = 0.04

ResultsList = ['APS', 'RAPS', 'NCP(APS)', 'NCP(RAPS)']
patha = './Results/Dataset_' + str(args.dataset) + '/alpha_' + str(args.alpha) 
AllResults = pd.DataFrame()

for resultName in ResultsList:
    path = patha + '/' + resultName + '.pt'
    results = torch.load(path)
    Coverage = results['Coverage'].to_numpy()
    Size = results['Size'].to_numpy()
    res = pd.DataFrame({'Marginal Coverage': Coverage, 'Prediction Size': Size})
    res['Method'] = resultName
    AllResults = AllResults.append(res)



SavePlot(args, patha  + '/' + str(args.dataset) + 'Coverage.png', x = 'Method', y = 'Marginal Coverage', data = AllResults, kind = 'boxen', legend = False,rotation=0, font_scale = 2, xticks = ResultsList)
SavePlot(args, patha  + '/' + str(args.dataset) + 'Size.png', x = 'Method', y = 'Prediction Size', data = AllResults, kind = 'boxen', legend = False, rotation=0, font_scale = 2, xticks = ResultsList, Title = False)
