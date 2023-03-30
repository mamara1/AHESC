"""
Created on Thursday Mar 30 11:51:32 2023

File containing functions to automatically layout plots and save them.

@author: thomas.vezin
"""

import matplotlib.pyplot as plt

# function to automatically set the desired layout
def setLayout(dictPlotParams,**kwargs):
    title=''
    if 'title' in kwargs: title=kwargs.get('title')
    legend=False  
    if 'legend' in kwargs: legend=kwargs.get('legend')
    xlbl = ''
    if 'xLabel' in kwargs: xlbl = kwargs.get('xLabel')
    ylbl = ''
    if 'yLabel' in kwargs: ylbl = kwargs.get('yLabel')
    xtickrot = False
    if 'xTickRotation' in kwargs: xtickrot = kwargs.get('xTickRotation')
        
    plt.title(kwargs.get('title'),fontsize=dictPlotParams['fontsize_title'])
    plt.gca().tick_params(axis='both',labelsize=dictPlotParams['fontsize'])
    plt.xlabel(xlbl,fontsize=dictPlotParams['fontsize'])
    plt.ylabel(ylbl,fontsize=dictPlotParams['fontsize'])
    if xtickrot: plt.xticks(rotation=dictPlotParams["xticks_rotation"])
    if legend: plt.legend(fontsize=dictPlotParams['fontsize_lgd'])
    
    return 1

# function to automatically save in png and pdf format.
def savePlot(filename):
    """Saves current figure both in png and pdf."""
    plt.savefig(foldFig+filename+'.png',dpi=300,transparent=True)
    plt.savefig(foldFig+filename+'.pdf',transparent=True)