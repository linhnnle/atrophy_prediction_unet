import matplotlib.pyplot as plt
import numpy as np
import h5py
def create_fig_for_model(results,output_path,figname = 'metrics.png',tablename = 'metrics.txt',comparable_level = []):
    '''
    this function plot the change of metrics as the threshold of the model changes
    inputs:
    results: should be a dictionary with keys that you need to plot
    output_path is the path you want figure to save.
    '''
    x = results['thres']
    y1 = results['auc']
    y2 = results['dice']
    y3 = results['volume_predicted']
    y4 = results['volume_difference']
    y5 = results['recall']
    y6 = results['precision']
    y7 = results['abs_volume_difference']
    y8 = results['specificity']

    fig, ((ax1,ax2),(ax3,ax4),(ax7,ax5),(ax6,ax8)) = plt.subplots(4,2,figsize= (8,12),sharex=False)


    ax1.set_xlabel('threshold of model')
    ax1.set_ylabel('auc')
    ax1.plot(x,y1,zorder=1, lw =1)
    ax1.scatter(x,y1,zorder=2,s=20)
    if 'auc' in comparable_level:
        ax1.axhline(y=comparable_level['auc'][1],color='lightslategrey', linestyle = '--')
        ax1.axhspan(ymin=comparable_level['auc'][0],ymax=comparable_level['auc'][2],color='lightslategrey', alpha=0.1)
    # ax2 = ax1.twinx()
    ax2.set_xlabel('threshold of model')
    ax2.set_ylabel('Volume Predicted (ml)')
    ax2.plot(x,y3, lw=1)
    ax2.scatter(x,y3,zorder=2,s=20)
    if 'volume_predicted' in comparable_level:
        ax2.axhline(y=comparable_level['volume_predicted'][1],color='lightslategrey', linestyle = '--')
        ax2.axhspan(ymin=comparable_level['volume_predicted'][0],ymax=comparable_level['volume_predicted'][2],color='lightslategrey', alpha=0.1)

    ax3.set_xlabel('threshold of model')
    ax3.set_ylabel('dice score')
    ax3.plot(x,y2,lw =1)
    ax3.scatter(x,y2,zorder=2,s=20)
    if 'dice' in comparable_level:
        ax3.axhline(y=comparable_level['dice'][1],color='lightslategrey', linestyle = '--')
        ax3.axhspan(ymin=comparable_level['dice'][0],ymax=comparable_level['dice'][2],color='lightslategrey', alpha=0.1)
    ax4.set_xlabel('threshold of model')
    ax4.set_ylabel('volume difference (ml)')
    ax4.plot(x,y4,lw =1)
    ax4.scatter(x,y4,zorder=2,s=20)
    if 'volume_difference' in comparable_level:
        ax4.axhline(y=comparable_level['volume_difference'][1],color='lightslategrey', linestyle = '--')
        ax4.axhspan(ymin=comparable_level['volume_difference'][0],ymax=comparable_level['volume_difference'][2],color='lightslategrey', alpha=0.1)

    ax5.set_xlabel('threshold of model')
    ax5.set_ylabel('recall')
    ax5.plot(x,y5,lw =1)
    ax5.scatter(x,y5,zorder=2,s=20)
    if 'recall' in comparable_level:
        ax5.axhline(y=comparable_level['recall'][1],color='lightslategrey', linestyle = '--')
        ax5.axhspan(ymin=comparable_level['recall'][0],ymax=comparable_level['recall'][2],color='lightslategrey', alpha=0.1)
    ax6.set_xlabel('threshold of model')
    ax6.set_ylabel('precision')
    ax6.plot(x,y6,lw =1)
    ax6.scatter(x,y6,zorder=2,s=20)
    if 'precision' in comparable_level:
        ax6.axhline(y=comparable_level['precision'][1],color='lightslategrey', linestyle = '--')
        ax6.axhspan(ymin=comparable_level['precision'][0],ymax=comparable_level['precision'][2],color='lightslategrey', alpha=0.1)
    ax7.set_xlabel('threshold of model')
    ax7.set_ylabel('absolute volume difference (ml)')
    ax7.plot(x,y7,lw =1)
    ax7.scatter(x,y7,zorder=2,s=20)
    if 'abs_volume_difference' in comparable_level:
        ax7.axhline(y=comparable_level['abs_volume_difference'][1],color='lightslategrey', linestyle = '--')
        ax7.axhspan(ymin=comparable_level['abs_volume_difference'][0],ymax=comparable_level['abs_volume_difference'][2],color='lightslategrey', alpha=0.1)
    ax8.set_xlabel('threshold of model')
    ax8.set_ylabel('specificity')
    ax8.plot(x,y8,lw =1)
    ax8.scatter(x,y8,zorder=2,s=20)
    if 'specificity' in comparable_level:
        ax8.axhline(y=comparable_level['specificity'][1],color='lightslategrey', linestyle = '--')
        ax8.axhspan(ymin=comparable_level['specificity'][0],ymax=comparable_level['specificity'][2],color='lightslategrey', alpha=0.1)
    fig.tight_layout()
    fig.savefig(output_path + figname)
    with open(output_path+tablename, 'w') as file:
      keylist = ','.join(str(key) for key in results.keys()) + '\n'
      file.write(keylist)
      for i in range(len(results['thres'])):
        row = []
        for key in results.keys():
          row += [results[key][i]]
          str1 = ','.join(str(n) for n in row) + '\n'
        file.write(str1)
    return print('figure and table saved at:', output_path + figname)


def create_roc(fpr,tpr,roc_auc,output_path, thresholds, figname = 'roc.png',tablename = 'table.csv',datawrite = True):
    youden = np.array(tpr) - np.array(fpr)
    max_youden = np.max(youden)
    cutoff_index = np.argmax(youden)
    # cutoff_index = youden.tolist().index(max_youden)

    cutoff = thresholds[cutoff_index]
    print(cutoff_index,cutoff, max_youden)
    plt.plot(fpr,tpr,color = 'darkorange', label = 'ROC curve (AUC = %0.3f)' % roc_auc)
    plt.plot([0,1],[0,1],color = 'navy', linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Cutoff = {0:.2f}, Youden Index = {1:.2f}, sensitivity {2:.2f}, specificity {3:.2f}'.format(cutoff, max_youden,tpr[cutoff_index],1-fpr[cutoff_index]) )
    plt.legend(loc="lower right")
    plt.savefig(output_path + figname)
    plt.close()
    if datawrite:
        output_csv = np.column_stack((thresholds,fpr,tpr))
        np.savetxt(output_path + tablename, output_csv, fmt='%4.3f,%4.3f,%4.3f', delimiter=",")
    print('data output =', datawrite, ', figure saved at:', output_path + figname)
    return cutoff

def save_dict(dict,output_path,filename,summary=True):
    '''
    save dictionary into csv file
    :param dict: dictionary you want to save
    :param output_path:
    :param filename: e.g. file.csv
    :return:
    '''
    header =[0,]
    median = [0,'median']
    percentile25 = [0,'25th percentile']
    percentile75 = [0,'75th percentile']
    for key in dict:
        length = len(dict[key])
    output_csv = np.empty((length,))
    for key in dict:
        output_csv = np.column_stack((output_csv,np.array(dict[key])))
        header += [key]
        if summary:
            if not key == 'subject':
                median += [np.median(dict[key])]
                percentile25 += [np.percentile(dict[key],25)]
                percentile75 += [np.percentile(dict[key],75)]
    if summary:
        output_csv = np.row_stack((np.array(header),output_csv,median,percentile25,percentile75))
    else:
        output_csv = np.row_stack((np.array(header), output_csv))
    np.savetxt(output_path + filename, output_csv[:,1:], fmt='%s', delimiter=",")
    return print('dictionary saved at ', output_path + filename, 'with summary', summary)
