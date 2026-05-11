import numpy as np
from scipy.spatial.distance import pdist,squareform
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram,linkage


def auto_split_range(data,cmap='bwr',force_range=None):
    output = {
        'vmin':None,
        'vmax':None,
        'cmap':cmap
    }

    data_max = np.max(data)
    data_min = np.min(data)

    abs_range = np.max([data_max,np.abs(data_min)])

    if force_range is not None:
        abs_range = force_range
        
    output['vmin'] = -1 * abs_range
    output['vmax'] = abs_range

    return output


def concordance_plot(m1,m2,label_1="Mtx 1",label_2="Mtx 2",title=None,
                     agglomerate=None,
                     metric="correlation",
                     path=None,
                     plot_figure=True,
                     ticks_1=None,ticks_2=None,
                     figsize=(10,10),
                     cmap="bwr",
                     cmap_clip=None,
                     plot_text=True,
                     fontsize=8,
                    ):
    
    dim_1 = m1.shape[1]
    dim_2 = m2.shape[1]

    if ticks_1 is None:
        ticks_1 = np.arange(dim_1)
    if ticks_2 is None:
        ticks_2 = np.arange(dim_2)

    stacked = np.hstack([m1,m2])

    if metric == "partial_correlation":
        covariance = np.cov(stacked.T)
        precision = np.linalg.pinv(covariance)
        normalization = np.sqrt(np.outer(np.diag(precision),np.diag(precision)))
        partial_correlation = -1 * np.divide(precision,normalization)
        dist = 1-partial_correlation

    elif metric == "cosine":
        # dist = -1 * (squareform(pdist(stacked.T,metric='cosine')) - 1)
        # dist = squareform(pdist(stacked.T,metric='cosine'))
        dist = cosine_similarity(stacked.T)
    elif metric == "correlation":
        dist = 1 - squareform(pdist(stacked.T,metric='correlation'))
    elif metric == "spearman":
        # spearman is the opposite column orientation of pdist -_-
        dist = spearmanr(stacked)[0]
    else: 
        raise Exception(f"Only the following metrics are supported: cosine, correlation, spearman. You entered {metric}")

    print(dist.shape)
    con_dist = dist[dim_1:,:dim_1]

    row_sort = np.arange(dim_1)
    col_sort = np.arange(dim_2)

    if agglomerate is True:
        row_sort = single_agg(con_dist)
        col_sort = single_agg(con_dist.T)
        con_dist = con_dist[row_sort]
        con_dist = con_dist.T[col_sort].T
        ticks_1 = np.array(ticks_1)[col_sort]
        ticks_2 = np.array(ticks_2)[row_sort]
    if isinstance(agglomerate,list):
        col_sort = agglomerate[0]
        row_sort = agglomerate[1]
        con_dist = con_dist[row_sort]
        con_dist = con_dist.T[col_sort].T
        if len(ticks_1) > 0:
            ticks_1 = np.array(ticks_1)[col_sort]
        if len(ticks_2) > 0:
            ticks_2 = np.array(ticks_2)[row_sort]
    if agglomerate is None:
        pass

    res = {'correlations':con_dist,'row_sort':row_sort,'col_sort':col_sort}

    if not plot_figure:
        return res

    if cmap_clip is None:
        cmap_clip = [-1,1]
    plt.figure(figsize=figsize)
    plt.imshow(con_dist,vmin=cmap_clip[0],vmax=cmap_clip[1],cmap=cmap,aspect='auto')
    if title is None:
        title = f"Concordance between \n {label_1} and {label_2}"
    plt.title(title)


    if plot_text:
        text_style = {'horizontalalignment':'center','verticalalignment':'center','fontsize':fontsize}
    
        for i in range(dim_2):
            for j in range(dim_1):
                plt.text(j,i,str(np.round(con_dist[i,j],2)),**text_style)


    plt.xticks(np.arange(dim_1),labels=ticks_1,rotation=90, fontsize=fontsize)
    plt.yticks(np.arange(dim_2),labels=ticks_2, fontsize=fontsize)

    plt.colorbar()
    plt.xlabel(label_1)
    plt.ylabel(label_2)

    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    plt.show()

    return res

def compound_argsort(*args):
    compound = list(zip(range(len(args[0])),*args))
    sorted_compound = sorted(compound,key=lambda x: (x[1],x[2]))
    return np.array([x[0] for x in sorted_compound])

def single_agg(mtx,metric='cosine',method='average'):
    return dendrogram(linkage(mtx, metric=metric, method=method), no_plot=True)['leaves']

def double_agg(mtx,metric='cosine',method='average'):
    row_agg = dendrogram(linkage(mtx, metric=metric, method=method), no_plot=True)['leaves']
    col_agg = dendrogram(linkage(mtx.T, metric=metric, method=method), no_plot=True)['leaves']
    return mtx[row_agg].T[col_agg].T

# Prisma color sets from Ceisel, saturation is low
score_colors_original = {
    'parthanatos': '#FFA0A0',
    'necroptosis': '#CAFFCA',
    'apoptosis': '#C8DEF9'
}

# Prisma color scheme boosted
score_colors = {
    'parthanatos': '#ff5050',
    'necroptosis': '#65ff65',
    'apoptosis': '#61a8ff',
}
