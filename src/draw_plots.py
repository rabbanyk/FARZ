import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import UnivariateSpline
from scipy import interpolate
from sklearn.neighbors.kde import KernelDensity
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from matplotlib.patches import Polygon

matplotlib.rcParams.update({'font.size': 14, 
                      'font.family': 'Times New Roman',
#                       'legend.frameon': False
                      })
colrs= ["#429617",
        "#4A76BD",
        "#E04927",
        "#EEC958",
        "#B962CE",
        "#57445A",
        "#716035",
        "#C6617E",
        "#92C4AE"]
markers=['o','>','s','d','*','v','<','o','>','s','d','*','v','<']



def rewirings(data):
    fig,axess = plt.subplots(2, 2, True,True, squeeze=False, figsize=(8,8))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.95,
                    wspace=0.02, hspace=0.14)
    
    network_models = ['CF', 'BA', 'FF', 'FF+BA']
    assign_methods = ["LFR","LFR-CN","LFR-NE"]
    
    for i in {0,1}:
        for j in {0,1}:
            mean, var = data[i][j]
            k = i+j
            ax = axess[i,j]
            for m in range(0, len(mean)):
                ax.errorbar(np.arange(1,len(mean[m])+1)*0.1, mean[m], yerr=var[m],
                             label =assign_methods[m], c=colrs[m], marker=markers[m],linewidth=1.2)#, xerr=0.4)
            ax.legend()
            ax.set_title(network_models[k])
    
    # axess[1,0].set_xlabel("mixing parameter $\mu$")
    # axess[0,0].set_ylabel("percentage of rewirings")
    axess[0,0].set_xlim(0, (len(mean[i])+1)*0.1)
    
    fig.text(0.5, 0.05, 'mixing parameter $\mu$', ha='center', va='center', size=22)
    fig.text(0.025, 0.5, 'percentage of rewirings', ha='center', va='center', rotation='vertical', size=22)
    
    # plt.tight_layout()
    plt.show()
    
    
# fakeMean = [[.7,.6,.3,.2,.3,.2],[.4,.3,.2,.4,.35,.25],[.2,.3,.4,.5,.55,.44]]
# fakevar = [[.05,.05,.05,.05,.05,.05],[.05,.05,.05,.05,.05,.05],[.05,.05,.05,.05,.05,.05]]
# fakeOnePlotData = (fakeMean,fakevar)
# rewirings( [[fakeOnePlotData,fakeOnePlotData],[fakeOnePlotData,fakeOnePlotData]] )    
#     
#     
    
    
#sequences for degrees, shortest_pathes, clustering_coeffients, degree_corelations    
def basic_properties_freq( sequences , axess=None, labl = None, logscale=False, markr=None, clr=None,offset=0):
    if axess is None:
        fig,axess = plt.subplots( 3,len(sequences),'col',False, squeeze=False, figsize=(10,8))
    plt.subplots_adjust(left=0.04, bottom=0.05, right=0.98, top=0.94,   wspace=0.28, hspace=0.1)

    for i in range(0,len(sequences)):
        ax = axess[offset,i]
        seq = sequences[i]
        seq = [f for f in seq if f>0]
        smax =max(seq)
        smin =min(seq)
        
        #print seq
        freqs , bin_edges = np.histogram(seq,  smax+1 if smax>1 else 100, range = (0,smax+1) if smax>1 else (0,smax))#, normed = True, density=True)
        bin_centers =  (bin_edges[:-1] + bin_edges[1:])/2.
        vals = range(0,smax+1) if smax>1 else bin_centers
        freqs=freqs*1.0/sum(freqs)
     
#         fplot = ax.loglog if lplot else ax.plot
       
#         his, = ax.plot(vals, freqs,lw=0, label=labl, alpha =0.8, color = clr ,  marker =markr)
#         if lplot:
#             his, = ax.loglog(vals, freqs,'.', marker ='.', label=labl, alpha =0.5)
#             his, = ax.loglog(vals, freqs,'.', marker ='.', label=labl, alpha =0.5)
#         else :
#             his, = ax.plot(vals, freqs,'.', marker ='.', label=labl, alpha =0.5)
#             his, = ax.loglog(vals, freqs,'.', marker ='.', label=labl, alpha =0.5)

      
#         x = bin_centers
#         f = UnivariateSpline(x, freqs)#, s=0.1*len(freq))
#         ax.plot(x, f(x),c= his.get_color(), alpha=0.5)
#         print len(freqs) #, freqs
#         print bin_edges
#         print bin_centers

        #remove zeros
        y = np.array(freqs)
        nz_indexes = np.nonzero(y)
        y = y[nz_indexes]
        x = np.array(vals)[nz_indexes]

#         ax.plot(x, y,':',c= his.get_color(),alpha=0.8)
        ax.plot(x, y,':', label=labl, alpha =0.8, color = clr ,  marker =markr)
#         f = interpolate.interp1d(x, y, kind='linear')
#         f = interpolate.UnivariateSpline(x, y, k=2)
#         
#         xs = np.linspace(min(x),max(x),200) 
#         ys = f(xs)   # use interpolation function returned by `interp1d`
#         ax.plot(xs, ys,'-',c= his.get_color(),)
#         
#         density = gaussian_kde(y)
#         xs = np.linspace(min(x),max(x),200)
# #         density.covariance_factor = lambda : .5
# #         density._compute_covariance()
#         ax.plot(xs,density(xs),c= his.get_color(), alpha=0.5)
        if len(logscale)==len(sequences): 
            if 'x' in logscale[i] : ax.set_xscale('log')
            if 'y' in logscale[i] : ax.set_yscale('log')
#         ax.legend()
#     plt.show()
    return axess



#sequences for degrees, shortest_pathes, clustering_coeffients, degree_corelations    
def basic_properties( sequences , axess=None, labl = None, logscale=[False], markr='.', clr='k',offset=0, alfa = 0.8,
                      distir = [False,False,False, False], bandwidths = [3, 0.1,0.01,1], limits = [(1,50),(0,1),(0,1),(1,25)] ):
    if axess is None:
        fig,axess = plt.subplots( 3, len(sequences),False,False, squeeze=False,figsize=(len(sequences)*3,8))#'col'
    plt.subplots_adjust(left=0.12, bottom=0.05, right=0.95, top=0.94,   wspace=0.28, hspace=0.1)
    plt.subplots_adjust(left=0.45, bottom=0.05, right=0.95, top=0.94,   wspace=0.28, hspace=1.2)

    for i in range(0,len(sequences)):
        ax = axess[offset][i]
        seq = sequences[i]
        smax =max(seq)
        smin =min(seq)

        if distir[i]==0:
            #print seq
            freqs , bin_edges = np.histogram(seq,  smax+1 if smax>1 else 100, range = (0,smax+1) if smax>1 else (0,smax))#, normed = True, density=True)
            bin_centers =  (bin_edges[:-1] + bin_edges[1:])/2.
            vals = range(0,smax+1) if smax>1 else bin_centers
            freqs=freqs*1.0/sum(freqs)
            #remove zeros
            y = np.array(freqs)
            nz_indexes = np.nonzero(y)
            y = y[nz_indexes]
            x = np.array(vals)[nz_indexes]
            ax.plot(x, y,':', label=labl, alpha =alfa, color = clr ,  marker ='.')
        else :
            X = np.array(seq)
            X = [ x for x in X if x>=limits[i][0] and x<=limits[i][1]]
    #         X= (np.abs(X))
#             print len(X)
            X = np.random.choice(X, size=min(10000, len(X)))
            X = X[:, np.newaxis]
            kde = KernelDensity(kernel = 'gaussian', bandwidth=bandwidths[i]).fit(X)#,atol=atols[i],kernel = 'tophat'kernel='gaussian'
#             if 'x' in logscale[i] : 
#                 X_plot = np.logspace( limits[i][0],  limits[i][1], 1000)[:, np.newaxis]
#             else :
            X_plot = np.linspace(limits[i][0], limits[i][1], 1000)[:, np.newaxis]
    
            log_dens = kde.score_samples(X_plot) #
    #         ax.fill(X_plot[:, 0], np.exp(log_dens), alpha =0.5, label=labl)
            Y  =  np.exp(log_dens)
            if  distir[i]==2: Y = np.cumsum(Y)
            ax.plot(X_plot[:, 0],Y, '-',label=labl, alpha =alfa, color = clr ,markersize=2,  marker ='')
    
            verts = [(limits[i][0]-1e-6, 0)] + list(zip(X_plot[:, 0],Y)) + [(limits[i][1]+1e-6, 0)]
            poly = Polygon(verts, facecolor=clr,  alpha =alfa ) #, edgecolor='0.5')
            ax.add_patch(poly)
    #         ax.set_yticks([])
    #         ax.set_ylim(bottom=-0.02)
            ax.set_xlim(limits[i][0],limits[i][1])
            
        if len(logscale)==len(sequences): 
            if 'x' in logscale[i] : 
                ax.set_xscale('log')
            if 'y' in logscale[i] : 
                ax.set_yscale('log')
                if i<3: ax.set_ylim(bottom=0.001)
#         ax.legend()
#         plt.show(block=False)
    return axess

def test_density_plot():
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    
    N=20
    X = np.concatenate((np.random.normal(0, 1, 0.3 * N),
                        np.random.normal(5, 1, 0.7 * N)))[:, np.newaxis]
                        
    print np.shape(X)
    X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]   
    print np.shape(X_plot)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax[0,0].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
    ax[0,0].text(-3.5, 0.31, "Gaussian Kernel Density")
    ax[0,0].plot(X[:, 0], np.zeros(X.shape[0]) - 0.01, '+k')
    
    plt.show()
    
