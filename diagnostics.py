import matplotlib.pyplot as plt
import numpy as np

def quick_plot(axis, f, T):
    axis.plot( f.axis, f.profile, '.-', label=T )
    # can shift the plus/minus curves


# plot the density and flux profile
class diagnostic_1:
    
    def __init__(self,win=(8,4)):
        fig, axs = plt.subplots(1,2,figsize=win)
        self.axs = axs

    def plot(self, density, Gamma, Time):

        tlabel = 'T = {:.2e}'.format(Time)
        self.axs[0].plot( density.axis, density.profile,'.-',label=tlabel )
        self.axs[1].plot( Gamma.axis,   Gamma.profile,  '.-',label=tlabel )

    def label(self, title='', n_max=4.2):

        a0,a1 = self.axs
        
        a0.set_title(title)
        a0.set_ylim(0,n_max)
        a0.grid()
        
        a1.set_title('Gamma(rho)')
        a1.legend()
        a1.grid()

# plot two general profiles
class diagnostic_2:
    
    def __init__(self,win=(8,4)):
        fig, axs = plt.subplots(1,2,figsize=win)
        self.axs = axs

    def plot(self, f, g, Time):

        tlabel = 'T = {:.2e}'.format(Time)
        a0, a1 = self.axs
        quick_plot(a0, f, tlabel)
        quick_plot(a1, g, tlabel)

    def label(self, t0='', t1=''):

        a0,a1 = self.axs
        
        a0.set_title(t0)
        a0.grid()
        
        a1.set_title(t1)
        a1.legend()
        a1.grid()

# plot a function and its two shifts 
class diagnostic_3:
    
    def __init__(self,win=(10,4)):
        fig, axs = plt.subplots(1,3,figsize=win)
        self.axs = axs

    def plot(self, f, time):

        a0,a1,a2 = self.axs
        tlabel = 'T = {:.2e}'.format(time)

        quick_plot(a0, f,       tlabel)
        quick_plot(a1, f.plus,  tlabel)
        quick_plot(a2, f.minus, tlabel)

    def label(self, tag='F'):

        a0,a1,a2 = self.axs
        
        a0.set_title(r'$%s$'%tag)
        a0.grid()
        
        a1.set_title(r'$%s_+$'%tag)
        a1.grid()

        a2.set_title(r'$%s_-$'%tag)
        a2.grid()

        a2.legend()

# plot four general profiles
class diagnostic_4:
    
    def __init__(self,win=(6,6)):
        fig, axs = plt.subplots(2,2,figsize=win)
        self.axs = np.ravel(axs)
        self.fig = fig

    def plot(self, f, g, h, i, Time):

        tlabel = 'T = {:.2e}'.format(Time)

        a0, a1, a2, a3 = self.axs
        quick_plot(a0, f, tlabel)
        quick_plot(a1, g, tlabel)
        quick_plot(a2, h, tlabel)
        quick_plot(a3, i, tlabel)

    def label(self, titles = ['', '', '', '']):


        for j in np.arange(4):
            a = self.axs[j]
            t = titles[j]

            a.set_title(t)
            a.grid()

        a.legend()

    def title(self,title):
        self.fig.suptitle(title)
        #self.axs[0].suptitle(title)
        self.fig.tight_layout()

