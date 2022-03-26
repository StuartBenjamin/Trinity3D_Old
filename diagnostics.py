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
    
    def __init__(self,win=(5,8)):
        fig, axs = plt.subplots(3,1,figsize=win)
        self.axs = axs
        self.fig = fig

    def plot(self, f,g,h, time):

        a0,a1,a2 = self.axs
        tlabel = 'T = {:.2e}'.format(time)

        quick_plot(a0, f, tlabel)
        quick_plot(a1, g, tlabel)
        quick_plot(a2, h, tlabel)

    def title(self,title):
        self.fig.suptitle(title)
        self.fig.tight_layout()

    def label(self, titles = ['', '', '']):

        for j in np.arange(3):
            a = self.axs[j]
            t = titles[j]

            a.set_title(t)
            a.grid()

        self.fig.tight_layout()

# plot four general profiles
class diagnostic_4:
    
    def __init__(self,win=(5,5)):
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

        self.fig.tight_layout()

    def title(self,title):
        self.fig.suptitle(title)
        self.fig.tight_layout()

    def legend(self,j=3):
        a = self.axs[j]
        a.legend()

# I could probably generalize the diagnostic function,
# such that the window size is variably set in INIT

class ProfileSaver:

    def __init__(self):

        # saves TRINITY data as a nested dictionary
        log = {}

        # init profiles
        log['time']  = []
        log['n']     = []
        log['pi']    = []
        log['pe']    = []
        log['Gamma'] = []
        log['Qi']    = []
        log['Qe']    = []

        self.log = log

    def save(self,engine):

        n  = engine.density.profile
        pi = engine.pressure_i.profile
        pe = engine.pressure_e.profile
        G  = engine.Gamma.profile
        Qi = engine.Qi.profile
        Qe = engine.Qe.profile
        t  = engine.time

        self.log['time'].append(t)
        self.log['n'].append(n)
        self.log['pi'].append(pi)
        self.log['pe'].append(pe)
        self.log['Gamma'].append(G)
        self.log['Qi'].append(Qi)
        self.log['Qe'].append(Qe)

    def store_system(self,engine):
    # saves settings for reproducing runs

        # time step info lives here
        time_settings = {}
        time_settings['alpha']    = engine.alpha
        time_settings['dtau']     = engine.dtau
        time_settings['N_steps']  = engine.N_steps
        time_settings['N_prints'] = engine.N_prints
        time_settings['model']    = engine.model
        self.log['system'] = time_settings

        # profile info lives here

        profile_settings = {}
        profile_settings['N_radial']   = engine.N_radial
        profile_settings['rho_edge']   = engine.rho_edge
        profile_settings['n_core']     = engine.n_core  
        profile_settings['n_edge']     = engine.n_edge 
        profile_settings['pi_core']    = engine.pi_core
        profile_settings['pi_edge']    = engine.pi_edge
        profile_settings['pe_core']    = engine.pe_core
        profile_settings['pe_edge']    = engine.pe_edge
        profile_settings['Sn_width']   = engine.Sn_width      
        profile_settings['Sn_height']  = engine.Sn_height  
        profile_settings['Spi_width']  = engine.Spi_width   
        profile_settings['Spi_height'] = engine.Spi_height 
        profile_settings['Spe_width']  = engine.Spe_width    
        profile_settings['Spe_height'] = engine.Spe_height 
        profile_settings['source_n' ]   = engine.source_n
        profile_settings['source_pi']   = engine.source_pi
        profile_settings['source_pe']   = engine.source_pe
        self.log['profiles'] = profile_settings
        

    def export(self, fout='trinity_log.npy'):
        np.save(fout, self.log)
