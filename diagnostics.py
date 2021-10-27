import matplotlib.pyplot as plt

# plot the density and flux profile
#    OLD, to be deleted
def diagnostic_1(density, Gamma, Time):

    plt.subplot(1,2,1)
    density.plot(label='T = {:.2e}'.format(Time))
    plt.subplot(1,2,2)
    Gamma.plot(label='T = {:.2e}'.format(Time))


# plot the density and flux profile
class diagnostic_2:
    
    def __init__(self,win=(8,4)):
        fig, axs = plt.subplots(1,2,figsize=win)
        self.axs = axs

    def plot(self, density, Gamma, Time):

        tlabel = 'T = {:.2e}'.format(Time)
        self.axs[0].plot( density.axis, density.profile,'.-',label=tlabel )
        self.axs[1].plot( Gamma.axis,   Gamma.profile,  '.-',label=tlabel )

    def label(self, title=''):

        a0,a1 = self.axs
        
        a0.set_title(title)
        a0.set_ylim(0,4.2)
        a0.grid()
        
        a1.set_title('Gamma(rho)')
        a1.legend()
        a1.grid()
