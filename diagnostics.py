import matplotlib.pyplot as plt

# plot the density and flux profile
def diagnostic_1(density, Gamma, Time):

    plt.subplot(1,2,1)
    density.plot(label='T = {:.2e}'.format(Time))
    plt.subplot(1,2,2)
    Gamma.plot(label='T = {:.2e}'.format(Time))
