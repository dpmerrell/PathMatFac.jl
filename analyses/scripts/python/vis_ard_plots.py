
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.stats import norm
import numpy as np

mpl.rc("font", family="serif")

def ard(x, alpha=1e-3, beta=1e-3):
    return (alpha + 0.5)*np.log(beta + 0.5*x*x) - (alpha+0.5)*np.log(beta)


if __name__=="__main__":
   
    #####################################
    # Lineplot
    ##################################### 
    x = np.linspace(-5.0, 5.0, 10001)

    alpha0 = 2.001
    #beta0 = (alpha0 - 1) 
    beta0 = 1.00 
    y_ard = ard(x, alpha=alpha0, beta=beta0)
    y_l2 = 0.5*x*x
    y_l1 = np.abs(x)    

    fig, axs = plt.subplots(figsize=(5,3))
    plt.subplot()
    plt.title("Regularizers")
    plt.plot(x, y_l1, linewidth=1.0, label="L1")
    plt.plot(x, y_l2, linewidth=1.0, label="L2") 
    plt.plot(x, y_ard, linewidth=1.0, label="ARD")
    plt.plot(x, np.zeros_like(x), "--", color="k", linewidth=0.5)
    plt.plot([0,0], [-5,10], "--", color="k", linewidth=0.5)
    plt.xlim([-5, 5])
    plt.ylim([-1,15])
    plt.legend()    
    plt.xticks([-1,0,1], [-1,0,1])

    plt.tight_layout()
    plt.savefig("regularizers-line-plot.png", dpi=300)

    #########################################
    # Contour plots
    #########################################
    
    #y = np.linspace(-5.0, 5.0, 10001)
    #X, Y = np.meshgrid(x,y)
    #fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9,3))
    #
    #Z_l1 = np.abs(X) + np.abs(Y)
    #axs[0].contourf(Z_l1, 20, cmap="Greys_r")
    #axs[0].set_xlabel("L1")
    #axs[0].set_xticks([])
    #axs[0].set_yticks([])

    #Z_l2 = 0.5*(X*X + Y*Y)
    #axs[1].contourf(Z_l2, 20, cmap="Greys_r")
    #axs[1].set_xlabel("L2")
    #axs[1].set_xticks([])
    #axs[1].set_yticks([])

    #Z_ard = ard(X, alpha=alpha0, beta=beta0) + ard(Y, alpha=alpha0, beta=beta0)
    #axs[2].contourf(Z_ard, 20, cmap="Greys_r")
    #axs[2].set_xlabel("ARD")
    #axs[2].set_xticks([])
    #axs[2].set_yticks([])

    #plt.tight_layout()
    #plt.savefig("regularizers-contour-wider.png")

