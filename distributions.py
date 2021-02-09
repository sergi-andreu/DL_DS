import numpy as np
import random

def binarycolor(Z, c1, c2):

    Z = np.array(Z)
    Z = np.where(Z==1, c1, c2)

    return Z

def spiral(npoints, curv = 6, dim=2):
    
    sp = [[] for i in range(npoints)]
    z = np.zeros((npoints,))
    
    
    count = 0
    i=0
    
    while i < npoints:

        a = 1
        b = -1
        theta = 0.1 + random.random()

        ra = random.randint(0,1)
        sgn = 2*ra-1

        r = a + b*theta

        z[i] = ra

        sp[i] = [ sgn*r*np.cos(curv*theta), sgn*r*np.sin(curv*theta)]
            
        if np.absolute(sp[i][0]) >=0.1 or np.absolute(sp[i][1]) >=0.1:
            i = i+1

    sp = np.transpose(sp)

    if dim==3:
        sp = [sp[0], sp[1], np.zeros((npoints,))]
            
    return np.array(sp), np.array(z)


def generatorringsgap(n, sp = 0.15, dim=2):
    
    n2 = int(n/2)
    
    z = np.zeros((n,))
    
    r1 = np.array([(0.5-sp)*random.random() for i in range(n2)])
    r2 = np.array([((0.5*random.random()+0.5)*(1-sp))+sp for i in range(n2)])
    r = np.zeros((n,))
    r[0:n2] = r1
    r[n2::] = r2
    z[n2::] = 1

    theta = np.array([2*np.pi*random.random() for i in range(n)])
    
    x = np.multiply(np.cos(theta), r)
    y = np.multiply(np.sin(theta), r)
    
    if dim==3:
        x = [x,y, np.zeros((n,))]
    elif dim==2:
        x = [x,y]
    
    return x, z

if __name__ == "__main__":

    import matplotlib.pyplot as plt 
    import numpy as np

    #sp , z = spiral(100, curv=12)
    sp, z = generatorringsgap(100)

    fig, ax = plt.subplots()

    ax.set_aspect('equal', 'box')
    ax.tick_params(direction="in")
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_xticks([-1,0,1])
    ax.set_yticks([-1,0,1])

    ax.scatter(sp[0], sp[1], c=z)
    fig.tight_layout()
    plt.show()