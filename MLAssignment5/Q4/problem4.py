import numpy as np


def JTA(test = True):
    n = 5
    potentials = []
    messages = []

    def normalizeFn(M):
        return M/np.sum(M)

    if(test):
        A = np.array([[0.1,0.7],[0.8,0.3]])
        B = np.array([[0.5,0.1],[0.1,0.5]])
        C = np.array([[0.1,0.5],[0.5,0.1]])
        D = np.array([[0.9,0.3],[0.1,0.3]])
        potentials.append(A)
        potentials.append(B)
        potentials.append(C)
        potentials.append(D)
        
    else:
        for i in range(0,n-1):
            A = np.random.rand(2,2)
            potentials.append(A)

    pairwise_marginals = []
    for i in range(0,n-2):
        messages.append(np.ones([2,2]))

    #Left-to-right message passing
    for i in range(0,n-2):
        messages[i] = np.sum(potentials[i],0)
        potentials[i+1] = np.multiply(potentials[i+1].T, messages[i].T).T
        

    #Right-to-left message passing
    for i in range(n-3,-1,-1):
        old = messages[i]
        messages[i] = np.sum(potentials[i+1],1)
        potentials[i] = np.multiply(potentials[i], (np.divide(messages[i].T,old)))


    #Compute pairwise marginals

    for i in range(0,n-1):
        pairwise_marginals.append(normalizeFn(potentials[i]))

    print(np.vstack(pairwise_marginals))



if __name__ == "__main__":
    JTA() #Pass test = False to initialise potentials randomly





