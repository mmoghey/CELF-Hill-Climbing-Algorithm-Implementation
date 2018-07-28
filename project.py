import snap
import numpy as np
import random
import matplotlib.pyplot as plt

import snap
import numpy as np
import random
import matplotlib.pyplot as plt
import sets
from sets import Set
import scipy
from scipy import stats
from collections import Counter

# Setup
erdosRenyi = None
smallWorld = None
collabNet = None


# Problem 1.1
def genErdosRenyi(N=5242, E=14484):

    Graph = snap.TUNGraph.New()
    
    for i in range (0,N):
        Graph.AddNode(i)
    
    adj = np.zeros ((N,N))
    count = 0

    while (count <= E+1):
        src = random.randint(0,N-1)
        dest = random.randint(0,N-1)
        if (src != dest):
            if(adj[src][dest] == 0):
                adj[src][dest] = 1
                
        count = count + 1
        Graph.AddEdge(src,dest)
    
    Count = snap.CntUniqUndirEdges(Graph)
    #print Count
    return Graph


def genCircle(N=5242):

    Graph = snap.TUNGraph.New()

    for i in range (0,N):
        Graph.AddNode(i)

    #print Graph.GetNodes()

    for i in range (0,N):
    #print i%N, (i+1)%N, (i+2)%N
        Graph.AddEdge(i%N, (i+1)%N)
    
    Count = snap.CntUniqUndirEdges(Graph)
   
    return Graph


def connectNbrOfNbr(Graph, N=5242):
    print Graph.GetNodes()

    for i in range (0,N):
    #print i%N, (i+1)%N, (i+2)%N
        Graph.AddEdge(i%N, (i+2)%N)
    
    Count = snap.CntUniqUndirEdges(Graph)
    

    return Graph


def connectRandomNodes(Graph, M=8000):
    count = 0
    N = Graph.GetNodes()
    adj = np.zeros ((N,N))

    for i in range (0,N):
        for j in range(0,N):
            if (Graph.IsEdge(i,j)):
                adj[i][j] = 1

    while (1):
        src = random.randint(0,N)
        dest = random.randint(0,N)
        if (src%N != dest%N):
            if(adj[src%N][dest%N] == 0):
                adj[src%N][dest%N] = 1
                Graph.AddEdge(src%N,dest%N)
        
                count = count + 1
                if (count == M):
                    break

    Count = snap.CntUniqUndirEdges(Graph)
    return Graph


def genSmallWorld(N=12008, E=32016):

    Graph = genCircle(N)
    Graph = connectNbrOfNbr(Graph, N)
    Graph = connectRandomNodes(Graph, 8000)
    return Graph


def loadCollabNet(path):

    Graph = snap.LoadEdgeList(snap.PUNGraph, "C:\Users\manas\Documents\eBooks\Advanced Databases\project\ca-HepPh\CA-HepPh.txt", 0, 1)
    for NI in Graph.Nodes():
        if (Graph.IsEdge(NI.GetId(), NI.GetId())):
            Graph.DelEdge(NI.GetId(), NI.GetId())

    return Graph


def getDataPointsToPlot(Graph):

    #find out the max out degree
    l1 = []
    
    for NI in Graph.Nodes():
        l1.append(NI.GetOutDeg())

    maxOutDegree = max(l1)

    # populate list l2 with the count of nodes with out degree as index ( for eg., l2[outdegree] = count)
    l2 = []

    #allocate the memory first
    for x in range(0, maxOutDegree+1):
        l2.append(snap.CntDegNodes(Graph, x))

    Y = np.array(l2)
    X = np.array(list(range(0,maxOutDegree+1)))

    return X, Y

def IC(Graph, A):
    #print ('In IC')
    finalActivated = set()
    for node in A:
        tempAct = {node}
        tempAct1 = set()
        notAct = set()
        activated = {node}

        while len(tempAct):
            for n in tempAct:
                NI = Graph.GetNI(n)
                outDeg = NI.GetOutDeg()
                for i in range(outDeg):
                    NbrId = NI.GetNbrNId(i)
                    r = random.random()
                    if NbrId not in notAct and NbrId not in activated:
                        if r < 0.01:
                            activated.add(NbrId)
                            tempAct1.add(NbrId)
                    else:
                        notAct.add(NbrId)
                tempAct = tempAct1
                tempAct = set()
                
        finalActivated.update(activated)
    #print ('out of IC')
    return (len(finalActivated))
    
    
def getKey(item):
    return item[1]
  
def CELF (Graph, k):
    print ('\n in CELF\n')
    V = set()
    
    for NI in Graph.Nodes():
            V.add(NI.GetId())
    
    Q = [] 
    S = set()
    inf = 0
    
    msg = 1
    iter = 2
    nid = 0
    
    for u in V:
        temp_l = []
        temp_l.append(u) #u
        temp_l.append(IC(Graph, {u})) # msg
        temp_l.append(0) #iter
        
        Q.append(temp_l)
        
    Q = sorted (Q, key=getKey)
        
    while len(S) < k :
        u = Q[0]
        if (u[iter]) == len(S):
            S.add (u[nid])
            temp_Q = []
            for l in Q:
                if l[0] != u[nid]:
                    temp_Q.append (l)
            Q = temp_Q
        
        else:
            u[msg] = IC(Graph, S | {u[0]}) - IC(Graph, S)
            u[iter] = len(S)
            Q = sorted (Q, key=getKey)
            
            max_inf = 0
            mi_node = 0
            for node in V:
                temp = IC(Graph, S | {node}) - IC(Graph, S)
                if (temp > max_inf):
                    max_inf = temp
                    mi_node = node
                    #inf = inf + max_inf
            
            S.add(mi_node)
			inf = inf + max_inf
            
    return S, inf
                
def call_CELF (Graph):
    print ('\n In call CELF \n')

        
    k_range = 30
    Y=[]
    
    for k in range ( k_range):
        print ('k : ', k)
        A, max_inf = CELF (Graph, k)
        Y.append(max_inf)
        
    return Y 
    
def degreeCentrality(Graph, k):
    I = set()
    result_dict = dict()
    for NI in Graph.Nodes():
        result_dict[NI.GetId()] = NI.GetDeg()
        result = dict(Counter(result_dict).most_common(k))
                
    for key, value in result.iteritems():
        I.add(key)

    return I, IC(Graph, I)
    
def call_degree (Graph):
    print ('\n In call degree \n')
    k_range = 30
    Y = []
    
    for k in range ( k_range):
        print ('k : ', k)
        A, max_inf = degreeCentrality (Graph, k)
        Y.append(max_inf)
        print ('max inf: ', max_inf)
    
    print (len(Y))
    return Y
    
    
def randomGraph(Graph, k):
    V = []
    
    for NI in Graph.Nodes():
        V.append(NI.GetId()) 
    
    A = [V[ random.randint(0, Graph.GetNodes())] for i in range(k)]
    return A, IC (Graph, A)
    
def call_random (Graph):
    print ('\n In call ramdom \n')
    k_range = 30

    Y = []
    
    for k in range ( k_range):
        print ('k : ', k)
        A, max_inf = randomGraph (Graph, k)
        Y.append(max_inf)
        print ('max inf', max_inf)
    
    print (len(Y))
    return Y
    

    
#Q_5()
    
    

def Q1_3():
  
    global smallWorld, collabNet
    
    smallWorld = genSmallWorld(12008, 32016)
    collabNet = loadCollabNet("ca-HepPh.txt")
        
    x_smallWorld, y_smallWorld = getDataPointsToPlot(smallWorld)
    plt.loglog(x_smallWorld, y_smallWorld, linestyle = 'dashed', color = 'r', label = 'Small World Network')

    x_collabNet, y_collabNet = getDataPointsToPlot(collabNet)
    plt.loglog(x_collabNet, y_collabNet, linestyle = 'dotted', color = 'b', label = 'Collaboration Network')

    plt.xlabel('Node Degree (log)')
    plt.ylabel('Proportion of Nodes with a Given Degree (log)')
    plt.title('Degree Distribution of Small World, and Collaboration Networks')
    plt.legend()
    plt.show()
	
	
	
	



# Execute code for Q1.1
Q1_3()




def calcClusteringCoefficient(Graph):

    C = 0.0
    ei = 0
    V = 0
    for NI in Graph.Nodes():
        for nid1 in NI.GetOutEdges():
            for nid2 in NI.GetOutEdges():
                if (nid1 != nid2):
                    if (Graph.IsEdge(nid1, nid2)):
                        ei = ei + 1
                        
        ki = NI.GetDeg()
        ei = ei / 2 
        if (ki >= 2):
            Ci =  (2 * ei) / ((1.0) * (ki * (ki - 1)))
        #print ei, Ci
        else:
            Ci = 0
            C = C + Ci
        V = V + 1
    print C, V
    C = C /(1.0 * V)
                    
   return C

def Q1_4():
    C_smallWorld = calcClusteringCoefficient(smallWorld)
    C_collabNet = calcClusteringCoefficient(collabNet)
    
    
    print('Clustering Coefficient for Small World Network: %f' % C_smallWorld)
    print('Clustering Coefficient for Collaboration Network: %f' % C_collabNet)



    
def Q_5 ():
    print ('\n in question 5')

    global smallWorld, collabNet
    k_range=30

    X = []
    for i in range(1,k_range+1):
        X.append(i)

    
    #smallWorld = genSmallWorld(12008, 32016)
    #collabNet = loadCollabNet("ca-HepPh.txt")
    
    print ('CELF small world')
    C_S = call_CELF(smallWorld)
    
    print ('CELF CollabNet')
    C_C = call_CELF(collabNet)
    
    print ('Degree SmallWorld')
    D_S = call_degree(smallWorld)
    print (D_S)
    
    print('Degree CollabNet')
    D_C = call_degree(collabNet)
    print (D_C)
    
    print('random smallWorld')
    R_S = call_random(smallWorld)
    print(R_S)
    
    print('random CollabNet')
    R_C = call_random(collabNet)
    print (R_C)
    
    
    plt.plot(X, C_C , linestyle = 'dotted', color = 'b', label = 'Collaboration Network')
    plt.plot(X, D_C,  linestyle = 'dashed', color = 'r', label = 'Degree Centrality')
    plt.plot(X, R_C , linestyle = 'dotted', color = 'g', label = 'Random Node')
   
    plt.xlabel('Set size (k)')
    plt.ylabel('F(Sk) (active set size)')
    plt.title('Collab Net')
    
    plt.legend()
    plt.show()
    
    plt.plot(X, C_S, linestyle = 'dashed', color = 'r', label = 'Small World Network')
    plt.plot(X, D_S , linestyle = 'dotted', color = 'b', label = 'Degree Centrality')
    plt.plot(X, R_S, linestyle = 'dashed', color = 'g', label = 'Random Node')
  
    plt.xlabel('Set size (k)')
    plt.ylabel('F(Sk) (active set size')
    plt.title('Small World')    
    plt.legend()
    plt.show()

Q1_4()
Q_5()

        
    
    

    
            

    