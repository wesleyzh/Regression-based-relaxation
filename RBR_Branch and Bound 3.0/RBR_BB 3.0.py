"""
Replace the cost with the proability obtained by logistic regression

1, monitor the optimization progress, and outputs progress using defalt setting

2, pull out the objective value and incumbent solution

3, re-caculate the score for each variable and inject the incumbent solution with Regression-based Relaxation 

4, monitor the optimization progress, and outputs progress with RBR


implimented by weili zhang, 20150113
"""

from __future__ import division
import operator
import time
from numpy import * 
import math
from gurobipy import *
import matplotlib.pyplot as plt
import random
import copy
from itertools import izip

import randFCNF as CN

import Gurobi_solve_FCNF as Gurobi
 



#define the logistic regression, from R step wise
def glm(predictors):
        
    intercept = 8.324336e+00
    
    coefficients = [
        -4.804224e-02  ,  -5.515908e-03  ,  -8.788519e-02  ,  -1.494716e-04   ,  1.277916e-07  ,-5.431036e-01   ,  2.174355e+00  ,  -3.011054e+00    ,-2.380808e+00  ,   9.668389e-01  ,   7.649356e-01 , 8.780797e-01 ,   -2.151698e-01  ,   4.638464e-01  ,  -7.968709e-01   , -9.173211e-01 ,   -1.221596e-01 , 5.762761e+00 ,   -2.592250e+00 ,   -2.693845e+00   , -1.300506e+00  ,   1.839915e+00  ,   4.572218e+00 , -2.300354e+00  ,  -1.581288e+00   ,  3.187158e+00   ,  1.262547e+00   ,  1.403304e+00 
    ]

    #check the length of coefficients and predictors
    if len(coefficients) != len(predictors):
        sys.exit("The length of coefficients and predictors are not same {} {}".format(len(coefficients), len(predictors)))
        
    response1 = intercept
    
    for z in range(len(coefficients)):
        response1 += coefficients[z]*predictors[z]
    
        
    #probability that an arc is used in the optimal solution        
    response2 = math.exp(response1)/(1+math.exp(response1))  
    
    
    #if response2 < cutoff:
    #   response2 = 0
    #else:
    #    response2 = 1
        
    return response2

#get the values of predictors and call logistic regression model
def CharacterizeArcsIJK ( m,  numNodes, Nodes, commodities, arcs, varcost, fixedcost, requirements, LPSolution):
    
    networkSupply = 0
    
    for i in Nodes:
        if requirements[i,0] > 0:
            networkSupply = networkSupply + requirements[i,0]  
    
    arc_use = {}
    for i,j in arcs:
        fromNodeReq = requirements[i,0]
        toNodeReq = requirements[j,0]
        
        fromOutDegree = 0
        fromOutSupplyDegree = 0
        fromOutSupplyAmt = 0
        fromOutDemandDegree = 0
        fromOutDemandAmt = 0 
        
        for a,c in arcs.select(i,'*'):
            fromOutDegree = fromOutDegree + 1
            if requirements[c,0] > 0:
                fromOutSupplyDegree = fromOutSupplyDegree + 1
                fromOutSupplyAmt = fromOutSupplyAmt + requirements[c,0]
            elif requirements[c,0] < 0:
                fromOutDemandDegree = fromOutDemandDegree + 1
                fromOutDemandAmt = fromOutDemandAmt + requirements[c,0]                
        
        fromInDegree = 0
        fromInSupplyDegree = 0
        fromInSupplyAmt = 0
        fromInDemandDegree = 0
        fromInDemandAmt = 0 
        
        for a,c in arcs.select('*',i):
            fromInDegree = fromInDegree + 1
            if requirements[c,0] > 0:
                fromInSupplyDegree = fromInSupplyDegree + 1
                fromInSupplyAmt = fromInSupplyAmt + requirements[c,0]
            elif requirements[c,0] < 0:
                fromInDemandDegree = fromInDemandDegree + 1
                fromInDemandAmt = fromInDemandAmt + requirements[c,0]    
 
        toOutDegree = 0
        toOutSupplyDegree = 0
        toOutSupplyAmt = 0
        toOutDemandDegree = 0
        toOutDemandAmt = 0 
        
        for a,c in arcs.select(j,'*'):
            toOutDegree = toOutDegree + 1
            if requirements[c,0] > 0:
                toOutSupplyDegree = toOutSupplyDegree + 1
                toOutSupplyAmt = toOutSupplyAmt + requirements[c,0]
            elif requirements[c,0] < 0:
                toOutDemandDegree = toOutDemandDegree + 1
                toOutDemandAmt = toOutDemandAmt + requirements[c,0]                
        
        toInDegree = 0
        toInSupplyDegree = 0
        toInSupplyAmt = 0
        toInDemandDegree = 0
        toInDemandAmt = 0 
        
        for a,c in arcs.select('*',j):
            toInDegree = toInDegree + 1
            if requirements[c,0] > 0:
                toInSupplyDegree = toInSupplyDegree + 1
                toInSupplyAmt = toInSupplyAmt + requirements[c,0]
            elif requirements[c,0] < 0:
                toInDemandDegree = toInDemandDegree + 1
                toInDemandAmt = toInDemandAmt + requirements[c,0]    
        
        #f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(problemID, seed, numNodes, networkSupply, i,j,arcsUsed[i,j], solution[i,j,0], LPSolution[i,j,0], varcost[i,j,0], fixedcost[i,j], fromNodeReq, toNodeReq, fromOutDegree, fromOutSupplyDegree, fromOutSupplyAmt, fromOutDemandDegree, fromOutDemandAmt, fromInDegree,  fromInSupplyDegree, fromInSupplyAmt, fromInDemandDegree, fromInDemandAmt,toOutDegree, toOutSupplyDegree, toOutSupplyAmt, toOutDemandDegree, toOutDemandAmt, toInDegree,  toInSupplyDegree, toInSupplyAmt, toInDemandDegree, toInDemandAmt)) 
        
        fromNodeType = (fromNodeReq > 0) - (fromNodeReq < 0)
        #toNodeType = (toNodeReq > 0) - (toNodeReq < 0)
        #fRratio = fromNodeReq/networkSupply
        #tRratio = toNodeReq/networkSupply
        #fOutS_dratio =fromOutSupplyDegree/numNodes;
        #fInD_dratio =fromInDemandDegree /numNodes;
        #fInS_dratio =fromInSupplyDegree/numNodes;
        #fOutD_dratio = fromOutDemandDegree/numNodes    
        #tInS_dratio =toInSupplyDegree/numNodes        
        
        AveSup = networkSupply/numNodes
        density = len(arcs)/(numNodes*(numNodes-1))
        XLPSolutionij ,   Xvarcostij,    Xfixedcostij = LPSolution[i,j,0], varcost[i,j,0], fixedcost[i,j]
        
        
        fromNodeType = int(fromNodeReq > 0) - int(fromNodeReq < 0)
        toNodeType = int(toNodeReq > 0) - int(toNodeReq < 0)        
        fRratio = fromNodeReq/networkSupply
        tRratio = toNodeReq/networkSupply
        FOSratio = fromOutSupplyAmt/networkSupply
        FODratio = fromOutDemandAmt/networkSupply
        FISratio = fromInSupplyAmt/networkSupply
        FIDratio =fromInDemandAmt/networkSupply
        TOSratio = toOutSupplyAmt/networkSupply
        TODratio = toOutDemandAmt/networkSupply
        TISratio = toInSupplyAmt/networkSupply
        TIDratio = toInDemandAmt/networkSupply
        LPratio = XLPSolutionij/networkSupply
        
        FOdegree_ratio = fromOutDegree/numNodes
        FSdegree_ratio = fromOutSupplyDegree/numNodes
        FODdegree_ratio = fromOutDemandDegree/numNodes
        FIdegree_ratio = fromInDegree/numNodes
        FISdegree_ratio = fromInSupplyDegree/numNodes
        FIDdegree_ratio = fromInDemandDegree/numNodes
        TOdegree_ratio = toOutDegree/numNodes
        TOSdegree_ratio = toOutSupplyDegree/numNodes
        TODdegree_ratio = toOutDemandDegree/numNodes
        TIdegree_ratio = toInDegree/numNodes
        TISdegree_ratio = toInSupplyDegree/numNodes
        TIDdegree_ratio = toInDemandDegree/numNodes        
        
        sqrNode	= numNodes**(-2)
        expNode	 = math.exp(-numNodes)
        
        #new predictors
        numarcs = len(arcs)   #number of directed arcs
        fcratio = Xfixedcostij/float(Xvarcostij)   #ration between fixed cost and variable cost
        fromNodeType0 = int(fromNodeType == 0)     #from node type equals 0
        fromNodeType1 = int(fromNodeType == 1)     #from node type equals 1
        toNodeType0 = int(toNodeType == 0)
        toNodeType1 = int(toNodeType == 1)
        LPbinary = int(LPratio != 0)   #binary version of linearized relaxation solution
        
        #predictors are fixed by step wise from R
        predictors = [  
        numNodes	,
        numarcs	,
        Xvarcostij,
        Xfixedcostij,
        fcratio,
        
        fromNodeType0	,
        fromNodeType1	,
        toNodeType0	,
        toNodeType1     ,
        
        fRratio	,
        FOSratio,
        FODratio	,
        FISratio	,
        FIDratio	,
        TOSratio	,
        TODratio        ,
        TISratio	,
        
        LPratio	,
        FOdegree_ratio	,
        FSdegree_ratio,
        
       
        FODdegree_ratio	,
        FISdegree_ratio	,
        FIDdegree_ratio	,

        TOdegree_ratio	,
        TODdegree_ratio	,
        TISdegree_ratio	,
        TIDdegree_ratio	,
        
        LPbinary,
        ]
    
        arc_use[i,j] = glm(predictors)
        
    return arc_use


class Regression_based_relaxation (object):  #class of regression-based relaxation 
    
    def _init_(self):
	self.score = {}        # probability for each variable
	self.solution = {}     # regression-based relaxation solution
	self.objective = 0     # regression-based relaxation objective value
	
    def update_score (self, m,  LPSolution, numNodes, Nodes, commodities, arcs, varcost, fixedcost, requirements):          #update the score
	
	
	self.score = CharacterizeArcsIJK ( m, numNodes, Nodes, commodities, arcs, varcost, fixedcost, requirements, LPSolution)
	
    def solve (self, m):   # solve the regression-based relaxation and get the solution and obj
	
	global arcs,  varcost, fixedcost
	
	K = 1
	
	RBRflow, RBRvarcost, RBRdecision, RBRfixedcost = {}, {}, {}, {}
	
	model = m.copy()
	for i,j in arcs:
	    for k in range(K):
		RBRflow[i,j,k] = model.getVarByName('flow_%s_%s_%s' % (i, j, k))        #create dictionary of continuous variables
		RBRvarcost[i,j,k] = flow[i,j,k].Obj                                 #extract varcosts from model   
	    RBRdecision[i,j] = model.getVarByName('decision_%s_%s' % (i,j))             #create dictionary of binary variables
	    RBRfixedcost[i,j] = decision[i,j].Obj                                   #extract fixed costs from model     
	    
			
	for i,j in arcs:
	    for k in range(K):
		RBRflow[i,j, k].Obj = -math.log(self.score[i,j] )  #replace the variable cost with the probabilty if self.score[i,j]>0 else 0.0000001
		RBRdecision[i,j].Obj = 0.0   #fixed costs are 0 for all decision variables, not remove because of the capacity constraint
		RBRdecision[i,j].ub = 1.0
		RBRdecision[i,j].lb = 0.0
	
	
	#RBRstart = time.clock()
	
	model.optimize()   #get optimal solution for logistic regression based linearized FCNF
	
	#RBRtime = time.clock() - RBRstart
	
	RBRx = model.getAttr('x', RBRflow)  # the solution vector, x_ijk
	RBRy = model.getAttr('x', RBRdecision)  # the solution vector, y_ijk
	
	for i,j in arcs:
	    if quicksum(RBRx[i,j,k] for k in xrange(K)) > 0.0:         #if there is positive flow on arc (i,j), 
		RBRy[i,j] = 1                                                     #  set y_ij = 1 
	    else:                                                                
		RBRy[i,j] = 0                                                     #  otherwise, set y_ij = 0        
    
	RBRobj = quicksum(varcost[i,j,k]*RBRx[i,j,k] + fixedcost[i,j]*RBRy[i,j] for i,j in arcs)  #calculate the true objective cost
	
	self.solution = dict(RBRx.items() + RBRy.items()) 
	self.objective = RBRobj
       
		
	
def mycallback(model, where):
    
    global numNodes, Nodes, commodities, arcs, varcost, fixedcost, requirements
    global start_time, time_limit, RBR_frequence
    
    if where == GRB.callback.MIPNODE:  
	
	mipnode_status = model.cbGet(GRB.callback.MIPNODE_STATUS)
	if mipnode_status != GRB.OPTIMAL:
	    return
	
	# Currently exploring a MIP node
	#print('*** New node')
	#if model.cbGet(GRB.callback.MIPNODE_STATUS) == GRB.status.OPTIMAL:
	    #x = model.cbGetNodeRel(model.getVars())
	    #model.cbSetSolution(model.getVars(), x)
	objbst = model.cbGet(GRB.callback.MIPNODE_OBJBST)  #Current best objective
	objbnd = model.cbGet(GRB.callback.MIPNODE_OBJBND)  #Current best objective bound
	nodecnt = model.cbGet(GRB.callback.MIPNODE_NODCNT)
	
	#create an RBR class at here
	RBR = Regression_based_relaxation()	
	
	if nodecnt == 0:
	    
	    
	    #get the node relaxation solution
	    Out_time = time.clock()   #time to pull out branching node information
	    
	    flow_vars = [v for v in model.getVars() if v.VType in ('C')]	#continuous flow variables
	    xs = model.cbGetNodeRel(flow_vars)   #Retrieve values from the node relaxation solution at the current node.
	    flow_vars_LP = dict([(v.varName, x) for v, x in izip(flow_vars, xs)])
	    
	    Out_time = time.clock() - Out_time
	    
	    flow_value = {}
	    
	    for i,j in arcs:
		for k in range(K):
		    flow_value [i,j,k] = flow_vars_LP ['flow_%s_%s_%s' % (i, j, k)]
		
	    #update the RBR
	    
	    RBRupdate_time = time.clock()
	    
	    RBR.update_score (model,  flow_value, numNodes, Nodes, commodities, arcs, varcost, fixedcost, requirements)
	    
	    RBRupdate_time = time.clock() - RBRupdate_time
	    
	    #solve the RBR
	    RBRsolve_time = time.clock()
	    
	    RBR.solve(model)
	    
	    RBRsolve_time = time.clock() - RBRsolve_time
	    
	    #inject solution from RBR to gurobi
	    
	    in_time = time.clock()
	    
	    #transfer RBR.solution to a list to desired variables
	    var = model.getVars()
	    
	    if RBR.objective < objbst:
		for i,j in arcs:
		    for k in range(K):
			flowvar = model.getVarByName('flow_%s_%s_%s' % (i, j, k))        #create dictionary of continuous variables
			model.cbSetSolution(flowvar, RBR.solution[i,j,k])                #set the new solution  
		    decvar = model.getVarByName('decision_%s_%s' % (i,j))		 #create dictionary of binary variables
		    model.cbSetSolution(decvar, RBR.solution[i,j])                                   
		    
		
		model.update()
	    
	    in_time = time.clock() - in_time
	    
	else:
	    Out_time, RBRupdate_time, RBRsolve_time, in_time, RBR.objective = None, None, None, None, None
	
	searchtime = time.clock() - start_time
	
	if searchtime > time_limit:
	    return
	
	#print('%d %g %g %g %g' % (int(nodecnt), searchtime, objbst, objbnd, RBR.objective))
	
	q = open('RBR_Node_Up_Low_{}_{}.txt'.format(nodeCnt, seed),'a')
	q.write('{}, {}, {}, {}, {}, {}, {}, {}, {} \n'.format(nodecnt, searchtime, Out_time, RBRupdate_time, RBRsolve_time, in_time, objbst, objbnd, RBR.objective))
	q.close()	
		
		
#FCNF parameters
nodemin = 800
nodemax = 800
nodelist = [100, 300, 900]
supplyPct = 0.2
demandPct = 0.2 
rhsMin = 1000
rhsMax = 2000
cMin = 0 
cMax = 10
fMin = 20000
fMax = 60000
K = 1 

time_limit = 3600   #total running time
RBR_frequence = 8 #frequence to run RBR


#Gurobi model set
m = Model('FCNF')
m.setParam( 'OutputFlag', 0) 
m.setParam( 'LogToConsole', 0 )
m.setParam( 'LogFile', "" )   
m.params.threads = 7
m.params.NodefileStart = 0.5
m.params.timeLimit = time_limit
m.params.RINS = 0 #set the frequency of RINS
m.params.Heuristics = 0 #remove heuristics

#seed
seedMin = 200
seedMax = 200

seedlist = [113, 121, 123, 127, 200, 1019, 1027, 1051, 1055, 1059]

#s = open('Confusion Matrix_{}.txt'.format(nodeCnt),'w')
#s.close()

arcs=[]
decision = {}
flow = {}
varcost = {}
fixedcost = {}

f = open('FCNF RBR.txt','a')
f.close()

g = open('FCNF Characteristics.txt','a')
g.close()

#global numNodes, Nodes, commodities, arcs, varcost, fixedcost, requirements

for nodeCnt in nodelist:
    #g = open('FCNF RBR_{}.txt'.format(nodeCnt,same_time),'w')
    #g.close()    

    #for seed in range(seedMin,seedMax+10,10):
    for seed in seedlist:
        
	print "new problem node %s seed %s" % (nodeCnt, seed)
	
        m.reset()
        for v in m.getVars():
            m.remove(v)
        for c in m.getConstrs():
            m.remove(c)     
    
        arcs[:]=[]
        decision.clear()
        flow.clear()
        varcost.clear()
        fixedcost.clear() 
        
        random.seed(seed)
        
        result = CN.FCNFgenerator(seed, m, nodeCnt, supplyPct, demandPct, rhsMin, rhsMax, cMin, cMax, fMin, fMax, K, arcs)
        RQ = result[0]
        
        Nodes = range(0,nodeCnt)
	commodities = range(0,K)
	arcs = tuplelist(arcs)
	
	for i,j in arcs:
	    for k in range(K):
		flow[i,j,k] = m.getVarByName('flow_%s_%s_%s' % (i, j, k))        #create dictionary of continuous variables
		varcost[i,j,k] = flow[i,j,k].Obj                                 #extract varcosts from model   
	    decision[i,j] = m.getVarByName('decision_%s_%s' % (i,j))             #create dictionary of binary variables
	    fixedcost[i,j] = decision[i,j].Obj                                   #extract fixed costs from model     
 
 
	numNodes, Nodes, commodities, arcs, varcost, fixedcost, requirements = nodeCnt, Nodes, commodities, arcs, varcost, fixedcost, RQ,
        
        commodities = range(0,K)
        arcs=tuplelist(arcs)
        noV = len(arcs)
        
        
        #g = open('OSA_PROBS_20131511_RANDGEM.txt','a')
        #g.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'.format(problemID, seed, nodeCnt, len(arcs), m.NumVars,m.NumBinVars, m.NumConstrs, supplyPct, demandPct, rhsMin, rhsMax, cMin, cMax, fMin, fMax, K))
        #g.close() 
        
       
        #fixed cost to variable cost ratio
        ratio = quicksum(fixedcost[i,j] for i,j in arcs)/quicksum(varcost[i,j,k] for i,j in arcs)
        
        #density
        density = noV/(nodeCnt*(nodeCnt-1))
        
        #percentage of supply nodes and demand nodes
        supplynode = 0
        supply = 0
        demandnode = 0
        for key, value in RQ.items():
            if value > 0 :
                supplynode += 1
                supply += value
            elif value < 0 :
                demandnode += 1
        
        #average supply
        supplyratio = supply/float(supplynode)
        
        g = open('FCNF Characteristics.txt','a')
        g.write('{}, {}, {}, {}, {}, {}, {}, {} \n'.format(seed, nodeCnt, noV, density, supplyratio, ratio, supplynode/float(nodeCnt), demandnode/float(nodeCnt)))
        g.close()        
        
	q = open('RBR_Node_Up_Low_{}_{}.txt'.format(nodeCnt, seed),'w')
	q.close()            
	
	try: 
	    #change back to the original FCNF
	    for i,j in arcs:
		decision[i,j].vType = GRB.BINARY
		decision[i,j].Obj = fixedcost[i,j]
		for k in range(K):
		    flow[i,j,k].Obj = varcost[i,j,k]  #replace the variable cost with the probabilty             
	    
	    global start_time
	    start_time = time.clock()
	    
	    m.optimize(mycallback)   #get optimal solution for testing...
	    #m.optimize()
	    
	    if  m.status == 2 or m.status == 9:
		GRBobjval = m.objval
		GRBRuntime = m.Runtime
		GRBMIPgap = m.MIPgap
		print "RBR", nodeCnt, seed, GRBobjval, GRBRuntime, GRBMIPgap
		
	    f = open('FCNF RBR.txt','a')
	    f.write('{}, {}, {}, {}, {} \n'.format( nodeCnt, seed, GRBobjval, GRBRuntime, GRBMIPgap))
	    f.close()                       
		    
	except:
	    print 'Gurobi Error reported' 
	    
	
	m.reset()
	
	#use gurobi to solve the same problem 
	rins = -1
	heuristics = 0
	Gurobi.main(m, nodeCnt, seed, rins, heuristics)