"""
Solve fixed charge network flow problem with gurobi

using callback to track the upper bound and lower bound

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

import randFCNF as CN


def mycallback(model, where):
    
    global GRBnodeCnt, GRBseed, GRBstart_time
    
    """
    print the upper bound and lower bound at each searching node
    
    """
    if where == GRB.callback.MIPNODE:
        
        mipnode_status = model.cbGet(GRB.callback.MIPNODE_STATUS)
	if mipnode_status != GRB.OPTIMAL:
	    return

        
        # MIP node callback
        objbst = model.cbGet(GRB.callback.MIPNODE_OBJBST)  #Current best objective
        objbnd = model.cbGet(GRB.callback.MIPNODE_OBJBND)  #Current best objective bound        
        nodecnt = model.cbGet(GRB.callback.MIPNODE_NODCNT) #Current node explored
        
        searchtime = time.clock()- GRBstart_time
        
        #print nodecnt, objbst, objbnd
        q = open('GRB_RINS{}_Node_Up_Low_{}_{}.txt'.format(GRBRINS, GRBnodeCnt, GRBseed),'a')
        q.write('{}, {},{},  {} \n'.format(nodecnt, searchtime, objbst, objbnd))
        q.close()          
        

def main(m, nodeCnt, seed, RINS, HEURISTICS):
    
    global GRBnodeCnt, GRBseed, GRBRINS, GRBHEUR
    
    GRBnodeCnt, GRBseed, GRBRINS, GRBHEUR = nodeCnt, seed, RINS, HEURISTICS
    
    GRBmodel = m.copy()
   
    GRBmodel.reset()
    
    GRBmodel.params.RINS = GRBRINS #set the frequency of RINS
    GRBmodel.params.Heuristics = GRBHEUR #set the running time of heuristics

    q = open('GRB_RINS{}_Node_Up_Low_{}_{}.txt'.format(GRBRINS, GRBnodeCnt, GRBseed),'w')
    q.close()            
    
    try: 
        
	global GRBstart_time
	GRBstart_time = time.clock()                 

        GRBmodel.optimize(mycallback)   #get optimal solution for testing...
        
        if  GRBmodel.status == 2 or GRBmodel.status == 9:
            GRBobjval = GRBmodel.objval
            GRBRuntime = GRBmodel.Runtime
            GRBMIPgap = GRBmodel.MIPgap
            print "GRB", GRBnodeCnt, GRBseed, GRBobjval, GRBRuntime, GRBMIPgap
            
        f = open('FCNF GRB RINS {}.txt'.format(GRBRINS),'a')
        f.write('{}, {}, {}, {}, {} \n'.format( GRBnodeCnt, GRBseed, GRBobjval, GRBRuntime, GRBMIPgap))
        f.close()                       
                
    except:
        print 'Gurobi Error reported'    
               
                 

if __name__ == "__main__":
    main()