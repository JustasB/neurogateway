# -*- coding: utf-8 -*-
import os
import glob
import pickle
from utils import Utils
import numpy as np
import pdb 
# The readin flag when set enables the wiring to be read in from pre-existing 
# pickled files with rank specific file names.
from mpi4py import MPI
import neuron as h
MPI.COMM = MPI.COMM_WORLD
utils = Utils(NCELL=190,readin=1) #Read in the pre-established wiring files to speed up the simulation.

print MPI.COMM.Get_rank(), 'mpi rank'
if MPI.COMM.Get_rank()==0:
    pass
else:
    info_swc=utils.gcs(utils.NCELL)
utils.wirecells()#wire cells on different hosts.
#utils.h('forall{ for(x,0){ uninsert xtra}}')   #mechanism only needed for wiring cells not for simulating them. 

utils.global_icm=utils.matrix_reduce(utils.icm)
utils.global_ecm=utils.matrix_reduce(utils.ecm)
utils.global_visited=utils.matrix_reduce(utils.visited)
from rigp import NetStructure

if utils.COMM.rank==0:
    utils.dumpjson_graph()
else:
    utils.h('forall{ for(x,0){ uninsert xtra}}')   #mechanism only needed for wiring cells not for simulating them.     

hubtuple=0
ihubtuple=0
if utils.COMM.rank==0:
    hubs=NetStructure(utils,utils.global_ecm,utils.global_icm,utils.visited,utils.celldict)
    print 'experimental rig'
    hubs.save_matrix()
    # A global analysis of hub nodes, using global complete adjacency matrices.
    #Inhibitory hub
    (outdegreei,indegreei)=hubs.hubs(utils.global_icm)    
    #Excitatory hub
    ihubtuple=(outdegreei,indegreei)
    (outdegree,indegree)=hubs.hubs(utils.global_ecm)    
    hubtuple=(outdegree,indegree)
# This is a global gid, but I would need to bcast it to every host, 
# and then check each host to see if that GID actually exists there.
hubtuple = MPI.COMM.bcast(hubtuple, root=0)
ihubtuple = MPI.COMM.bcast(ihubtuple, root=0)

print hubtuple, ihubtuple, ' broadcast hub values from rank0 bcast to every rank'
#check if these hubs are in the GID dictionary.

amplitude=0.57
delay=20
duration=600
if hubtuple in utils.celldict.keys():
    hubs.insert_cclamp(hubtuple[0],hubtuple[1],amplitude,delay,duration)

if ihubtuple in utils.celldict.keys():
    hubs.insert_cclamp(ihubtuple[0],ihubtuple[1],amplitude,delay,duration)

hubs=NetStructure(utils,utils.ecm,utils.icm,utils.visited,utils.celldict)
(outdegreei,indegreei)=hubs.hubs(utils.global_icm)    
(outdegree,indegree)=hubs.hubs(utils.global_ecm)    
# A local analysis of hub nodes, using local incomplete adjacency matrices.
amplitude=0.27 #pA or nA?
delay=15# was 1020.0 ms, as this was long enough to notice unusual rebound spiking
duration=500.0 #was 750 ms, however this was much too long.
hubs.insert_cclamp(outdegreei,indegreei,amplitude,delay,duration)
amplitude=0.27 #pA or nA?
delay=200# was 1020.0 ms, as this was long enough to notice unusual rebound spiking
duration=1000.0 #was 750 ms, however this was much too long.
hubs.insert_cclamp(hubs.outdegree,hubs.indegree,amplitude,delay,duration)
if utils.COMM.rank!=0:
    vec = utils.record_values()
print 'setup recording'
tstop = 3570
utils.COMM.barrier()
utils.prun(tstop)
if utils.COMM.rank==0:
    vec={}
utils.global_vec = utils.COMM.gather(vec,root=0) # Results in a list of dictionaries on rank 0 called utils.global_vec
# Convert the list of dictionaries into one big dictionary called global_vec (type conversion).
if utils.COMM.rank==0:
    utils.global_vec = {key : value for dic in utils.global_vec for key,value in dic.iteritems()  } 
    import matplotlib 
    import matplotlib.pyplot as plt
    matplotlib.use('Agg') 
    fig = plt.figure()
    fig.clf()
    plt.hold(True) #seems to be unecessary function call.
    #TODO outsource management of membrane traces to neo/elephant.
    #TODO use allreduce to reduce python dictionary to rank0
    
    import json
    jtl=[]

    #plt.hold(False) #seems to be unecessary function call.
    for gid,v in utils.global_vec['v'].iteritems():
        jtl.append((utils.global_vec['t'].to_python(),v.to_python()))
        plt.plot(utils.global_vec['t'].to_python(),v.to_python())
    json.dump(jtl,f)
    f=open('membrane_traces','w')
    fig.savefig('membrane_traces_from_all_ranks'+str(utils.COMM.rank)+'.png')    
    plt.hold(False) #seems to be unecessary function call.
    plt.xlabel('time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('traces')
    plt.grid(True)
    
utils.cell_info_gather()
utils.spike_gather()
if utils.COMM.rank==0:    
    def collate_spikes():
        #tvec and gidvec are Local variable copies of utils instance variables.
        tvec=[]
        gidvec=[]
        assert type(tvec)!=type(utils.h)
        assert type(gidvec)!=type(utils.h)
        for i,j in utils.global_spike:# Unpack list of tuples
            tvec.extend(i)
            gidvec.extend(j)
        return tvec, gidvec     
    tvec,gidvec=collate_spikes()
    utils.dumpjson_spike(tvec,gidvec)
    # open URL in running web browser
    #http_server.load_url('web/index.html')

def plot_raster(tvec,gidvec):
    pallete=[[0.42,0.67,0.84],[0.50,0.80,1.00],[0.90,0.32,0.00],[0.34,0.67,0.67],[0.42,0.82,0.83],[0.90,0.59,0.00], 
                [0.33,0.67,0.47],[0.42,0.83,0.59],[0.90,0.76,0.00],[1.00,0.85,0.00],[0.71,0.82,0.41],[0.57,0.67,0.33]]

    fig = plt.figure()
    fig.clf()
    color=[1.00,0.38,0.60] # Choose differe Colors for each cell population
    plt.title("Raster Plot")
    plt.hold(True)
    plt.plot(tvec,gidvec,'.',c=color, markeredgecolor = 'none')
    plt.savefig('raster'+str(utils.COMM.rank)+'.png')
    
if utils.COMM.rank==0:
    plot_raster(tvec,gidvec)
    # Compute the multivariate SPIKE distance
    list_spike_trains = [ i for i in xrange(0,int(np.max(gidvec)+1))] #define a list of lists.
    for i,j in enumerate(gidvec):
        list_spike_trains[int(j)]=int(tvec[int(i)])
    ti = 0
    tf = np.max(np.array(list_spike_trains))
    list_spike_trains=np.array(list_spike_trains).astype(int)
    from isi_distance import Kdistance
    K = Kdistance()
    t, Sb = K.multivariate_spike_distance(list_spike_trains, ti, tf, 2000)
    fig = plt.figure()
    fig.clf()    
    plt.title('Inter Spike Intervals')
    plt.plot(t,Sb,'k')
    plt.xlabel("Time (ms)")
    sfin = 'kreuz_multivariate'+str(utils.COMM.rank)+'.png'    
    plt.savefig(sfin)
    import json
    d =[]
    d.append(t.tolist())
    d.append(Sb.tolist())
    json.dump(d, open('web/js/spike_distance.json','w'))    
    print('Wrote node-link JSON data to web/js/network.json')
#Probably just get the spike distance.

#import http_server as hs
#hs.load_url('force.json')

