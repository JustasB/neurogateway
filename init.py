# -*- coding: utf-8 -*-
"""
None of this stuff would work on NSG.
Created on Tue Sep  8 17:01:22 2015
The parallel wiring related functions are written by Russell Jarvis rjjarvis@asu.edu
import allensdk
from allensdk.api.queries.biophysical_perisomatic_api import \
    BiophysicalPerisomaticApi
from allensdk.api.queries.cell_types_api import CellTypesApi
import allensdk.core.swc as swc
from allensdk.core.nwb_data_set import NwbDataSet
from allensdk.model.biophysical_perisomatic.utils import Utils
from allensdk.model.biophys_sim.config import Config
bp = BiophysicalPerisomaticApi('http://api.brain-map.org')

    """

import os
import glob
import pickle
from utils import Utils
import numpy as np
import pdb 
config = Config().load('config.json')
# The readin flag when set enables the wiring to be read in from pre-existing 
# pickled files with rank specific file names.
utils = Utils(config,NCELL=60,readin=0)
info_swc=utils.gcs(utils.NCELL)
utils.wirecells()#wire cells on different hosts.
utils.global_icm=utils.matrix_reduce(utils.icm)
utils.global_ecm=utils.matrix_reduce(utils.ecm)
utils.global_visited=utils.matrix_reduce(utils.visited)
if utils.COMM.rank==0:    
    utils.dumpjson_graph()
utils.h('forall{ for(x,0){ uninsert xtra}}')   #mechanism only needed for wiring cells not for simulating them. 
from rigp import NetStructure
if utils.COMM.rank==0:
    hubs=NetStructure(utils,utils.global_ecm,utils.global_icm,utils.visited,utils.celldict)
    print 'experimental rig'
    hubs.save_matrix()
    # A global analysis of hub nodes, using global complete adjacency matrices.
    #Inhibitory hub
    (outdegreei,indegreei)=hubs.hubs(utils.global_icm)    
    #Excitatory hub
    (outdegree,indegree)=hubs.hubs(utils.global_ecm)    
    amplitude=0.27 #pA or nA?
    delay=60 # was 1020.0 ms, as this was long enough to notice unusual rebound spiking
    duration=400.0 #was 750 ms, however this was much too long.
    #Inhibitory hub
    hubs.insert_cclamp(outdegreei,indegreei,amplitude,delay,duration)
    #Excitatory hub
    hubs.insert_cclamp(outdegree,indegree,amplitude,delay,duration)
    hubs.insert_cclamp(hubs.outdegree,hubs.indegree,amplitude,delay,duration)
hubs=NetStructure(utils,utils.ecm,utils.icm,utils.visited,utils.celldict)
(outdegreei,indegreei)=hubs.hubs(utils.global_icm)    
(outdegree,indegree)=hubs.hubs(utils.global_ecm)    
# A local analysis of hub nodes, using local incomplete adjacency matrices.
amplitude=0.27 #pA or nA?
delay=15# was 1020.0 ms, as this was long enough to notice unusual rebound spiking
duration=400.0 #was 750 ms, however this was much too long.
hubs.insert_cclamp(outdegreei,indegreei,amplitude,delay,duration)
amplitude=0.27 #pA or nA?
delay=200# was 1020.0 ms, as this was long enough to notice unusual rebound spiking
duration=400.0 #was 750 ms, however this was much too long.
hubs.insert_cclamp(hubs.outdegree,hubs.indegree,amplitude,delay,duration)
vec = utils.record_values()
print 'setup recording'
tstop = 1570
utils.COMM.barrier()
utils.prun(tstop)
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
    for gid,v in utils.global_vec['v'].iteritems():
        plt.plot(utils.global_vec['t'].to_python(),v.to_python())
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
    http_server.load_url('web/index.html')

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
    list_spike_trains = [ [] for i in xrange(0,int(np.max(gidvec)+1))] #define a list of lists.
    for i,j in enumerate(gidvec):
        list_spike_trains[int(j)].append(tvec[int(i)])
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

