'''
@author: The parallel wiring related functions are written by Russell Jarvis rjjarvis@asu.edu


'''
import unittest
import networkx
from mpi4py import MPI
import numpy as np
        


#how to pass attributes from one object into this cls type object?

class NetStructure():
    def __init__(self,utils,global_ecm,global_icm,global_visited,celldict):
        #assert rank=0
        self.COMM = MPI.COMM_WORLD
        self.SIZE = self.COMM.Get_size()
        self.RANK = self.COMM.Get_rank()
        setattr(self,'global_ecm',global_ecm)        
        #self.global_ecm=global_ecm
        self.global_icm=global_icm
        self.global_visited=global_visited
        setattr(self,'outdegree',0)
        setattr(self,'indegree',0)
        #self.indegree=0
        #self.outdegree=0
        self.celldict=celldict
        self.old_sum=0    
        self.utils=utils
        self.h=self.utils.h



    def setup_iclamp_step(self, target_cell, amp, delay, dur):
        self.stim = self.h.IClamp(target_cell.soma[0](0.5))
        self.stim.amp = amp
        self.stim.delay = delay
        self.stim.dur = dur
        
        

    def net_stat(self):
        '''
        This method is called on every rank with graphs that contain only partial connectivity information.
        If there are two equal structural out-degree hubs this method only finds the first one.
        '''
        excin=networkx.in_degree_centrality(networkx.DiGraph(self.global_ecm))
        excout=networkx.in_degree_centrality(networkx.DiGraph(self.global_ecm))
        return (excin,excout)

    def hubs(self, global_numpy_matrix):
        '''
        This method is called only on rank 0 with a complete list of global identifiers.
        If there are two equal structural out-degree hubs this method only finds the first one.
        '''
        colsums=np.array([np.sum(i) for i in np.column_stack(global_numpy_matrix)])
        rowsums=np.array([np.sum(i) for i in np.row_stack(global_numpy_matrix)])        
        outdegree=np.where(colsums == np.max(colsums))[0][0]
        indegree=np.where(rowsums == np.max(rowsums))[0][0]
        return (outdegree, indegree)

    def insert_cclamp(self,outdegree,indegree,amplitude,delay,duration):
        if outdegree in self.celldict.keys():
            self.setup_iclamp_step(self.celldict[int(outdegree)], amplitude,delay,duration)#0.27, 1020.0, 750.0) 
        if indegree in self.celldict.keys():
            self.setup_iclamp_step(self.celldict[int(indegree)], amplitude,delay,duration)#0.27, 1020.0, 750.0)  
        
    def save_matrix(self):    
        #save and plot matrix.
        SIZE=self.SIZE
        RANK=self.RANK
        #TODO replace with plotly.
        import matplotlib 
        matplotlib.use('Agg') 
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm 
        import pickle
        assert self.COMM.rank==0
        with open('excitatory_matrix.p', 'wb') as handle:
            pickle.dump(self.global_ecm, handle)
        with open('inhibitory_matrix.p', 'wb') as handle:
            pickle.dump(self.global_icm, handle)
        print 'connection matrices saved'
        fig = plt.figure()
        fig.clf()

        import numpy as np
        assert RANK==0
        assert np.sum(self.global_ecm)!=0
        #assert np.sum(self.global_icm)!=0

        fig = plt.figure()
        fig.clf()
        im = plt.imshow(self.global_visited, interpolation='nearest')    
        plt.autoscale(True)
        plt.colorbar(im)
        plt.xlabel('columns = targets')
        plt.ylabel('rows = sources')
        plt.title('visited Adjacency Matrix')
        plt.grid(True)
        sfin = 'visited_Adjacency_Matrix.png'
        fig.savefig(sfin)


        fig = plt.figure()
        fig.clf()
        im = plt.imshow(self.global_ecm, interpolation='nearest')    
        plt.autoscale(True)
        plt.colorbar(im)
        plt.xlabel('columns = targets')
        plt.ylabel('rows = sources')
        plt.title('Ecitatory Adjacency Matrix')
        plt.grid(True)
    
        sfin = 'Excitatory_Adjacency_Matrix.png'
        fig.savefig(sfin)

        fig = plt.figure()
        fig.clf()
    
        im = plt.imshow(self.global_icm, interpolation='nearest')
    
        plt.autoscale(True)
        plt.colorbar(im)
        plt.xlabel('columns = targets')
        plt.ylabel('rows = sources')
        plt.title('Inhibitory Adjacency Matrix')
        plt.grid(True)

        sfin = 'Inhibitory_Adjacency_Matrix.png'
        fig.savefig(sfin)


    def record_values(self):
        vec = { "v": [],
                "t": self.h.Vector() }
    
        for i, cell in enumerate(self.cells):
            vec["v"].append(self.h.Vector())
            vec["v"][i].record(cell.soma[0](0.5)._ref_v)
        vec["t"].record(self.h._ref_t)
    
        return vec



    '''

    def wirecells_test(self):
        """This function constitutes the outermost loop of the parallel wiring algor
        The function returns two adjacency matrices. One matrix whose elements are excitatory connections and another matrix of inhibitory connections"""
        from segment_distance import dist3D_segment_to_segment
        NCELL=self.NCELL
        SIZE=self.SIZE
        COMM = self.COMM
        RANK=self.RANK
        h=self.h    
        pc=h.ParallelContext()
        secnames = ''
        cellind =0 
        polarity = 0
        h('objref coords')
        h('coords = new Vector(3)')
        h('objref pc')
        h('pc = new ParallelContext()')        
        coordict=None
        coordictlist=None
        #Iterate over all CPU ranks, iterate through all GIDs (global 
        #identifiers, stored in the python dictionary).
        for s in xrange(0, SIZE):
            celliter= iter( (i, j) for i,j in self.celldict.iteritems() )  
            for (i,j) in celliter:  
                cell1=pc.gid2cell(i)
                coordictlist=self.nestedpre_test(i)
            data = COMM.bcast(coordictlist, root=s)  # ie root = rank
            if len(data) != 0:
                self.nestedpost_test(data)
        print('sums of connectivity\n')
        
        return (self.nclist, self.ecm, self.icm)
    def pre_cell(self,j):
        """
        
        This function searches for pre synaptic sources in the parallel wiring algorim
        This algorithm includes the soma as proxy for other regions which can exocytosis
        TODO: this algorithm is not currently limited to leaf nodes (such as dendrite sp
        """
        from neuron import h
        pc=h.ParallelContext()
        h=self.h   
        coordictlist=[]
        coordict={}
        pc=h.pc
        h('objref coords') 
        h('coords = new Vector(3)')
        if j in self.celldict.keys():
            seglist= iter( (seg, sec, self.celldict[j]) for sec in self.celldict[j].spk_trig_ls for seg in sec )              
            for (seg,sec, cellc) in seglist:
                    get_cox = str('coords.x[0]=x_xtra('
                                  + str(seg.x) + ')')
                    h(get_cox)                   
                    get_coy = str('coords.x[1]=y_xtra('
                                  + str(seg.x) + ')')
                    h(get_coy)
                    get_coz = str('coords.x[2]=z_xtra('
                                  + str(seg.x) + ')')
                    h(get_coz)
                    coordict['hostfrom']=pc.id()
                    coordict['coords'] = np.array(h.coords.to_python(),
                                              dtype=np.float64)
                    coordict['gid']= int(j)
                    coordict['seg']= seg.x                    
                    secnames = sec.name()                         
                    coordict['secnames'] = str(secnames)
                    coordictlist.append(coordict)               
        return coordictlist
       
    def post_cell(self,data):
        """
        This is the inner most loop of the parallel wiring algorithm.
        13For ever GID
        For every coordinate thats received from a broadcast.
        for i,t in self.celldict.iteritems():
        For ever GID thats on this host (in the dictionary)
        if i in self.celldict.keys():
        for k,i,t in iterdata :
        if the gids are not the same.
        Rule out self synapsing neurons (autopses), with the condition
        pre GID != post GID
        If the putative post synaptic gid exists on this CPU, the referen
        tree.
        My wiring algorithm uses HOC variables that interpolate the middl
        some C libraries to achieve collision detection for synapse alloc
        from neuromac.segment_distance import dist3D_segment_to_segment
        from segment_distance import dist3D_segment_to_segment
        and I am considering using them here also
        """
        from segment_distance import dist3D_segment_to_segment
        NCELL=self.NCELL
        SIZE=self.SIZE
        COMM = self.COMM
        RANK=self.RANK
        from neuron import h
        pc=h.ParallelContext()
        h=self.h    
        secnames = ''
        cellind =0 
        polarity = 0
        h('objref coords')
        h('coords = new Vector(3)')
        h('objref pc')
        h('pc = new ParallelContext()')        
        h('objref coords2') 
        h('coords2 = new Vector(3)')
        iterdata=iter( (i,t) for i,t in self.celldict.iteritems() if i in self.celldict.keys() )
                 
        for i,t in iterdata :                          
            # call a tigrimite function here.
            #
            #
            
            iterseg=iter( (seg,sec) for sec in t.spk_rx_ls for seg in sec)                    
            for seg,sec in iterseg:
                #print seg.x, sec.name(), k['secnames']
                h('objref cell1')
                h('cell1=pc.gid2cell('+str(i)+')')
                secnames = sec.name()
                cellind = int(secnames[secnames.find('Cell[') + 5:secnames.find('].')])  # This is the index of the post synaptic cell.
                h('insert xtra')
                #h('for cell1.all{for(x,0){ insert xtra}}')    
                h(str('coords2.x[2]=') + str('z_xtra(')
                  + str(seg.x) + ')')
                h(str('coords2.x[1]=') + str('y_xtra(')
                  + str(seg.x) + ')')
                h(str('coords2.x[0]=') + str('x_xtra(')
                + str(seg.x) + ')')
    
                h('coordsx=0.0')
                h.coordsx = k['coords'][0]
                h('coordsy=0.0')
                h.coordsy = k['coords'][1]
                h('coordsz=0.0')
                h.coordsz = k['coords'][2]
    #h('coordsx') and coordsx are not tautolous they are
    #One is a variable in the HOC space, the other is in
    #coordsx from the Python space has been broadcast ov      
                coordsx = float(k['coords'][0])
                coordsy = float(k['coords'][1])
                coordsz = float(k['coords'][2])
    
    #Find the euclidian distance between putative presynaptic segments, 
    #and putative post synaptic segments.    
    #If the euclidian distance is below an allowable threshold in micro 
    #meters, continue on with code responsible for assigning a 
    #synapse, and a netcon. Neurons parallel context class can handle the actual message passing associated with sending and receiving action potentials on different hosts.                               
    
    
                r = 0.
                import math
                r=math.sqrt((h.coords2.x[0] - coordsx)**2+(h.coords2.x[1] - coordsy)**2+(h.coords2.x[2] - coordsz)**2)
                gidn=k['gid']    
                r = float(r)
                if r < 10:  
                    
                    print r,# 'this is not hopefuly wiring everything to everything'
                    polarity = 0        
                    polarity=int(h.Cell[int(cellind)].polarity)
                    print seg.x, k['seg'], k['secnames'], sec.name(), RANK, k['hostfrom'], k['gid'], int(h.Cell[int(cellind)].gid1)
                    
                    print polarity, 'polarity'
                    h('objref syn_')        
                    if int(polarity) == int(0):
                        post_syn = secnames + ' ' + 'syn_ = new GABAa(' + str(seg.x) + ')'
                        self.icm[i][gidn] = self.icm[i][gidn] + 1
                        print i,gidn
                        assert np.sum(self.icm)!=0
  
                    else:
                        #TODO modify mod file for exp2syn such that cid exists.
                        #Also because exp2syn is an inbuilt mechanism need to refactor explicitly such that custom file 
                        #myexp2syn is used instead.
                        #post_syn = secnames + ' ' + 'syn_ = new exp2syn(' + str(seg.x) + ')'
                        #post_syn='syn_ = self.h.Exp2Syn('+str(seg.x)+',sec='+secnames+')'
                        
                        #syn_.e = connection["erev"]
                        

                        post_syn = secnames + ' ' + 'syn_ = new AMPA(' + str(seg.x) + ')'
                        
                        self.ecm[i][gidn] = self.ecm[i][gidn] + 1
                        print i,gidn
                        assert np.sum(self.ecm)!=0
    
                    h(post_syn)
                    print post_syn
                    h('print syn_')
                    syn_=h.syn_
                    h.syn_.cid=i
                    h.Cell[cellind].ampalist.append(h.syn_)
                    h.Cell[cellind].div.append(k['gid'])
                    h.Cell[cellind].gvpre.append(k['gid'])
                    nc=pc.gid_connect(k['gid'],syn_)                                        
                    nc.delay=1+r/0.4
                    nc.weight[0]=r/0.4    
                    self.nclist.append(nc)
                    
                    #source_section = source_cell.soma[0]
                    
                    #Below syntax wont work on a parallel architecture.
                    #nc = self.h.NetCon(source_section(0.5)._ref_v, syn, sec=source_section)
                    #nc.weight[0] = connection["weight"]
                    nc.threshold = -20
                    nc.delay = 2.0

                    #logger.debug('This is a critical message.',nc, self.ecm, self.icm)
                    #logger.debug('This is a low-level debug message.',nc, self.ecm, self.icm)

            #h('uninsert xtra')                          
        #return self.nclist, self.ecm, self.icm
    
    def te_between(self):
        """This function constitutes the outermost loop of the parallel wiring algor
        The function returns two adjacency matrices. One matrix whose elements are excitatory connections and another matrix of inhibitory connections"""
        from segment_distance import dist3D_segment_to_segment
        NCELL=self.NCELL
        SIZE=self.SIZE
        COMM = self.COMM
        RANK=self.RANK
        h=self.h    
        pc=h.ParallelContext()
        secnames = ''
        cellind =0 
        polarity = 0
        h('objref coords')
        h('coords = new Vector(3)')
        h('objref pc')
        h('pc = new ParallelContext()')        
        coordict=None
        coordictlist=None
        #Iterate over all CPU ranks, iterate through all GIDs (global 
        #identifiers, stored in the python dictionary).
        for s in xrange(0, SIZE):
            celliter= iter( (i, j) for i,j in self.celldict.iteritems() )  
            for (i,j) in celliter:  
                cell1=pc.gid2cell(i)
                coordictlist=self.precell(i)
            data = COMM.bcast(coordictlist, root=s)  # ie root = rank
            if len(data) != 0:
                self.postcell(data)
        print('sums of connectivity\n')
        
        return (self.nclist, self.ecm, self.icm)

    '''    

    #def testrankzero(self):
    #   assert rank=0


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
