'''
Author information. This is an extension to the Utils class from Allen Brain API. 
The parallel wiring related functions are written by Russell Jarvis rjjarvis@asu.edu
'''


import logging
import glob
from mpi4py import MPI
import numpy as np
import logging
from copy import deepcopy
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(funcName)s - %(lineno)d')
fh = logging.FileHandler('wiring.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)
import pdb as pdb
import pickle
import json
import os

#Numba, something I should include in the optimizer package.
#from numba import jit
#from numpy import arange
#@jit


from neuron import h
class Utils():
    def __init__(self,NCELL=40,readin=0):
        #super(Utils, self).__init__(description)
        # To Do 
        # reduce the number of side effects associated with functions in this code
        # by replacing a lot of instance attributes with local variables in methods.
        # This will reduce the size of the interface, and increase modularity/maintainability.
        # The hocobject and its name space is designed in such a way that that modifying its state one function
        # will always result in global side effects.
        #setattr(self,'h',h)
        #h=h
        #h=h
        ##from neuron import h
        self.h=h
        h('objref pc, py')
        h('pc = new ParallelContext()')
        h('py = new PythonObject()')
        setattr(self, 'readin',readin)
        setattr(self, 'synapse_list',[])
        setattr(self, 'namedict',{})
        setattr(self, 'global_namedict',{})
        setattr(self,'global_spike',tuple)
        self.tvec=h.Vector()
        self.gidvec=h.Vector()
        self.has_cells=0
        #self.readin=readin
        #self.synapse_list=[]
        self.stim = None
        self.stim_curr = None
        self.sampling_rate = None
        self.cells = []
        self.gidlist=[]
        #TODO update initial attributes using the more pythonic setattr
        self.global_vec=[]
        #self.NCELL=NCELL
        setattr(self,'NCELL',NCELL)
        setattr(self,'celldict',{})
        setattr(self,'name_list',[])
        #self.celldict={}
        self.COMM = MPI.COMM_WORLD
        self.SIZE = self.COMM.Get_size()
        self.RANK = self.COMM.Get_rank()
        self.allsecs=None #global list containing all NEURON sections, initialized via mkallsecs
        self.coordict=None
        self.celldict={}
        self.cellmorphdict={}
        self.nclist = []
        self.seclists=[]
        self.icm = np.zeros((self.NCELL, self.NCELL))
        self.ecm = np.zeros((self.NCELL, self.NCELL))
        self.visited = np.zeros((self.NCELL, self.NCELL))
        self.global_visited = np.zeros_like(self.icm)
        self.global_icm = np.zeros_like(self.icm)
        self.global_ecm = np.zeros_like(self.ecm)
        self.debugdata=[]
        self.names_list=np.zeros((self.NCELL, self.NCELL))
        self.global_names_list=np.zeros((self.NCELL, self.NCELL))
      
        h('load_file("nrngui.hoc")')
        h('load_file("import3d.hoc")')
        h('load_file("morph4.hoc")')
    
    def prep_list(self):                    
        '''
        Construct a new list with 1 in 3 inhibitory neurons and 2 out of 3 excitatory neurons. 
        It would be preferable to make an exhaustive list of all neurons
        however this is not practical for debugging small models, composed
        of a balance between excitation and inhibition.
        '''
        cil = pickle.load(open('cellinfolist.p', 'rb'))
        cil.remove(cil[0])#The first list element is the column titles.
        cil = [i for i in cil if int(len(i))>9 ]
        assert len(cil)!=0
        markram = [i for i in cil if "Markram" in i]
        aspiny=[i for i in cil if not "interneuron" in i if not "pyramid" in i]
        return markram
    
    #ef _move_cells(self):
        '''
        Not something that would typically be executed.
        '''
        #for m in markram:
            #execute_string='mv main/'+m[len(m)-2]+' swcfolder'
            #execute_string='git add swcfolder
            #print execute_string
        #    os.system(execute_string)        
        
    	return markram
    
    def both_trans(self,markram):
        '''
        A private method. Prepend _ to all private methods in this class, to be conventional.
        Make sure that the network is composed of 2/3 excitatory neurons 1/3 inhibitory neurons.
        '''
        assert len(markram)!=0
        bothtrans=[]                                                                      
        bothtrans=[i for j,i in enumerate(markram) if "pyramid" in i if (j<=2*(self.NCELL/3.0))]
        interneuron=[i for j,i in enumerate(markram) if "interneuron" in i ]   
        bothtrans.extend(interneuron[0:int(2*(self.NCELL/3.0))])
        import platform
        plat=platform.platform(aliased=0, terse=0)
        print 'platform of NSG', plat 
        if 'Darwin' in plat:
            import btmorph
            import os
            cwd=os.getcwd()
            btmorph.population_density_projection(destination=str(cwd), \
            filter='*.swc', outN=str(cwd)+"/density.png", precision=[10, 10, 10],depth='Y')
            #btmorph.plot_3D_SWC(filtered_fn)

        return bothtrans
            
    def make_cells(self,cell_list):
        '''
        Distribute cells across the hosts in a
        Round robin distribution (circular dealing of cells)
        https://en.wikipedia.org/wiki/Round-robin
        '''
        #import neuron
        #from neuron import h    
        coords = [0 for i in xrange(0,3)]#define list as a local variable.
        h('objref py')
        h('py = new PythonObject()')
        NCELL=self.NCELL
        SIZE=self.SIZE
        RANK=self.RANK
        pc=h.ParallelContext()
        h('objref tvec, gidvec')
        h('gidvec = new Vector()')
        h('tvec = new Vector()')
        assert len(cell_list)!=0
        d = { x: y for x,y in enumerate(cell_list)} 
        itergids = iter( (d[i][3],i) for i in range(RANK, NCELL, SIZE) )       
        #Create a dictionary, where keys are soma centre coordinates to check for two cells occupying the same position.
        #since dictionary keys have to be unique should throw error once two cell somas occupy exactly the same coordinates.         
        #TODO keep rank0 free of cells, such that all the memory associated with that CPU is free for graph theory related objects.
        #This would require an iterator such as the following.
        for (j,i) in itergids:
            has_cells=1#RANK specific attribute simplifies later code.
            cell = h.mkcell(j)
            self.names_list[i]=j
            print cell, j,i 
            cell.geom_nseg()
            cell.gid1=i 
            cell.name=j
            # Populate the dictionary with appropriate keys for the sanity check.           
            #excitatory neuron.
            #self.test_cell(d[i])
            if 'pyramid' in d[i]:                
                cell.pyr()
                cell.polarity=1                        
            #inhibitory neuron.
            else:                              
                cell.basket()
                cell.polarity=0           
            #8 lines of trailing code is drawn from.
            #http://neuron.yale.edu/neuron/static/docs/neuronpython/ballandstick5.html        
            pc.set_gid2node(i,RANK)
            nc = cell.connect2target(None)
            pc.cell(i, nc) # Associate the cell with this host and gid
            #### Record spikes of this cell
            pc.spike_record(i, self.tvec, self.gidvec)        
            assert None!=pc.gid2cell(i)
            self.celldict[i]=cell
            self.cells.append(cell)
    '''
    def plot_cell_centre(self,cell_list):

        Method string:
        plot all of these cell centres in java script, and or plotly
        Either run make cells or this method but not both.
                
        import neuron
        from neuron import h    
        coords = [0 for i in xrange(0,3)]#define list as a local variable.
        h('objref py')
        h('py = new PythonObject()')
        NCELL=self.NCELL
        SIZE=self.SIZE
        RANK=self.RANK
        pc=h.ParallelContext()
        h('objref tvec, gidvec')
        h('gidvec = new Vector()')
        h('tvec = new Vector()')
        print len(cell_list)
        d = { x: y for x,y in enumerate(cell_list)} 
        itergids = iter( (d[i][3],i) for i in range(RANK, NCELL, SIZE) )       
        #Create a dictionary, where keys are soma centre coordinates to check for two cells occupying the same position.
        #since dictionary keys have to be unique should throw error once two cell somas occupy exactly the same coordinates.         
        checkd={} 
        plotx=[]
        ploty=[]
        plotz=[]
        #TODO keep rank0 free of cells, such that all the memory associated with that CPU is free for graph theory related objects.
        #This would require an iterator such as the following.
        for (j,i) in itergids:
            has_cells=1#RANK specific attribute simplifies later code.
            cell = h.mkcell(j)
            self.names_list[i]=j
            print cell, j,i 
            cell.geom_nseg()
            cell.gid1=i 
            cell.name=j
            # Populate the dictionary with appropriate keys for the sanity check.
           
            sec=cell.soma[0]        
            sec.push()
            plotx.append(neuron.h.x3d(0))
            ploty.append(neuron.h.y3d(0))
            plotz.append(neuron.h.z3d(0))

            key_checkd=str(neuron.h.x3d(0))+str(neuron.h.y3d(0))+str(neuron.h.z3d(0))
            #Dictionary values may as well be the morphology SWC name, in case of a file that commits offensive duplicating of position.
            #Additionally I may as well make a plot of soma positions.
            #assert !(key_checkd in checkd.keys()) #If the key is not in the dictionary, then add it and proceed with the business of cell instantiation.
            checkd[key_checkd] = (j, str(neuron.h.x3d(0)),str(neuron.h.y3d(0)),str(neuron.h.z3d(0)) ) 
            print key_checkd, checkd[key_checkd]
            h.pop_section()
            #excitatory neuron.
            #self.test_cell(d[i])
            if 'pyramid' in d[i]:                
                cell.pyr()
                cell.polarity=1                        
            #inhibitory neuron.
            else:                              
                cell.basket()
                cell.polarity=0           
            #8 lines of trailing code is drawn from.
            #http://neuron.yale.edu/neuron/static/docs/neuronpython/ballandstick5.html        
            pc.set_gid2node(i,RANK)
            nc = cell.connect2target(None)
            pc.cell(i, nc) # Associate the cell with this host and gid
            #### Record spikes of this cell
            pc.spike_record(i, self.tvec, self.gidvec)        
            assert None!=pc.gid2cell(i)
            self.celldict[i]=cell
            self.cells.append(cell)
        import platform
        plat=platform.platform(aliased=0, terse=0)
        print 'platform of NSG', plat 
        if 'Darwin' in plat:
            import plotly.plotly as py
            import plotly.graph_objs as go
            plotly.offline.plot({
            trace1 = go.Scatter3d(
                x=plotx,
                y=ploty,
                z=plotz,
                mode='markers',
                marker=dict(
                    color='rgb(127, 127, 127)',
                    size=12,
                    symbol='circle',
                    line=dict(
                        color='rgb(204, 204, 204)',
                        width=1
                    ),
                    opacity=0.9
                )
            )  
            data=[trace1] 
            layout = go.Layout(
                margin=dict(
                    l=0,
                    r=0,
                    b=0,
                    t=0
                )
            )
            })
            fig = go.Figure(data=data, layout=layout)
            plot_url = py.plot(fig, filename='soma centre positions')
            from neuronvisio.controls import Controls
            controls = Controls()
        checkd=None
        '''
    def gcs(self,NCELL):
        '''
        Instantiate NEURON cell Objects in the Python variable space such
        that all cells have unique identifiers.
        '''
        NCELL=self.NCELL
        SIZE=self.SIZE
        RANK=self.RANK
        #from neuron import h    
        pc=h.ParallelContext()     
        h('objref nc, cells')
        swcdict={}
        NFILE = 3175
        bothtrans=self.both_trans(self.prep_list())   
        self.names_list=[0 for x in xrange(0,len(bothtrans))]
        os.chdir(os.getcwd() + '/swclist') 
        self.make_cells(bothtrans)
        os.chdir(os.getcwd() + '/../')               
        h.define_shape()        
        h('forall{ for(x,0){ insert xtra }}')
        h('forall{ for(x,0){ insert extracellular}}')    
        h('xopen("interpxyz.hoc")')
        h('grindaway()')    
    
    def spike_gather(self):
        NCELL=self.NCELL
        SIZE=self.SIZE
        COMM = self.COMM
        RANK=self.RANK
        self.global_spike=COMM.gather([self.tvec.to_python(),self.gidvec.to_python()], root=0)

    def cell_info_gather(self):
        NCELL=self.NCELL
        SIZE=self.SIZE
        COMM = self.COMM
        RANK=self.RANK
        self.namedict= { key : (value.name, int(value.polarity)) for key,value in self.celldict.iteritems() }
        self.global_namedict=COMM.gather(self.namedict, root=0)        
        if RANK==0:
            self.global_namedict = {key : value for dic in self.global_namedict for key,value in dic.iteritems()  }
    
    def matrix_reduce(self, matrix=None):
        '''
        collapse many incomplete rank specific matrices into complete global matrices on rank0.
        This function has side effects (it mutates object arguments, although it currently has no arguments, this will become clearer after refacttoring).
        '''
        # TODO make this method argument based so it can handle arbitary input matrices not a few different particular
        # types
        # TODO apply function decorator.
        NCELL=self.NCELL
        SIZE=self.SIZE
        COMM = self.COMM
        RANK=self.RANK
        global_matrix = np.zeros_like(matrix)
        COMM.Reduce([matrix, MPI.DOUBLE], [global_matrix, MPI.DOUBLE], op=MPI.SUM,
                    root=0)
        #if RANK==0:
        #    assert np.sum(global_matrix)!=0
        # The icm might be zero for example.
        return global_matrix
        
    def prun(self,tstop):
        #from neuron import h    
        pc=h.ParallelContext()
        NCELL=self.NCELL
        SIZE=self.SIZE
        COMM = self.COMM
        RANK=self.RANK
        checkpoint_interval = 50000.

        #The following definition body is from the open source code at:
        #http://senselab.med.yale.edu/ModelDB/ShowModel.asp?model=151681
        #with some minor modifications
        cvode = h.CVode()
        cvode.cache_efficient(1)
        # pc.spike_compress(0,0,1)
    
        pc.setup_transfer()
        mindelay = pc.set_maxstep(10)
        if RANK == 0:
            print 'mindelay = %g' % mindelay
        runtime = h.startsw()
        exchtime = pc.wait_time()
    
        inittime = h.startsw()
        h.stdinit()
        inittime = h.startsw() - inittime
        if RANK == 0:
            print 'init time = %g' % inittime
    
        while h.t < tstop:
            told = h.t
            tnext = h.t + checkpoint_interval
            if tnext > tstop:
                tnext = tstop
            pc.psolve(tnext)
            if h.t == told:
                if RANK == 0:
                    print 'psolve did not advance time from t=%.20g to tnext=%.20g\n' \
                        % (h.t, tnext)
                break    
            print 'working', h.t
        runtime = h.startsw() - runtime
        comptime = pc.step_time()
        splittime = pc.vtransfer_time(1)
        gaptime = pc.vtransfer_time()
        exchtime = pc.wait_time() - exchtime
        if RANK == 0:
            print 'runtime = %g' % runtime
        print comptime, exchtime, splittime, gaptime


    def pre_synapse(self,j):
        '''
        search for viable synaptic vesicle sites.
        '''
        #from neuron import h   
        pc=h.ParallelContext()
        shiplist=[]
        h('objref coords') 
        h('coords = new Vector(3)')
        #self.celldict.items()[0]
        if j in self.celldict.keys():
            seglist= iter( (seg, sec, self.celldict[j]) for sec in self.celldict[j].spk_trig_ls for seg in sec )     
            for (seg,sec, cellc) in seglist:
                sec.push()
                get_cox = str('coords.x[0]=x_xtra('
                              + str(seg.x) + ')')
                h(get_cox)                   
                get_coy = str('coords.x[1]=y_xtra('
                              + str(seg.x) + ')')
                h(get_coy)
                get_coz = str('coords.x[2]=z_xtra('
                              + str(seg.x) + ')')
                h(get_coz)
                coordict={} 
                coordict['hostfrom'] = pc.id()
                coordict['coords'] = np.array(h.coords.to_python(),
                                          dtype=np.float64)
                coordict['gid']= int(j)
                coordict['seg']= seg.x                    
                secnames = h.cas().name()#sec.name()  
                coordict['secnames'] = str(secnames)
                shiplist.append(coordict)
                h.pop_section()

        '''                
        total_matrix=np.matrix(( 3,len(shiplist) ))
        total_list=[ (x['coords'][0],x['coords'][1],x['coords'][2]) for x in shiplist ]
        for i,j in enumerate(total_list):
            print type(j)
            #pdb.set_trace()
            total_matrix[i][0]=j[0]
            total_matrix[i][1]=j[1]
            total_matrix[i][2]=j[2]

        print total_array[:]
        '''
        
        return shiplist
        
    def alloc_synapse_ff(self,r,post_syn,cellind,k,gidn,i):

        NCELL=self.NCELL
        SIZE=self.SIZE
        COMM = self.COMM
        RANK=self.RANK
        pc=h.ParallelContext()
        polarity = 0        
        polarity=int(h.Cell[int(cellind)].polarity)
        if polarity==1:
            #TODO pickle load the graphs here instead of making them manually.
            self.ecm[i][gidn] = self.ecm[i][gidn] + 1
            #self.ecg.add_edge(i,gidn,weight=r/0.4)
            assert np.sum(self.ecm)!=0
        else:
            self.icm[i][gidn] = self.icm[i][gidn] + 1
            #self.icg.add_edge(i,gidn,weight=r/0.4)
            assert np.sum(self.icm)!=0                
            #TODO Add other edge attributes like secnames etc.
        print post_syn
        h('objref syn_')   
        h(post_syn)
        syn_=h.syn_
        h.syn_.cid=i
        h.Cell[cellind].ampalist.append(h.syn_)
        h.Cell[cellind].div.append(k['gid'])
        h.Cell[cellind].gvpre.append(k['gid'])
        nc=pc.gid_connect(k['gid'],syn_)                                        
        nc.threshold = -20
        nc.delay=1+r/0.4
        nc.weight[0]=r/0.4    
        self.nclist.append(nc)
        

    def alloc_synapse(self,r,h,sec,seg,cellind,secnames,k,i,gidn):
        '''
        Allocate a synaptic cleft from exhuastive collision detection.
        '''
        NCELL=self.NCELL
        SIZE=self.SIZE
        COMM = self.COMM
        RANK=self.RANK
        #from neuron import h
        pc=h.ParallelContext()
        #from neuron import h
        self.visited[i][gidn] = self.visited[i][gidn] + 1              
        if r < 2.5: #2.5 micro metres.
            polarity = 0        
            polarity=int(h.Cell[int(cellind)].polarity)
            h('objref syn_')        
            if int(polarity) == int(0):
                post_syn = secnames + ' ' + 'syn_ = new FastInhib(' + str(seg.x) + ')'
                #post_syn = secnames + ' ' + 'syn_ = new GABAa(' + str(seg.x) + ')'
                self.icm[i][gidn] = self.icm[i][gidn] + 1
                #self.icg.add_edge(i,gidn,weight=r/0.4)
                #self.icg[i][gidn]['post_loc']=secnames
                #self.icg[i][gidn]['pre_loc']=k['secnames']
                assert np.sum(self.icm)!=0
        
            else:
                if (k['gid']%2==0):
                    #TODO Find standard open source brain affiliated code for NMDA synapse
                    post_syn = secnames + ' ' + 'syn_ = new AmpaNmda(' + str(seg.x) + ')'                       
                    self.ecm[i][gidn] = self.ecm[i][gidn] + 1
                    #self.ecg.add_edge(i,gidn,weight=r/0.4)
                    #self.ecg[i][gidn]['post_loc']=secnames
                    #self.ecg[i][gidn]['pre_loc']=k['secnames']
                    self.seclists.append(secnames)
                    assert np.sum(self.ecm)!=0
                else:
                    #TODO Find standard open source brain affiliated code for NMDA synapse
                    post_syn = secnames + ' ' + 'syn_ = new ExpSid(' + str(seg.x) + ')'                       
                    self.ecm[i][gidn] = self.ecm[i][gidn] + 1
                    #self.ecg.add_edge(i,gidn,weight=r/0.4)
                    #self.ecg[i][gidn]['post_loc']=secnames
                    #self.ecg[i][gidn]['pre_loc']=k['secnames']
                    self.seclists.append(secnames)
                    assert np.sum(self.ecm)!=0
        
            h(post_syn)
            print post_syn
            self.synapse_list.append((r,post_syn,cellind,k,gidn,i))
            syn_=h.syn_
            h.syn_.cid=i
            h.Cell[cellind].ampalist.append(h.syn_)
            h.Cell[cellind].div.append(k['gid'])
            h.Cell[cellind].gvpre.append(k['gid'])
            nc=pc.gid_connect(k['gid'],syn_)                                        
            nc.threshold = -20
            nc.delay=1+r/0.4
            nc.weight[0]=r/0.4    
            self.nclist.append(nc)

        
       
    def post_synapse(self,data):
        """
        search viable post synaptic receptor sites.
        
        This is the inner most loop of the parallel wiring algorithm.
        For every GID
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
        This wiring algorithm uses HOC variables that interpolate the middl
        some C libraries to achieve collision detection for synapse alloc
        from neuromac.segment_distance import dist3D_segment_to_segment
        from segment_distance import dist3D_segment_to_segment
        and I am considering using them here also
        """
        #from segment_distance import dist3D_segment_to_segment
        NCELL=self.NCELL
        SIZE=self.SIZE
        COMM = self.COMM
        RANK=self.RANK
        #from neuron import h
        pc=h.ParallelContext()
        #from neuron import h    
        secnames = ''
        cellind =0 
        polarity = 0
        h('objref coords')
        h('coords = new Vector(3)')
        h('objref pc')
        h('pc = new ParallelContext()')        
        h('objref coords2') 
        h('coords2 = new Vector(3)')
        for q,s in enumerate(data):
            for t in s:
                k={} #The only point of this redundantvariable switching is to force the dictionary k to be redclared 
                k=t #such that it is not prevented from updating
                itercell= ( (i,t) for i,t in self.celldict.iteritems() if i in self.celldict.keys() if int(t.gid1) != int(k['gid']) )       
                for i,t in itercell :                          
                    # TODO save time by checking if the somas of the two cells are reasonably close before checking every sec,seg in every neuron.
                    #
                    # t.soma[0].
                    iterseg=iter( (seg,sec) for sec in t.spk_rx_ls for seg in sec)               
                    for (seg,sec) in iterseg:
                        segxold=seg.x
                        h('objref cell1')
                        h('cell1=pc.gid2cell('+str(i)+')')
                        secnames = sec.name()
                        cellind = int(secnames[secnames.find('Cell[') + 5:secnames.find('].')])  # This is the index of the post synaptic cell.
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
            #h('coordsx') and coordsx are not tautolous. 
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
                        self.alloc_synapse(r,h,sec,seg,cellind,secnames,k,i,gidn)

    '''
    def destroy_isolated_cells(self):        
        
        To be called locally on every rank
        This method is intended to do two things.
        First it finds and deletes isolated nodes from the 3 networkx objects with degree 0.
        
        Then it intends destroys the associated HOC cell objects with degree 0.
        If this does not prove fatal to the subsequent NEURON simulation. 
        
        
        import networkx as nx
        self.whole_net=nx.compose(self.ecg, self.icg)
        
        #self.whole_net.compose(self.ecg, self.icg)
        isolatedlist=nx.isolates(self.whole_net)
        self.whole_net.remove_nodes_from(nx.isolates(self.whole_net))
        self.icg.remove_nodes_from(nx.isolates(self.icg))
        self.ecg.remove_nodes_from(nx.isolates(self.ecg))
        
        for i in isolatedlist:
            print i, "isolated", celldict[i]
            celldict[i]=None #hopefully this will destroy the cell.
        pass
        #TODO hoc object level code that destroys the cell object.
        #    cell=pc.gid2cell(i)
             
        #    h('objref cell')
    '''    
    def wirecells(self):
        """This function constitutes the outermost loop of the parallel wiring algor
        The function returns two adjacency matrices. One matrix whose elements are excitatory connections and another matrix of inhibitory connections"""
        #from segment_distance import dist3D_segment_to_segment
        import pickle
        NCELL=self.NCELL
        SIZE=self.SIZE
        COMM = self.COMM
        RANK=self.RANK
        #from neuron import h    
        pc=h.ParallelContext()
        secnames = ''
        cellind =0 
        polarity = 0
        h('objref coords')
        h('coords = new Vector(3)')
        h('objref pc')
        h('pc = new ParallelContext()')        
        coordict=None
        coordictlist=[]

        #Iterate over all CPU ranks, iterate through all GIDs (global 
        #identifiers, stored in the python dictionary).
        if self.readin!=1:    
            for s in xrange(0, SIZE):
                
                #Synchronise processes here, all ranks must have finished receiving 
                #transmitted material before another transmission of the coordictlis begins, potentially
                #overwritting a coordictlist before it has been properly exhausted.
                COMM.barrier() #New line could save CPU but functional? Can be removed
                coordictlist=[]
                if COMM.rank==s:
                    print 'begin creating message for transmision on rank ', COMM.rank,' s ', s
                    celliter= iter(i for i in self.celldict.keys())  
                    for i in celliter:  
                        cell1=pc.gid2cell(i)
                        coordictlist.append(self.pre_synapse(i))
                    print 'end tx on rank ', COMM.rank

                data = COMM.bcast(coordictlist, root=s)  # ie root = rank
                print 'checking for rx on rank ', COMM.rank
                if len(data) != 0:
                    print 'receieved rx on rank ', COMM.rank
                    self.post_synapse(data)
                    print 'using received message on rank ', COMM.rank
                    print len(data)
            print('finished wiring of connectivity\n')
            fname='synapse_list'+str(RANK)+'.p'
            if COMM.rank!=0: assert len(self.synapse_list)!=0
            with open(fname, 'wb') as handle:
                pickle.dump(self.synapse_list, handle)
            fname='visited'+str(RANK)+'.p'
            with open(fname, 'wb') as handle:
                pickle.dump(self.visited,handle)
            #self.destroy_isolated_cells()
        else:
            #if COMM.rank!=0:               
            fname='synapse_list'+str(RANK)+'.p'
            with open(fname, 'rb') as handle:
                self.synapse_list=pickle.load(handle)
                #for s in self.synapse_list:
                for (r,post_syn,cellind,k,gidn,i) in self.synapse_list:
                    self.alloc_synapse_ff(r,post_syn,cellind,k,gidn,i)
            #destroy_isolated_cells()

  
    def tracenet(self):
        '''
        This method does two things.
        1 Send a matrix to rank0.
        2 Do a local hub node computation.
        Ideally there should be too many neurons to properly visualise them in a network graph.
        Future design decision. Keep rank0 free of cells, such that it has the RAM to store
        big matrices.
        # Then destroy after sending to rank 0.
        # Maybe stimulate one cell per CPU.
        #

        '''
        ncsize=len(h.NetCon)
        NCELL=self.NCELL
        SIZE=self.SIZE
        COMM = self.COMM
        RANK=self.RANK
        self.matrix_reduce()
        lsoftup=[]
        for i, j in enumerate(h.NetCon):
            if type(j)!=None:
                assert type(j)!=None
                srcind=int(j.srcgid())
                tgtind=int(j.postcell().gid1)
                print int(j.srcgid()),int(j.postcell().gid1),self.celldict[srcind],self.celldict[tgtind]
                lsoftup.append((int(self.h.NetCon[i].srcgid()),int(self.h.NetCon[i].postcell().gid1),self.celldict[srcind],self.celldict[tgtind]))
        return lsoftup
      
        
    def dumpjson_graph(self):
        assert self.COMM.rank==0        
        import json
        d =[]
        whole=self.global_ecm+self.global_icm
        d.append(whole.tolist()) 
        d.append(self.global_namedict)
        json.dump(d, open('web/js/global_whole_network.json','w'))
        d=json.load(open('web/js/global_whole_network.json','r'))
        #read the object just to prove that is readable.
        d=None #destroy the object.    
        print('Wrote JSON data to web/js/network.json')
    
        # open URL in running web browser
        #http_server.load_url('force/force.html')

    def dumpjson_spike(self,tvec,gidvec):
        assert self.COMM.rank==0        
        import json
        h=self.h    
        d =[]
        d.append(self.global_namedict)
        assert (type(tvec)!=type(self.h) and type(gidvec)!=type(self.h))
        d.append(tvec)
        d.append(gidvec)
        json.dump(d, open('web/js/spike.json','w'))
        d=json.load(open('web/js/spike.json','r'))
        #read the object just to prove that is readable.
        d=None #explicitly destroy the object, as garbage collection would do anyway.   
        print('Wrote JSON data to web/js/spike.json')
    
   

    def generate_morphology(self, cell, morph_filename):
        '''
        This code is from the Allen Brain API examples.
        This code is no longer executed.
        morph4.hoc is executed instead.
        '''
        h = h
        swc = h.Import3d_SWC_read()
        swc.input(morph_filename)
        imprt = h.Import3d_GUI(swc, 0)
        h('execute("forall delete_section()",cell)')
        imprt.instantiate(cell)
        
      
        
        for seg in cell.soma[0]:
            seg.area()
        for sec in cell.allsec():
            sec.nseg = 1 + 2 * int(sec.L / 40)
        h.define_shape()
        
        #cell.simplify_axon()
        #for sec in cell.axonal:
        #    sec.L = 30
        #    sec.diam = 1
        #    sec.nseg = 1 + 2 * int(sec.L / 40)
        #cell.axon[0].connect(cell.soma[0], 0.5, 0)
        #cell.axon[1].connect(cell.axon[0], 1, 0)
    
    def load_cell_parameters(self, cell, type_index):
        #This code is from the Allen Brain API examples.
        '''
        This code is from the Allen Brain API examples.
        This code is no longer executed.
        morph4.hoc is executed instead.
        It is just good py-hoc example code.
        '''

        #from neuron import h
        passive = self.description.data['fit'][type_index]['passive'][0]
        conditions = self.description.data['fit'][type_index]['conditions'][0]
        genome = self.description.data['fit'][type_index]['genome']

        # Set passive properties
        cm_dict = dict([(c['section'], c['cm']) for c in passive['cm']])
        for sec in cell.all:
            sec.Ra = passive['ra']
            sec.cm = cm_dict[sec.name().split(".")[1][:4]]
            sec.insert('pas')
            for seg in sec:
                seg.pas.e = passive["e_pas"]

        # Insert channels and set parameters
        for p in genome:
            sections = [s for s in cell.all if s.name().split(".")[1][:4] == p["section"]]
            for sec in sections:
                sec.push()
                if p["mechanism"] != "":
                    print p["mechanism"]
                    sec.insert(p["mechanism"])
                    h('print psection()')
                setattr(sec, p["name"], p["value"])
                h.pop_section()
        # Set reversal potentials
        for erev in conditions['erev']:
            sections = [s for s in cell.all if s.name().split(".")[1][:4] == erev["section"]]
            for sec in sections:
                sec.ena = erev["ena"]
                sec.ek = erev["ek"]


    def setup_iclamp_step(self, target_cell, amp, delay, dur):
        self.stim = h.IClamp(target_cell.soma[0](0.5))
        self.stim.amp = amp
        self.stim.delay = delay
        self.stim.dur = dur

    def record_values(self):
        vec = { "v": {}, #define a dictionary.
                "t": h.Vector() }
        for i, cell in enumerate(self.cells):
            vec["v"][int(cell.gid1)]=h.Vector()
            vec["v"][int(cell.gid1)].record(cell.soma[0](0.5)._ref_v)
        vec["t"].record(h._ref_t)
    
        return vec
  
    def spikerecord(self):   
        '''
        This method duplicates other code. I intend to keep it as a method, and delete the duplicate lines.
        '''
        h('objref tvec, gidvec')
        h('gidvec = new Vector()')
        h('tvec = new Vector()')
        for cell in self.cells:
            h.pc.spike_record(int(cell.gid1), h.tvec, h.idvec)


    #TODO use neuro electro to test cortical pyramidal cells, and baskett cells before including
    #them in the network.
    #Call a method test_cell inside the make_cells function.
    '''
    def test_cell(self,d):#celltype='hip_pyr'):
        from neuronunit.neuroelectro import NeuroElectroSummary
        from neuronunit import neuroelectro
        x = neuroelectro.NeuroElectroDataMap()
        if 'hippocampus' in d:
            summary = NeuroElectroSummary(neuron={'name':'Hippocampus CA1 Pyramidal Cell'},
                                        ephysprop={'name':'spike width'})
            observation = summary.get_observation(show=True)
            #from neuronunit.tests import SpikeWidthTest
            #ca1_pyramdical_spike_width_test=SPikeWidthTest(observation=observation)
            #Does not work due to problem with elephant.
            #Note elephant requires pre-release version of neo.
            pass
        if 'neocortex' in d:
  
            x.set_neuron(nlex_id='sao2128417084')
            #pass
            #x.set_neuron(nlex_id='nifext_152') # neurolex.org ID for 'Amygdala basolateral
                                           # nucleus pyramidal neuron'.
            x.set_ephysprop(id=23) # neuroelectro.org ID for 'Spike width'.
            #TODO find neurolex.org ID for Vm
 
            pass
            #x.get_values() # Gets values for spike width from this paper. 
            #pdb.set_trace() 
            #width = x.val # Spike width reported in that paper. 
        if 'basket' in d:
            x.set_neuron(nlex_id='nifext_56')
            pass
        if 'dg_basket' in d:
            x.set_neuron(nlex_id='nlx_cell_100201')
            pass
        
        
        

    #TODO use neuro electro to test cortical pyramidal cells, and baskett cells before including
    #them in the network.
    #Call a method test_cell inside the make_cells function.
    def test_cell(self,d):#celltype='hip_pyr'):
        from neuronunit.neuroelectro import NeuroElectroSummary
        from neuronunit import neuroelectro
        x = neuroelectro.NeuroElectroDataMap()
        if 'hippocampus' in d:
            summary = NeuroElectroSummary(neuron={'name':'Hippocampus CA1 Pyramidal Cell'},
                                        ephysprop={'name':'spike width'})
            observation = summary.get_observation(show=True)
            #from neuronunit.tests import SpikeWidthTest
            #ca1_pyramdical_spike_width_test=SPikeWidthTest(observation=observation)
            #Does not work due to problem with elephant.
            #Note elephant requires pre-release version of neo.
            pass
        if 'neocortex' in d:
  
            x.set_neuron(nlex_id='sao2128417084')
            #pass
            #x.set_neuron(nlex_id='nifext_152') # neurolex.org ID for 'Amygdala basolateral
                                           # nucleus pyramidal neuron'.
            x.set_ephysprop(id=23) # neuroelectro.org ID for 'Spike width'.
            #TODO find neurolex.org ID for Vm 
            pass
            #x.get_values() # Gets values for spike width from this paper. 
            #pdb.set_trace() 
            #width = x.val # Spike width reported in that paper. 
        if 'basket' in d:
            x.set_neuron(nlex_id='nifext_56')
            pass
        if 'dg_basket' in d:
            x.set_neuron(nlex_id='nlx_cell_100201')
            pass
    '''    
    def my_decorator(self,some_function):
        def wrapper(self):
            #from neuron import h    
            NCELL=self.NCELL
            SIZE=self.COMM.size
            RANK=self.COMM.rank
            pc=h.ParallelContext()            
            self.some_function()
        return wrapper
    
    #@my_decorator
    #makecells()#I want to pass the function makecells as a function to the decorator.
    #So, @my_decorator is just an easier way of saying just_some_function = my_decorator(just_some_function). 
    #It's how you apply a decorator to a function
       
            
