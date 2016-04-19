from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
import pdb
mca = MouseConnectivityApi()
print mca
# get metadata for all non-Cre experiments
#experiments = mca.experiment_source_search(injection_structures='root', transgenic_lines=0)
# download the projection density volume for one of the experiments
#pd = mca.download_projection_density('example.nrrd', experiments[0]['id'], resolution=25)

#enter selected projection numbers here, e.g. 167794131, 297231636, 287495026
id1 = 167794131
id2 = 297231636
id3 = 272736450
# import allensdk python api
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
mca = MouseConnectivityApi()
# get metadata for all experiments
experiments = mca.experiment_source_search(injection_structures=['VIS','PTLp','RSP'])
# find selected experiments and format filenames, **Use the %paste magic function if pasting into ipython
for i in range(len(experiments)):
    if (experiments[i]['id'] == id1):
        fn1 = str(experiments[i]['injection-structures'][0]['abbreviation']) + '_' + str(experiments[i]['id']) + '_' + str(experiments[i]['transgenic-line'])
        print fn1
    if (experiments[i]['id'] == id2):
        fn2 = str(experiments[i]['injection-structures'][0]['abbreviation']) + '_' + str(experiments[i]['id']) + '_' + str(experiments[i]['transgenic-line'])
        print fn2
    if (experiments[i]['id'] == id3):
        fn3 = str(experiments[i]['injection-structures'][0]['abbreviation']) + '_' + str(experiments[i]['id']) + '_' + str(experiments[i]['transgenic-line'])
        print fn3
# download selected experiment projection density files at 25 um resolution
mca.download_projection_density(fn1 + '.nrrd', id1, resolution=25)
mca.download_projection_density(fn2 + '.nrrd', id2, resolution=25)
mca.download_projection_density(fn3 + '.nrrd', id3, resolution=25)
import datetime, nrrd
import matplotlib.pyplot as plt

A1, metadata1 = nrrd.read(fn1 + '.nrrd')
A1.shape

plt.imshow(A1[300,:,:])
plt.savefig('mouse_connectivity1.png')
mxProj1 = amax(A1,0)  #second input is the dimension you want to project over
plt.imshow(mxProj1)
plt.savefig('mouse_connectivity2.png')



for exp in range(len(experiments)):
    
    pd = mca.download_projection_density('example.nrrd', experiments[exp]['id'], resolution=25)
    print(type(pd))
    
    
    pdb.set_trace()
    
# The manifest file is a simple JSON file that keeps track of all of
# the data that has already been downloaded onto the hard drives.
# If you supply a relative path, it is assumed to be relative to your
# current working directory.
mcc = MouseConnectivityCache(manifest_file='connectivity/manifest.json')

# open up a list of all of the experiments
all_experiments = mcc.get_experiments(dataframe=True)


from allensdk.api.queries.biophysical_perisomatic_api import \
    BiophysicalPerisomaticApi

from allensdk.api.queries.cell_types_api import CellTypesApi
import allensdk.core.swc as swc
import os
from allensdk.core.nwb_data_set import NwbDataSet
from IPython import __main__


bp = BiophysicalPerisomaticApi('http://api.brain-map.org')

ct = CellTypesApi()
cells = ct.list_cells()


#Below is one way, I can create URL, queries and get the result in a local python dictionary.
import allensdk 
import mpi4py as MPI
import numpy as np
import csv
import pickle
import sys
#import glob

def __main__():
    print 'main'
    


def query_all_neurons():
    '''
    Return API queries about biophysical perisomatic_neurons.
    Returns a list of well known file keys, but not yet properly formed URLs.
    '''
    instance=allensdk.api.api.Api('http://api.brain-map.org')
    #Request only files associated with a model.
    returned=allensdk.api.api.Api.retrieve_parsed_json_over_http(instance,'http://api.brain-map.org/api/v2/data/query.json?criteria=model::NeuronalModel')
    bphys = [n for n in returned['msg'] if 'Biophysical' in n['name'] ]
    sid=[ b['id'] for b in bphys ]
    listwk = [ bp.get_well_known_file_ids(n) for n in sid ]
    return listwk

def retrieve_files(filename,wkfc,instance,working_directory=None):    
    '''
    Arguments: the file name and the well known file code, an instance of the Allen Brain API object.
    Builds a well known file URL, and actually downloads and caches the files locally.    
    '''    
    wk=instance.construct_well_known_file_download_url(wkfc) 
    instance.retrieve_file_over_http(wk,filename)    
    return #files are cached on hard disk nothing to return
    

def get_all_files(instance,bphys):
    ''' 
    phys is a big list of dictionaries, that contains all of the different types of files you might be 
    interested in, the list of dictionaries needs to be traversed in order to download all of the files.
    Note that calling retrieve_files iteratively inside a list comprehension is sufficient to
    download all the requested files, and store them to the current working directory.
    '''
    [ retrieve_files(y,x,instance) for n in bphys for x,y in n['modfiles'].iteritems() ]
    [ retrieve_files(y,x,instance) for n in bphys for x,y in n['morphology'].iteritems() ]
    [ retrieve_files(y,x,instance) for n in bphys for x,y in n['fit'].iteritems() ]
    [ retrieve_files(y,x,instance) for n in bphys for x,y in n['stimulus'].iteritems() ]    
    for n in bphys:
        bp.create_manifest(str(n['fit'].values()[0]),
                           str(n['stimulus'].values()[0]),
                           str(n['morphology'].values()[0]),
                           [0,1,2,3,4,5])
        print mp.manifest
        manifest_path = os.path.join(os.getcwd(), str(n['morphology'].values()[0])+str(manifest.json))
        with open(manifest_path, 'wb') as f:
            f.write(json.dumps(mp.manifest, indent=2))


bphys = query_all_neurons()   
#Download everything, only do this if you have not downloaded everything already, as some files are large
#and time consuming.
get_all_files(instance,bphys)
instance=allensdk.api.api.Api('http://api.brain-map.org')
downloads = [ instance.construct_well_known_file_download_url(d) for d in bphys ]
working_directory='neuronal_model'


#Well Known File Format
#This retrieves a mod file. Only small files, can be obtained in an achievable time, 
#nwb files take much too long
#http://api.brain-map.org/api/v2/well_known_file_download/[WellKnownFile.id]
#Example http://api.brain-map.org/api/v2/well_known_file_download/3486
d='http://api.brain-map.org/api/v2/well_known_file_download/395337019'
instance.retrieve_file_over_http(d,'neuronal_model/'+str(files[0])) 


#bphys is a list containing a dictionary of dictionaries. The keys of the most nested dictionary are the suffixs of well known files. 
d=instance.construct_well_known_file_download_url(files[0]) 
cached_file_path = os.path.join(working_directory, d)
instance.retrieve_file_over_http(d,cached_file_path) 
for d in downloads:
    cached_file_path = os.path.join(working_directory, d)
    instance.retrieve_file_over_http(d,cached_file_path) 
#Above and below are the same. Above is online version, below is offline version that acts on local files.
#Use the other methods in the biophysical_perisomatic_api to one by one do the things needed to cache whole models.
def read_local_json():
    f1= open('neuron_models_from_query_builder.json')
    information = allensdk.api.api.load(f1)
    return information
information=read_local_json()
bp.cache_data(395310469, working_directory='neuronal_model')
for i in information['msg']:    
    bp.cache_data(i['id'], working_directory='neuronal_model')
#==============================================================================
#Excitatory and inhibitory cells are categorised according to the suffix's listed below:
#Pvalb:  Inhibitory aspiny cells (i.e. it clustered the fast-spiking Pvalb cells)
#Scnn1a: Layer 4 excitatory pyramidal neurons.
import glob
from allensdk.model.biophysical_perisomatic.utils import Utils
from allensdk.model.biophys_sim.config import Config
from utils import Utils
config = Config().load('config.json')
utils = Utils(config)

def read_local_swc():
    morphs=[]
    cells1=[]
    Utils.initialize_hoc()
    swclist=glob.glob('*.swc')
    for swcf in swclist:
        #morphology = swc.read_swc(swcf)
        morphology=Utils.generate_morphology(swcf)
        cell1=Utils.load_cell_parameters()       
        cells1.append(cell1)
        print type(cells1)
        print type(cell1)
        morphology.root
        morphs.append(morphology)
    return morphs,swclist,cells1

morphs,swclist,cells1=read_local_swc()        
cells1
def mkjson(): #Only ascii as in dictionary contents
    from allensdk.core.json_utilities import write

    #can be serialised into dictionary contents.
    for m in morphs:
        write(str(m)+'.json',m.root)
    return 0

