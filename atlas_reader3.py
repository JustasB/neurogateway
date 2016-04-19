#Ignore this sed code echo "126,55,55,71INTER,Cameron" | sed 's/\([0-9].*,[0-9].*,[0-9].*\)/\1/' 
#\1 is the reference to the first stored pattern.
import csv
import os
import pickle
d={}#declare a dictionary
dl=[]#A list to store the dictionary in.


fn = open('neocortex_region.txt')
neocortex = [line.strip() for line in open('neocortex_region.txt', 'r')]

fh = open('hippocampus_region.txt')
hippocampus = [line.strip() for line in open('hippocampus_region.txt', 'r')]


fbg = open('basalganglia_region.txt')
basalganglia = [line.strip() for line in open('basalganglia_region.txt', 'r')]

fbf = open('basalforebrainregion.txt')
basalforebrain = [line.strip() for line in open('basalforebrainregion.txt', 'r')]

fi = open('list_interneurons.txt')
interneurons = [line.strip() for line in open('list_interneurons.txt', 'r')]

fa = open('list_aspiny_cell.txt')
aspiny = [line.strip() for line in open('list_aspiny_cell.txt', 'r')]

fp = open('pyramidalcell.txt')
pyramid = [line.strip() for line in open('pyramidalcell.txt', 'r')]

with open('cngrid.csv', 'rb') as csvfile:#First just read the values.
   allrows=list(csv.reader(open('cngrid2.csv')))
   
os.chdir('main')
os.system('ls *.swc > swc_names.txt')
os.system('sed -i "s/.CNG.swc/ /" swc_names.txt')
f = open('swc_names.txt')
names = [line.strip() for line in open('swc_names.txt', 'r')]
f.close()
for x in range(3,len(names)-1): names[x]=names[x+1]
#now that I have names. Find them in the dictionary.
os.chdir('../')    
import re # ie regexp
cnt=0
for i in range(0,len(allrows)):
   a=allrows[i]
   allrows2=[]
   for j in range(0,len(interneurons)-1):
      #if(s['name']==names[j]):
      if(interneurons[j]==a[3]): 
         allrows[i].append("interneuron") #type 3
         allrows[i].append(3)
         #print allrows[i]
   for k in range(0,len(aspiny)-1):
      #if(s['name']==names[j]):
      if(aspiny[k]==a[3]): 
         allrows[i].append("aspiny") #type 4
         allrows[i].append(4)
         #print allrows[i]
   for l in range(0,len(pyramid)-1):
      #if(s['name']==names[j]):
      if(pyramid[l]==a[3]): 
         allrows[i].append("pyramid") # type 5
         allrows[i].append(5)
         #print allrows[i]
   for m in range(0,len(neocortex)-1):
       #if(s['name']==names[j]):
       if(neocortex[m]==a[3]): 
          allrows[i].append("neocortex") # type 5
          #allrows[i].append(5)
          #print allrows[i]
   for n in range(0,len(hippocampus)-1):
       #if(s['name']==names[j]):
       if(hippocampus[n]==a[3]): 
          allrows[i].append("hippocampus") # type 5
          #allrows[i].append(5)
          #print allrows[i]
   for o in range(0,len(basalforebrain)-1):
       #if(s['name']==names[j]):
       if(basalforebrain[o]==a[3]): 
          allrows[i].append("basalforebrain") # type 5
          #allrows[i].append(5)
          #print allrows[i]
   for p in range(0,len(basalganglia)-1):
       #if(s['name']==names[j]):
       if(basalganglia[p]==a[3]): 
          allrows[i].append("basalganglia") # type 5
          #allrows[i].append(5)
          #print allrows[i]
    #basalforebrain      
          
          

for i in range(0,len(allrows)):
   a=allrows[i]
   allrows2=[]
   for j in range(0,len(names)-1):

      
      #if(s['name']==names[j]):
      if(names[j]==a[3]): #if the cell name is found. 
         #print j
         #print  a[3], names[j] , a[4]==names[j], i, j
         a[3]=names[j]+".CNG.swc"
         #print a[3] 
         allrows[i].append(a[3])
         allrows[i].append(1)
         #print allrows[i]
         #allrows2.append(allrows[i])
         cnt+=1  

while i in range(1,len(allrows)-1): 
    b=allrows[i]
     
    ##print allrows[i]
    
    if(len(b)==5):
       #print allrows[i]
       allrows.remove(allrows[i])

#print allrows[3759+1]

pickle.pickledump(allrows, open("allrows.p", "wb" ) )

"""
import pandas as pd
csvalues = pd.read_csv('cngrid2.csv')


#columns=dict([(x[0:],x[1:]) for x in zip(*allrows)])
#atlasreader = csv.reader(csvfile,delimiter=',')#, fieldname='x'
for row in csv.DictReader(csvfile,delimiter=','):

   for key in row:
        #print 'key=%s, value=%s' % (key, row[key])
#d = dict(filter(None, csv.reader('cngrid.csv')))
#d = {rows[0]:rows[1] for rows in atlasreader}
#d={rows[:4] for rows in atlasreader if rows}
#atlasreader = csv.DictReader(csvfile)#,delimiter=',')#, fieldname='x'
#for row in list(csv.reader(open("copy-john.csv", "rb")))[1:]:
cnt=0
# The problem is to do with getting rows instead of columns.
# I need to go back to my general task of getting the trivial networks to plot stuff.
#
#

for row in atlasreader:
   #(key,val)=xposition,x
   if(row==1):
     d['xp']=row[0]#x  
     d['yp']=row[1]#y 
     d['zp']=row[2]#z
     d['name']=row[3]#name
     d['repo']=row[4]#repo
     d['number']=cnt
     d['returned']=0#assign zero. Reassign 1 if NEURON can import morphology.
     d['filename']=0
     #print d, row
     dl.append(d)
     cnt+=1


#csv[csv.repo=='Markram']

class Nposition(object):
    
    def __init__(self):
        with open('cngrid.csv', 'rb') as csvfile:#First just read the values.
            atlasreader = csv.reader(csvfile,delimiter=',')#, fieldname='x'
            #atlasreader = csv.DictReader(csvfile)#,delimiter=',')#, fieldname='x'
            #for row in list(csv.reader(open("copy-john.csv", "rb")))[1:]:
            for row in atlasreader:
               self.x=row[0]
               self.y=row[1]
               self.z=row[2]
               self.name=row[3]
               self.repo=row[4]
               
    def get(self):
       return (self.repo, self.name, self.x, self.y, self.z)            
               
               
               
               
#               #print row
     # for key in row:
      #   #print 'key=%s, value=%s' % (key, row[key]) 
      
    #for row in atlasreader:
        
        ##print row
"""
