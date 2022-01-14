'''
    This library reads from a template,
    writes GX input files,
    and launches batch jobs

    7 Jan 2022
    tqian@pppl.gov
'''

class GX():

    def __init__(self,template):
        
        self.read(template)

    def read(self, fin):

        with open(fin) as f:
            data = f.readlines()

        obj = {}
        header = ''
        for line in data:

            # skip comments
            if line.find('#') > -1:
                continue

            # parse headers
            if line.find('[') == 0:
                header = line.split('[')[1].split(']')[0]
                obj[header] = {}
                continue

            # skip blanks
            if line.find('=') < 0:
                continue

            # store data
            key, value = line.split('=')
            key   = key.strip()
            value = value.strip()
            
            if header == '':
                obj[key] = value
            else:
                obj[header][key] = value

        self.inputs = obj

    def load_geometry(self,geo_in):
        
        # get ntheta from file string produced by GX geometry module
        ntheta = geo_in.split('nt')[-1].split('_')[1]

        self.inputs['Dimensions']['ntheta'] = ntheta
        self.inputs['Geometry']['geofile']  = geo_in


    def write(self, fout='temp.in'):

        with open(fout,'w') as f:
        
            for item in self.inputs.items():
                 
                if ( type(item[1]) is not dict ):
                    print('  %s = %s ' % item, file=f)  
                    continue
    
                header, nest = item
                print('\n[%s]' % header, file=f)
    
                longest_key =  max( nest.keys(), key=len) 
                N_space = len(longest_key) 
                for pair in nest.items():
                    s = '  {:%i}  =  {}' % N_space
                    print(s.format(*pair), file=f)

        print('  wrote to:', fout)



import sys
fin = sys.argv[1]

g = GX(fin)
g.write()

import pdb
pdb.set_trace()
