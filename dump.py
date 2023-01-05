

import pickle
from bson.json_util import dumps as to_json
from bson.json_util import loads as from_json

import orion.storage.legacy


def dump_pkl(path):
    bad_types = {
        orion.storage.legacy.Legacy
    }
    
    
    def convert_bad_types(v):
        obj = {}
        
        for k in dir(v):
            attr = getattr(v, k)
            obj[k] = str(attr)
        
    
        return {str(type(v)): obj}
    
    with open(path, "rb") as f:
        data = f.read()
        
        database = pickle.loads(data)
        
        data = database._db
        
        ok_dict = dict()
        
        for k, v in data.items():
            collections = []
            ok_dict[k] = collections
            
            for val in v._documents:
                obj = dict()
                collections.append(obj)
                
                for kk, vv in val._data.items():
                    value = vv if type(vv) not in bad_types else convert_bad_types(vv)
                
                    obj[kk] = value
                
        
        with open('dump.json', 'w') as ff:
            ff.write(to_json(ok_dict, indent=2))

c = '/home/newton/work/orion/examples/tutorials/current_db.pkl'
p = '/home/newton/work/orion/examples/tutorials/previous_db.pkl'

dump_pkl(c)