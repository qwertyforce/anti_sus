from sklearn.mixture import GaussianMixture
import lmdb
import numpy as np
import lmdb
from tqdm import tqdm

DB_features = lmdb.open('./clean/features.lmdb/',map_size=1200*1_000_000) #5000mb
    
def get_all_data(db, size=20000):
    with db.begin(buffers=True) as txn:
        with txn.cursor() as curs:
            features = [] # np.zeros((batch_size,dim),np.float32)
            i=0
            for data in tqdm(curs.iternext(keys=True, values=True)):
                if i>=size:
                    break
                features.append(np.frombuffer(data[1],dtype=np.float32))
                i+=1
            return features

all_features = np.array(get_all_data(DB_features, 300000))
gmm = GaussianMixture(n_components = 16, covariance_type = 'full')
gmm.fit(all_features)

import pickle 
with open("./gmm.model","wb") as file:
    pickle.dump(gmm,file)