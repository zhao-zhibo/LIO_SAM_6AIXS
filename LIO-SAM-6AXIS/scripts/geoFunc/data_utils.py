import numpy as np
import re
import geoFunc.trans as trans
import math

def loadIE(path : str, all_data : dict, Ri0i1 : np.ndarray, ref_xyz : np.ndarray = None):
    fp = open(path,'rt')

    if ref_xyz is None:
        is_ref_set  = False
    else:
        is_ref_set = True
        Ten0 = np.eye(4,4)
        Ten0[0:3,0:3] = trans.Cen(ref_xyz)
        Ten0[0:3,3] = ref_xyz

    while True:
        line = fp.readline().strip()
        if line == '':break
        if line[0] == '#' :continue
        line = re.sub('\s\s+',' ',line)
        elem = line.split(' ')
        sod = float(elem[1])
        if sod not in all_data.keys():
            all_data[sod] ={}
        all_data[sod]['X0']   = float(elem[2])
        all_data[sod]['Y0']   = float(elem[3])
        all_data[sod]['Z0']   = float(elem[4])
        all_data[sod]['VX0']  = float(elem[15])
        all_data[sod]['VY0']  = float(elem[16])
        all_data[sod]['VZ0']  = float(elem[17])
        all_data[sod]['ATTX0']= float(elem[25])
        all_data[sod]['ATTY0']= float(elem[26])
        all_data[sod]['ATTZ0']= -float(elem[24])
        Ren = trans.Cen([all_data[sod]['X0'],all_data[sod]['Y0'],all_data[sod]['Z0']])
        ani0 = [all_data[sod]['ATTX0']/180*math.pi,\
                all_data[sod]['ATTY0']/180*math.pi,\
                all_data[sod]['ATTZ0']/180*math.pi]
        Rni0 = trans.att2m(ani0)
        Rni1 = np.matmul(Rni0,Ri0i1)
        Rni1= Rni0
        Rei1 = np.matmul(Ren,Rni1)
        tei1 = np.array([all_data[sod]['X0'],all_data[sod]['Y0'],all_data[sod]['Z0']])
        Tei1 = np.eye(4,4)
        Tei1[0:3,0:3] = Rei1
        Tei1[0:3,3] = tei1
        if not is_ref_set:
            is_ref_set = True
            Ten0 = np.eye(4,4)
            Ten0[0:3,0:3] = trans.Cen(tei1)
            Ten0[0:3,3] = tei1
        Tn0i = np.matmul(np.linalg.inv(Ten0),Tei1)
        all_data[sod]['T'] = Tn0i
    fp.close()
    return Ten0[0:3,3]