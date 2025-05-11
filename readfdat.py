import fortranformat as ff
import numpy as np
import matplotlib.pyplot as plt
import sys


toy_44_spc_old = """T_RO_O0                         NO2                             T_O0_NO3                        T_O2_NO3                        T_O1_NO3                        T_O3_NO3            
   T_RO_O4                         Tuni_O1_O                       Tuni_O3_O                       Tuni_O2_O                       Tuni_O4_O                       T_O0_OOH                   
     T_O2_OOH                        T_O1_OOH                        T_O3_OOH                        T_O1_2OH                        T_O0_2OH                        T_O2_2OH                 
       T_O0_O                          T_O2_O                          T_O1_O                          T_O3_O                          TO1_TO1                         TO1_TO3                
         TO1_TO2                         TO1_TO4                         TO3_TO3                         TO3_TO2                         TO3_TO4                         TO2_TO2              
           TO2_TO4                         TO4_TO4                         T_RO_O1                         T_RO_O3                         T_RO_O2                         O3                 
             TOY                             OH                              HO2                             T_RO2_O5                        NO                              T_RO2_O2         
               T_RO2_O4                        T_RO2_O3"""

toy_44_spc = """fragdummy                       T_RO_O4                         NO2                             T_O0_NO3                        T_O2_NO3     
                   T_O1_NO3                        T_O3_NO3                        Tuni_O1_O                       Tuni_O3_O                  
     Tuni_O2_O                       Tuni_O4_O                       T_O0_OOH                        T_O2_OOH                        T_O1_OOH 
                       T_O3_OOH                        T_O1_2OH                        T_O0_2OH                        T_O2_2OH               
         T_O0_O                          T_O2_O                          T_O1_O                          T_O3_O                          TO1_TO1                         TO1_TO3                         TO1_TO2                         TO1_TO4                         TO3_TO3            
             TO3_TO2                         TO3_TO4                         TO2_TO2                         TO2_TO4                         TO4_TO4                         T_RO_O3                         T_RO_O2                         O3                              TOY            
                 T_RO_O1                         OH                              NO                              T_RO2_O2                     
   T_RO2_O4                        T_RO2_O3                        T_RO2_O5                        HO2"""


def csv_from_dat(datfile, spc_name=toy_44_spc, fformat='(E24.16,100(1X,E24.16))'):
  spc_names = spc_name.split()
  id = datfile.split('.')[0]

  ff.config.RET_WRITTEN_VARS_ONLY = True
  ff_reader = ff.FortranRecordReader(fformat)

  record_list = []

  with open(datfile) as fdat:
    for line in fdat:
      record = ff_reader.read(line)
      record_list.append(record)
  
  data = np.array(record_list)
  result_file = f"{id}.csv"
  np.savetxt(result_file, data, delimiter=',', header="time,"+",".join(spc_names))
  return result_file



def plot_records(data, n_species):
  # n_species = data.shape[1]
  fig, ax = plt.subplots((10 if n_species > 10 else n_species), n_species//10+1,
    squeeze=False, layout='constrained',sharex=True)
  c = data.transpose(1,0)
  for i in range(n_species):
    ax[i%10][i//10].plot(c[i], label=header[i+1])
    ax[i%10][i//10].set_title(spc_names[i], fontsize='small', loc='left')
  fig.savefig(f"{id}.png", dpi=300)

if __name__ == "__main__":
  if len(sys.argv) > 1:
    datfile = sys.argv[1]
  if len(sys.argv) > 2:
    spc_names = sys.argv[2]
  
  for i in range(1,601):
    csv_from_dat(f"results_t10_dt1e-1/sed_{i}.dat")
    print(i, end=',')
  
  if len(sys.argv) > 3:
    plot_records(data[:, 1:], int(sys.argv[3]))