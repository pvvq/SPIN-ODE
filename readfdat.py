import re

import numpy as np

# from ROOT_Monitor.f90 produced by kpp
f90_spc_names_text = """
  CHARACTER(LEN=32), PARAMETER, DIMENSION(44) :: SPC_NAMES = (/ &
     'fragdummy                       ','T_RO_O4                         ','NO2                             ', & ! index 1 - 3
     'T_O0_NO3                        ','T_O2_NO3                        ','T_O1_NO3                        ', & ! index 4 - 6
     'T_O3_NO3                        ','Tuni_O1_O                       ','Tuni_O3_O                       ', & ! index 7 - 9
     'Tuni_O2_O                       ','Tuni_O4_O                       ','T_O0_OOH                        ', & ! index 10 - 12
     'T_O2_OOH                        ','T_O1_OOH                        ','T_O3_OOH                        ', & ! index 13 - 15
     'T_O1_2OH                        ','T_O0_2OH                        ','T_O2_2OH                        ', & ! index 16 - 18
     'T_O0_O                          ','T_O2_O                          ','T_O1_O                          ', & ! index 19 - 21
     'T_O3_O                          ','TO1_TO1                         ','TO1_TO3                         ', & ! index 22 - 24
     'TO1_TO2                         ','TO1_TO4                         ','TO3_TO3                         ', & ! index 25 - 27
     'TO3_TO2                         ','TO3_TO4                         ','TO2_TO2                         ', & ! index 28 - 30
     'TO2_TO4                         ','TO4_TO4                         ','T_RO_O3                         ', & ! index 31 - 33
     'T_RO_O2                         ','O3                              ','TOY                             ', & ! index 34 - 36
     'T_RO_O1                         ','OH                              ','NO                              ', & ! index 37 - 39
     'T_RO2_O2                        ','T_RO2_O4                        ','T_RO2_O3                        ', & ! index 40 - 42
     'T_RO2_O5                        ','HO2                             ' /) ! index up to 44
"""


if __name__ == "__main__":
    spc_names = [s.strip() for s in re.findall(r"'([^']+)'", f90_spc_names_text)]
    print(len(spc_names))
    print(spc_names)
    dat = np.loadtxt("toy_44.dat")
    print(dat.shape)
    np.savetxt("toy_44.csv", dat, delimiter=',', header="time,"+",".join(spc_names))