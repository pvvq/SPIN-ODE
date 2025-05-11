import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.integrate import odeint, solve_ivp

# https://github.com/Xiangjun-Huang/training_stiff_NODE_in_WW_modelling/blob/348f32da56a86ae57462910a7eaab2352a014e1f/ASM1_Python/collocate_data_torch.py
# from collocate_data_torch import collocate_data_torch

# from readfdat import csv_from_dat

# K literal for toy 44 scheme
TEMP = 270
K_LITERAL = {
    'KRO2NO': 2.7E-12*np.exp(360./TEMP),
    'KRO2HO2': 2.91E-13*np.exp(1300./TEMP),
    'KDEC': 1.00E+06,
    'RO2': 1,  # Replace as 1 for now, use concentration later
}
# RO2 = C(ind_T_RO2_O2) + C(ind_T_RO2_O4) + C(ind_T_RO2_O3) + C(ind_T_RO2_O5)
RO2_SPC = ['T_RO2_O2', 'T_RO2_O4', 'T_RO2_O3', 'T_RO2_O5']

SORT_RATE = False

def_rober_text = \
"""
{1.} A = B : 0.04 ; 
{2.} B + B = C + B : 3*1e7 ; 
{3.} B + C = A + C : 1e4 ; 
"""

def_pollu_text = \
"""
{1.} NO2 = NO + O3P : 0.350E+00 ;  
{2.} NO + O3 = NO2 : 0.266E+02 ;  
{3.} HO2 + NO = NO2 + OH : 0.120E+05 ;  
{4.} HCHO = HO2 + HO2 + CO : 0.860E-03 ;  
{5.} HCHO = CO : 0.820E-03 ;  
{6.} HCHO + OH = HO2 + CO : 0.150E+05 ;  
{7.} ALD = MEO2 + HO2 + CO : 0.130E-03 ;  
{8.} ALD + OH = C2O3 : 0.240E+05 ;  
{9.} C2O3 + NO = NO2 + MEO2 + CO2 : 0.165E+05 ;  
{10.} C2O3 + NO2 = PAN : 0.900E+04 ;  
{11.} PAN = C2O3 + NO2 : 0.220E-01 ;  
{12.} MEO2 + NO = CH3O + NO2 : 0.120E+05 ;  
{13.} CH3O = HCHO + HO2 : 0.188E+01 ;  
{14.} NO2 + OH = HNO3 : 0.163E+05 ;  
{15.} O3P = O3 : 0.480E+07 ;  
{16.} O3 = O1D : 0.350E-03 ;  
{17.} O3 = O3P : 0.175E-01 ;  
{18.} O1D = OH + OH : 0.100E+09 ;  
{19.} O1D = O3P : 0.444E+12 ;  
{20.} SO2 + OH = SO4 + HO2 : 0.124E+04 ;  
{21.} NO3 = NO : 0.210E+01 ;  
{22.} NO3 = NO2 + O3P : 0.578E+01 ;  
{23.} NO2 + O3 = NO3 : 0.474E-01 ;  
{24.} NO3 + NO2 = N2O5 : 0.178E+04 ;  
{25.} N2O5 = NO3 + NO2 : 0.312E+01 ;  
"""

def_toy_44_text = """
#EQUATIONS
 
 
// reactions that get species from MCM to PRAM:
// TOY-VOC reacts with OH:
{0.}	TOY + OH = T_RO2_O3	:	1.e-11 ;
{1.}	TOY + O3 = T_RO2_O2 + OH	:	1.e-15 ;
 
//Autoxidation: RO2 -> H-shift + O2 addition -> RO2 
{2.}      T_RO2_O2 = T_RO2_O4 : 		1.0; 
{3.}      T_RO2_O3 = T_RO2_O5 : 		1.0; 

//Peroxy radicals reacting with NO to form RNO3: -> RO2 + NO -> R-NO3 
{4.}      T_RO2_O2 + NO = T_O0_NO3 : 		KRO2NO*0.4 ; 
{5.}      T_RO2_O4 + NO = T_O2_NO3 : 		KRO2NO*0.4 ; 
{6.}      T_RO2_O3 + NO = T_O1_NO3 : 		KRO2NO*0.4 ; 
{7.}      T_RO2_O5 + NO = T_O3_NO3 : 		KRO2NO*0.4 ; 

//Peroxy radicals reacting with NO to form RO: -> RO2 + NO -> RO 
{8.}      T_RO2_O2 + NO = T_RO_O1 + NO2 : 		KRO2NO*0.6 ; 
{9.}      T_RO2_O4 + NO = T_RO_O3 + NO2 : 		KRO2NO*0.6 ; 
{10.}      T_RO2_O3 + NO = T_RO_O2 + NO2 : 		KRO2NO*0.6 ; 
{11.}      T_RO2_O5 + NO = T_RO_O4 + NO2 : 		KRO2NO*0.6 ; 

//RO2 formation from RO: RO -> RO2 (#O -> #O+2 
{12.}      T_RO_O4 = fragdummy : 		KDEC; 
{13.}      T_RO_O2 = T_RO2_O4 : 		KDEC; 
{14.}      T_RO_O1 = T_RO2_O3 : 		KDEC; 
{15.}      T_RO_O3 = T_RO2_O5 : 		KDEC; 

//RO2 radicals abstracting H from alpha hydroxyl carbon -> RC=O 
{16.}      T_RO2_O2 = Tuni_O1_O : 		 1.0 ; 
{17.}      T_RO2_O4 = Tuni_O3_O : 		 1.0 ; 
{18.}      T_RO2_O3 = Tuni_O2_O : 		 1.0 ; 
{19.}      T_RO2_O5 = Tuni_O4_O : 		 1.0 ; 

//Peroxy radicals reacting with HO2 forming closed shell with -OOH: RO2 + HO2 -> ROOH 
{20.}      T_RO2_O2 + HO2 = T_O0_OOH : 		KRO2HO2*0.9 ; 
{21.}      T_RO2_O4 + HO2 = T_O2_OOH : 		KRO2HO2*0.9 ; 
{22.}      T_RO2_O3 + HO2 = T_O1_OOH : 		KRO2HO2*0.9 ; 
{23.}      T_RO2_O5 + HO2 = T_O3_OOH : 		KRO2HO2*0.9 ; 

//RO2 reacting with HO2 forming RO + O2 + OH: 
//RO2* + HO2 -> RO* + O2 + OH 
{24.}      T_RO2_O2 + HO2 = T_RO_O1 + OH : 		KRO2HO2*0.1; 
{25.}      T_RO2_O4 + HO2 = T_RO_O3 + OH : 		KRO2HO2*0.1; 
{26.}      T_RO2_O3 + HO2 = T_RO_O2 + OH : 		KRO2HO2*0.1; 
{27.}      T_RO2_O5 + HO2 = T_RO_O4 + OH : 		KRO2HO2*0.1; 

//RO2 reacting with sum of RO2s forming ROH by O removal and H addition: 
//RO2" + sum(RO2) -> ROH"; CxHyOz -> CxH(y+1)O(z-1) 
//{28.}      T_RO2_O2 = T_O-1_2OH : 		RO2*1e-13*0.3 ; 
{29.}      T_RO2_O4 = T_O1_2OH : 		RO2*1e-13*0.3 ; 
{30.}      T_RO2_O3 = T_O0_2OH : 		RO2*1e-13*0.3 ; 
{31.}      T_RO2_O5 = T_O2_2OH : 		RO2*1e-13*0.3 ; 

//RO2 reacting with sum of RO2s forming alkoxy radical by O removal: 
//RO2* + sum(RO2) -> RO* 
{32.}      T_RO2_O2 = T_RO_O1 : 		RO2*1e-13*0.3; 
{33.}      T_RO2_O4 = T_RO_O3 : 		RO2*1e-13*0.3; 
{34.}      T_RO2_O3 = T_RO_O2 : 		RO2*1e-13*0.3; 
{35.}      T_RO2_O5 = T_RO_O4 : 		RO2*1e-13*0.3; 

//RO2 reacting with sum of RO2s forming R=O by OH removal: 
//RO2" + sum(RO2) -> R=O"; CxHyOz -> CxH(y-1)O(z-1) 
{36.}      T_RO2_O2 = T_O0_O : 		RO2*1e-13*0.4 ; 
{37.}      T_RO2_O4 = T_O2_O : 		RO2*1e-13*0.4 ; 
{38.}      T_RO2_O3 = T_O1_O : 		RO2*1e-13*0.4 ; 
{39.}      T_RO2_O5 = T_O3_O : 		RO2*1e-13*0.4 ; 

//RO2 dimer formation: RO2 + RO2" -> ROOR" 
// ROOR formation: T_RO2_O2
{40.}      T_RO2_O2 + T_RO2_O2 = TO1_TO1 : 		1e-13*1.0 ; 
{41.}      T_RO2_O2 + T_RO2_O4 = TO1_TO3 : 		1e-13*1.0 ; 
{42.}      T_RO2_O2 + T_RO2_O3 = TO1_TO2 : 		1e-13*1.0 ; 
{43.}      T_RO2_O2 + T_RO2_O5 = TO1_TO4 : 		1e-13*1.0 ; 
// ROOR formation: T_RO2_O4
{44.}      T_RO2_O4 + T_RO2_O4 = TO3_TO3 : 		1e-13*1.0 ; 
{45.}      T_RO2_O4 + T_RO2_O3 = TO3_TO2 : 		1e-13*1.0 ; 
{46.}      T_RO2_O4 + T_RO2_O5 = TO3_TO4 : 		1e-13*1.0 ; 
// ROOR formation: T_RO2_O3
{47.}      T_RO2_O3 + T_RO2_O3 = TO2_TO2 : 		1e-13*1.0 ; 
{48.}      T_RO2_O3 + T_RO2_O5 = TO2_TO4 : 		1e-13*1.0 ; 
// ROOR formation: T_RO2_O5
{49.}      T_RO2_O5 + T_RO2_O5 = TO4_TO4 : 		1e-13*1.0 ; 
"""

class ChemistryScheme():
    """
    Reaction
    
    spc_name: [list] of species names
    reactions_text: [[[r1_r1, r1_r2], [r1_p1, r1_p2], rate1], ...]
    reactions_id: [[[r1_r1, r1_r2], [r1_p1, r1_p2], rate1], ...]
    rconst_text: [list] of rate constant in text
    rconst: [list] of rate constant in number
    """
    def __init__(self, scheme_id):
        self.RO2_IDX = None    # index of RO2 spcies, for TOY scheme only
        self.RO2_K_IDX = None  # index of k contains variable 'RO2 ', for TOY scheme only
        self.spc_names, self.reactions_text, self.reactions_id, self.rconst_text, self.rconst = self.parse_scheme(scheme_id)
        self.num_spc = len(self.spc_names)
        self.num_react = len(self.reactions_id)
        self.coef_in, self.coef_out = self.get_react_coef()

        # seperate too large rate constant
        if scheme_id == "toy_44" and SORT_RATE:
            sort_order = np.argsort(self.rconst)
            self.rconst = self.rconst[sort_order]
            self.coef_out = self.coef_out[:, sort_order]
            self.coef_in = self.coef_in[:, sort_order]
            self.RO2_K_IDX = [list(sort_order).index(idx) for idx in self.RO2_K_IDX]

    def parse_scheme(self, scheme_id):
        if scheme_id == "toy_44":
            # filename = scheme_id+".def"
            # with open(filename) as f:
            #     def_text = [line for line in f]
            def_text = def_toy_44_text.split('\n')
        elif scheme_id == "rober":
            def_text = def_rober_text.split('\n')
        elif scheme_id == "pollu":
            def_text = def_pollu_text.split('\n')

        species = []
        reactions_text = []
        reactions_id = []
        rconst_text = []
        rconst = []
        for line in def_text:
            if '=' not in line or ':' not in line or "//" in line:
                continue
            texts = line.strip("<>{}0123456789.").split(':')
            
            reaction = texts[0].split('=')
            reactants = [reactant.strip() for reactant in reaction[0].split('+')]
            products = [product.strip() for product in reaction[1].split('+')]
            
            rate = texts[1].strip(" ;\t\n")
            reactions_text.append([reactants, products, rate])

        # Species names
        for reactants, products, rate in reactions_text:
            for reactant in reactants:
                if reactant not in species:
                    species.append(reactant)
            for product in products:
                if product not in species:  
                    species.append(product)
            rconst_text.append(rate)
            
        # For toy44, replace K literal
        if scheme_id == "toy_44":
            self.RO2_IDX = []
            self.RO2_K_IDX = []
            for i, text in enumerate(rconst_text):
                for t in text.split('*'):
                    if t == "RO2":
                        self.RO2_K_IDX.append(i)
                    if t in K_LITERAL.keys():
                        text = text.replace(t, str(K_LITERAL[t]))
                rconst.append(eval(text))
            for spc in RO2_SPC:
                self.RO2_IDX.append(species.index(spc))
        else:
            for i, text in enumerate(rconst_text):
                rconst.append(eval(text))
            
        # print(reactions_text)
        # convert spc text to idx
        for reactants, products, rate in reactions_text:
            reactants_id = [species.index(reactant) for reactant in reactants]
            products_id = [species.index(product) for product in products]
            reactions_id.append([reactants_id, products_id, rate])
        return species, reactions_text, reactions_id, rconst_text, np.array(rconst)

    def get_react_coef(self):
        coef_in = np.zeros([self.num_spc, self.num_react])  # reaction order
        coef_out = np.zeros([self.num_spc, self.num_react])  # stoichiometrix
        for i, (reactants, products, _) in enumerate(self.reactions_id):
            for reactant in reactants:
                coef_in[reactant][i] += 1
                coef_out[reactant][i] -= 1
            for product in products:
                coef_out[product][i] += 1
        return coef_in, coef_out


class ROBER(ChemistryScheme):
    def __init__(self):
        super().__init__(scheme_id="rober")

    @staticmethod
    def ode(t, y):
        k = np.array([0.04, 3e7, 1e4])
        dydt = np.array([
            -k[0]*y[0] + k[2]*y[1]*y[2],
             k[0]*y[0] - k[2]*y[1]*y[2] - k[1]*y[1]*y[1],
             k[1]*y[1]*y[1]
        ])
        return dydt

    def data(self, num_series, rand=False):  # rand unimplement
        y_list, t_list = [], []
        for i in range(num_series):
            y0 = np.array([1.0, 0.0, 0.0])
            t = np.logspace(-5, 5, num=50)
            # ode_np = lambda t, y: np.array(ROBER.ode(t, jnp.array(y)))  # slow!
            sol = solve_ivp(fun=ROBER.ode, t_span=(0, t[-1]), y0=y0, method="BDF", t_eval=t, rtol=1e-4, atol=[1.0e-8, 1.0e-14, 1.0e-6])
            y_list.append(sol.y.transpose(1, 0))
            t_list.append(t)
        return np.array(y_list), np.array(t_list)

class POLLU(ChemistryScheme):
    def __init__(self):
        super().__init__(scheme_id="pollu")
        self.rng = np.random.RandomState(42)

    @staticmethod
    def ode(t, y):
        k = np.array([
                0.35e0, 0.266e2, 0.123e5, 0.86e-3, 0.82e-3, 0.15e5, 0.13e-3, 0.24e5, 0.165e5, 0.9e4,
                0.22e-1, 0.12e5, 0.188e1, 0.163e5, 0.48e7, 0.35e-3, 0.175e-1, 0.1e9, 0.444e12, 0.124e4,
                0.21e1, 0.578e1, 0.474e-1, 0.178e4, 0.312e1
        ])
    
        r = np.array([
            k[0] * y[0],
            k[1] * y[1] * y[3],
            k[2] * y[4] * y[1],
            k[3] * y[6],
            k[4] * y[6],
            k[5] * y[6] * y[5],
            k[6] * y[8],
            k[7] * y[8] * y[5],
            k[8] * y[10] * y[1],
            k[9] * y[10] * y[0],
            k[10] * y[12],
            k[11] * y[9] * y[1],
            k[12] * y[13],
            k[13] * y[0] * y[5],
            k[14] * y[2],
            k[15] * y[3],
            k[16] * y[3],
            k[17] * y[15],
            k[18] * y[15],
            k[19] * y[16] * y[5],
            k[20] * y[18],
            k[21] * y[18],
            k[22] * y[0] * y[3],
            k[23] * y[18] * y[0],
            k[24] * y[19]
        ])
    
        dydt = np.array([
            -r[0] - r[9] - r[13] - r[22] - r[23] + r[1] + r[2] + r[8] + r[10] + r[11] + r[21] + r[24],
            -r[1] - r[2] - r[8] - r[11] + r[0] + r[20],
            -r[14] + r[0] + r[16] + r[18] + r[21],
            -r[1] - r[15] - r[16] - r[22] + r[14],
            -r[2] + r[3] + r[3] + r[5] + r[6] + r[12] + r[19],
            -r[5] - r[7] - r[13] - r[19] + r[2] + r[17] + r[17],
            -r[3] - r[4] - r[5] + r[12],
            r[3] + r[4] + r[5] + r[6],
            -r[6] - r[7],
            -r[11] + r[6] + r[8],
            -r[8] - r[9] + r[7] + r[10],
            r[8],
            -r[10] + r[9],
            -r[12] + r[11],
            r[13],
            -r[17] - r[18] + r[15],
            -r[19],
            r[19],
            -r[20] - r[21] - r[23] + r[22] + r[24],
            -r[24] + r[23]
        ])
        return dydt

    def data(self, num_series, rand=False):
        y_list, t_list = [], []
        for i in range(num_series):
            # Initial conditions
            u0 = np.zeros(20)
            if rand:
                u0[[1,3,6,7,8,16]] = [0.2, 0.04, 0.1, 0.3, 0.01, 0.007] * self.rng.uniform(0.99, 1.01, size=(6,))  # rand by 0.1
            else:
                u0[[1,3,6,7,8,16]] = [0.2, 0.04, 0.1, 0.3, 0.01, 0.007]  # no rand init
            t = np.linspace(0, 0.1, num=100)
            # ode_np = lambda t, y: np.array(POLLU.ode(t, jnp.array(y)))  # slow!
            sol = solve_ivp(fun=POLLU.ode, t_span=(0, t[-1]), y0=u0, method="BDF", t_eval=t)
            y_list.append(sol.y.transpose(1, 0))
            t_list.append(t)
        return np.array(y_list), np.array(t_list)

class TOY(ChemistryScheme):
    def __init__(self, data_dir):
        super().__init__(scheme_id="toy_44")
        self.data_dir = data_dir

    def data(self, num_series, rand=False):  # rand unimplement
        """Load data simulated by KPP"""
        y_list, t_list = [], []
        for i in range(num_series):
            csv_file = f"{self.data_dir}/sed_{i+1}.csv"
            try:
                with open(csv_file, 'r') as f:
                    header = (f.readline().strip("# \n")).split(',')
            except FileNotFoundError as e:
                print(f"File not found: {e}")
                print("Maybe requested more than exist?")
            permutation = [header.index(spc) for spc in self.spc_names]
            series = np.loadtxt(csv_file, delimiter=',', skiprows=1)
            t_list.append(series[:, 0]*3600)
            y_list.append(series[:, permutation])
        
        return np.array(y_list), np.array(t_list)

class AEDataset(Dataset):
    def __init__(self, y, t):
        self.y = torch.tensor(y)
        self.t = torch.tensor(t)
        self.num_series = self.t.shape[0]
        self.num_steps = self.t.shape[1]
        self.size = self.num_series * self.num_steps
        self.series_idxs = torch.randint(self.num_series, (self.size,))
        self.step_idxs = torch.randint(4, self.num_steps-5, (self.size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        i_series = self.series_idxs[idx]
        current_t = self.step_idxs[idx]
        # history_t = torch.randint(0, current_t, (4,))
        # history_t.sort()
        history_dt = torch.arange(-4, 1)
        # future_t = torch.randint(current_t+1, self.num_steps, (5,))
        future_dt = np.arange(1, 6, 1)
        # np.random.shuffle(future_dt)
        return {
            'history_dt': history_dt ,
            'future_dt' : future_dt,
            'global':    self.y[i_series].transpose(1,0),
            'initial':   self.y[i_series][current_t + history_dt].transpose(1,0),
            'target':    self.y[i_series][current_t + future_dt].transpose(1,0),
        }

class ChuckDataset(Dataset):
    """Break trajectories into chucks with optional slicing and random sample"""
    def __init__(self, ny, nt, chuck_len=None, stride_len=1, ratio=1.0):
        """
        Init

        Args:
            check_len: len of chuck
            stride_len: stride of the sliding chuck (no stride if check_len==series_len)
            ratio: randomly sample a ratio of points in chuck
        """
        self.ny = ny  # [n_series, t, n_spc]
        self.nt = nt  # [n_series, t]
        self.n_series = self.nt.shape[0]
        self.series_len = self.nt.shape[1]
        self.chuck_len = self.series_len if chuck_len is None else chuck_len
        self.stride_len = stride_len
        self.chuck_per_serie = (self.series_len - self.chuck_len) // self.stride_len + 1
        self.total_chuck = self.n_series * self.chuck_per_serie
        self.samples_per_chuck = int(self.chuck_len * ratio)
        self.rand_sample()

    def rand_sample(self):
        """shuffle the random index within the chuck, shared across batch"""
        self.idx_sample = np.sort(np.random.choice(range(0, self.chuck_len), size=self.samples_per_chuck, replace=False))

    def __len__(self):
        return self.total_chuck

    def __getitem__(self, idx):
        idx_series = idx // self.chuck_per_serie
        idx_chuck = idx % self.chuck_per_serie
        start = idx_chuck * self.stride_len

        return {
            "conc": self.ny[idx_series, start+self.idx_sample],  # [seq_len, n_spc]
            "time": self.nt[idx_series, start+self.idx_sample],  # [seq_len]
        }


class CollocateDataset(Dataset):
    def __init__(self, y_arr, t_arr, dy_arr=None, diff="finite"):
        self.t_arr = torch.tensor(t_arr)  # [n, n_t]
        self.y_arr = torch.tensor(y_arr)  # [n, n_t, n_spc]
        self.num_series = y_arr.shape[0]
        self.num_spc = y_arr.shape[2]
        if dy_arr is not None:
            self.dy_arr = torch.tensor(dy_arr)  # [n, n_t, n_spc]
            self.prepare_no_transform_data()
        else:
            if diff=="local_reg":
                self.prepare_collocate_data()    # opt 1: non-parametric regression
            elif diff=="finite":
                self.prepare_finite_diff_data()  # opt 2: finite diff

    def prepare_collocate_data(self):
        # compute dy use y and t
        yy_list = []
        dy_list = []
        for i in range(self.num_series):
            yy, dy = collocate_data_torch(self.t_arr[i], self.y_arr[i].unsqueeze(dim=1), kernel_str="LogisticKernel")
            yy_list.append(yy.transpose(1, 0))
            dy_list.append(dy.transpose(1, 0))
        self.yy_arr = torch.cat(yy_list, dim=0)
        self.dy_arr = torch.cat(dy_list, dim=0)
        tunc_idx = 3  # omit margin which are usually not well fitted
        self.yy_flat = self.yy_arr[:,tunc_idx:-tunc_idx].reshape(-1, self.num_spc)  # [n, n_spc]
        self.dy_flat = self.dy_arr[:,tunc_idx:-tunc_idx].reshape(-1, self.num_spc)  # [n, n_spc]
        self.tt_flat = self.t_arr[:,tunc_idx:-tunc_idx].reshape(-1, 1)  # [n, 1]

    def prepare_finite_diff_data(self):
        dy_list = []
        for i in range(self.num_series):
            dy_list.append(torch.gradient(self.y_arr[i], dim=0, spacing=(self.t_arr[i],))[0].unsqueeze(0))
        self.dy_arr = torch.cat(dy_list, dim=0)
        self.tt_flat = self.t_arr.reshape(-1, 1)  # [n, n_spc]
        self.yy_flat = self.y_arr.reshape(-1, self.num_spc)  # [n, n_spc]
        self.dy_flat = self.dy_arr.reshape(-1, self.num_spc)  # [n, n_spc]

    def prepare_no_transform_data(self):
        self.tt_flat = self.t_arr.reshape(-1, 1)  # [n, n_spc]
        self.yy_flat = self.y_arr.reshape(-1, self.num_spc)  # [n, n_spc]
        self.dy_flat = self.dy_arr.reshape(-1, self.num_spc)  # [n, n_spc]

    def __len__(self):
        return self.dy_flat.shape[0]

    def __getitem__(self, idx):
        return {'time':self.tt_flat[idx], 'conc':self.yy_flat[idx], 'dcdt':self.dy_flat[idx]}

