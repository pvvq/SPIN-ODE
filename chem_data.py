from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.integrate import solve_ivp
from torch.utils.data import default_collate
import jax
import jax.numpy as jnp

# https://github.com/Xiangjun-Huang/training_stiff_NODE_in_WW_modelling/blob/348f32da56a86ae57462910a7eaab2352a014e1f/ASM1_Python/collocate_data_torch.py
# from collocate_data_torch import collocate_data_torch

TEMP = 270
K_LITERAL = {
    'KRO2NO': 2.7E-12*np.exp(360./TEMP),
    'KRO2HO2': 2.91E-13*np.exp(1300./TEMP),
    'KDEC': 1.00E+06,
    'RO2': 1,  # Replace as 1 for now, use concentration later
}

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

toy_44_init_conc_text = """
  TOY = 3.10E+11 ;
  OH = 1.90E+5 ;
  O3  = 1.80E+12 ;
  HO2 = 4.30E+7 ;
  NO  = 1.10E+8 ;
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
        self.stoi_reac, self.stoi_prod = self.get_stoichiometric_coef()

        # seperate too large rate constant
        if scheme_id == "toy_44" and SORT_RATE:
            sort_order = np.argsort(self.rconst)
            self.rconst = self.rconst[sort_order]
            self.stoi_prod = self.stoi_prod[:, sort_order]
            self.stoi_reac = self.stoi_reac[:, sort_order]
            self.RO2_K_IDX = [list(sort_order).index(idx) for idx in self.RO2_K_IDX]

    def parse_scheme(self, scheme_id):
        if scheme_id == "toy_44":
            def_text = def_toy_44_text.split('\n')
            # RO2 = C(ind_T_RO2_O2) + C(ind_T_RO2_O4) + C(ind_T_RO2_O3) + C(ind_T_RO2_O5)
            RO2_SPC = ['T_RO2_O2', 'T_RO2_O4', 'T_RO2_O3', 'T_RO2_O5']
        elif scheme_id == "rober":
            def_text = def_rober_text.split('\n')
            RO2_SPC = []
        elif scheme_id == "pollu":
            def_text = def_pollu_text.split('\n')
            RO2_SPC = []

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
            
        # replace K literal
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

        # convert spc text to idx
        for reactants, products, rate in reactions_text:
            reactants_id = [species.index(reactant) for reactant in reactants]
            products_id = [species.index(product) for product in products]
            reactions_id.append([reactants_id, products_id, rate])
        return species, reactions_text, reactions_id, rconst_text, np.array(rconst)

    def get_stoichiometric_coef(self):
        stoi_reac = np.zeros([self.num_spc, self.num_react])  # reaction order
        stoi_prod = np.zeros([self.num_spc, self.num_react])  # stoichiometrix
        for i, (reactants, products, _) in enumerate(self.reactions_id):
            for reactant in reactants:
                stoi_reac[reactant][i] += 1
            for product in products:
                stoi_prod[product][i] += 1
        return stoi_reac, stoi_prod
    
    def rate_ode(self, t, y):
        RO2 = np.sum(y[self.RO2_IDX])
        rate = self.rconst * np.prod(y[:, None] ** self.stoi_reac, axis=0)
        rate[self.RO2_K_IDX] *= RO2
        dc_dt  = (self.stoi_prod - self.stoi_reac) @ rate
        return dc_dt


class ROBER(ChemistryScheme):
    def __init__(self):
        super().__init__(scheme_id="rober")

    def data(self, num_series, rand=False):  # rand unimplement
        y_list, t_list = [], []
        for i in range(num_series):
            y0 = np.array([1.0, 0.0, 0.0])
            t = np.logspace(-5, 5, num=50)
            sol = solve_ivp(fun=self.rate_ode, t_span=(0, t[-1]), y0=y0, method="BDF", t_eval=t, rtol=1e-4, atol=[1.0e-8, 1.0e-14, 1.0e-6])
            y_list.append(sol.y.transpose(1, 0))
            t_list.append(t)
        return np.array(y_list), np.array(t_list)

class POLLU(ChemistryScheme):
    def __init__(self):
        super().__init__(scheme_id="pollu")
        self.rng = np.random.RandomState(42)

    def data(self, num_series, rand=False):
        y_list, t_list = [], []
        for i in range(num_series):
            # Initial conditions
            u0 = np.zeros(20)
            u0[[1,3,6,7,8,16]] = [0.2, 0.04, 0.1, 0.3, 0.01, 0.007]
            if rand:
                u0 *= self.rng.uniform(0.99, 1.01, size=u0.shape)  # rand by 0.1
            # t = np.linspace(0, 0.1, num=100)
            t = np.array([1,60])
            sol = solve_ivp(fun=self.rate_ode, t_span=(0, t[-1]), y0=u0, method="BDF", t_eval=t, atol=1e-8, rtol=1e-8)
            y_list.append(sol.y.transpose(1, 0))
            t_list.append(t)
        return np.array(y_list), np.array(t_list)

class TOY(ChemistryScheme):
    def __init__(self):
        super().__init__(scheme_id="toy_44")

    def _data_solveivp(self, num_series, rand=False):
        y_list, t_list = [], []
        for i in range(num_series):
            # Initial conditions
            u0 = np.zeros(self.num_spc)
            for line in [line.strip(" \n;") for line in toy_44_init_conc_text.split("\n")]:
                if "=" in line:
                    spc_name, init_val = line.split("=")
                    u0[self.spc_names.index(spc_name.strip())] = float(init_val)
            if rand:
                u0 *= self.rng.uniform_(0.99, 1.01, size=u0.shape)  # rand by 0.1
            t = np.arange(101)
            sol = solve_ivp(fun=self.rate_ode, t_span=(0, t[-1]), y0=u0, method="BDF", t_eval=t, atol=1e-8, rtol=1e-8)
            y_list.append(sol.y.transpose(1, 0))
            t_list.append(t)
        return np.array(y_list), np.array(t_list)

    def _data_csv(self, filepath: str | Path):
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            header = (f.readline().strip("# \n")).split(',')
        permutation = [header.index(spc) for spc in self.spc_names]
        series = np.loadtxt(filepath, delimiter=',', skiprows=1)
        return series[:, permutation], series[:, 0]*3600

    def _data_KPP(self, num_series):  # rand unimplement
        """Load data simulated by KPP"""
        y_list, t_list = [], []
        for i in range(num_series):
            csv_file = f"data_t100_dt1_10/sed_{i+1}.csv"
            y, t = self._data_csv(csv_file)
            y_list.append(y)
            t_list.append(t)
        
        return np.array(y_list), np.array(t_list)
    
    def data(self, *args, rand=False, **kwargs):
        return self._data_KPP(*args, **kwargs)


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
    
def numpy_collate(batch):
    return jax.tree_util.tree_map(np.asarray, default_collate(batch))

def jax_collate(batch):
    return jax.tree_util.tree_map(jnp.array, default_collate(batch))

if __name__ == "__main__":
    # Regression test for scheme simulator =====================================
    # chem = TOY()
    # y_arr, t_arr = chem._data_solveivp(1)
    # true_toy_44_conc, _ = chem._data_csv("data_t100_dt1_10/toy_44.csv")
    # print("TOY Regression test: ", np.allclose(y_arr[0][-1], true_toy_44_conc[-1], rtol=1e-3))
    # y_arr, t_arr = chem.data(10)
    # print(y_arr.shape)

    # from plots.plot import plot_series
    # chem = ROBER()
    # y_arr, t_arr = chem.data(1)
    # fig = plot_series(y_arr[0], t=t_arr[0])
    # for ax in fig.axes:
    #     ax.set_xscale("log")
    # Path("plots").mkdir(parents=True, exist_ok=True)
    # fig.savefig("plots/pollu.png", dpi=300)

    chem = POLLU()
    y_arr, t_arr = chem.data(1)
    print(y_arr[0].shape)
    
    # from Verwer, 1994
    true_end_conc_text = """
    5.64625548e-02 1.34248413e-01 4.13973433e-09 5.52314021e-03
    2.01897726e-07 1.46454186e-07 7.78424912e-02 3.24507535e-01
    7.49401338e-03 1.62229316e-08 1.13586383e-08 2.23050598e-03
    2.08716288e-04 1.39692102e-05 8.96488486e-03 4.35284637e-18
    6.89921970e-03 1.00780304e-04 1.77214651e-06 5.68294329e-05
    """
    true_end_conc = np.fromstring(true_end_conc_text, sep=' ')
    print("POLLU Regression test: ", np.allclose(y_arr[0][-1], true_end_conc))

    # Regression test for jax rate law and solver ==============================
    import network as nt
    rate_law = nt.LogRateLaw(
        chem.stoi_reac, chem.stoi_prod,
        chem.RO2_IDX, chem.RO2_K_IDX,
        k_init=chem.rconst
    )
    solver = nt.Solverax(rate_law)
    y = solver(t_arr[0], y_arr[0][0])
    print("Jax Regression test: ", jnp.allclose(y[-1], jnp.asarray(true_end_conc)))