#!/lrz/sys/tools/python/intelpython27_u4/intelpython2/bin/python

import sys, os
sys.path.append("/home/hpc/pr94vu/di73miv/work/RECHNUNGEN/benchmarks/common/")
from util import *

from pytriqs.gf import Gf, MeshImFreq, iOmega_n, inverse, MeshBrillouinZone, MeshProduct
from pytriqs.lattice import BravaisLattice, BrillouinZone
from pytriqs.operators import c, c_dag, n
from pytriqs.operators.util import h_int_kanamori, U_matrix_kanamori
from itertools import product
from numpy import matrix, array, diag, pi
import numpy.linalg as linalg

#from pytriqs.archive import HDFArchive
#from pytriqs.utility import mpi
from triqs_cthyb import Solver, version
#from w2dyn_cthyb import Solver

from pytriqs.statistics.histograms import Histogram
from pytriqs.archive import HDFArchive

#import pytriqs.utility 
from mpi4py import MPI

#from tight_binding_model import *

#sys.path.append("/home/hpc/pr94vu/di73miv/work/w2dynamics___patrik_alexander_merge")
sys.path.append("/gpfs/work/pr94vu/di73miv/w2dynamics_github___neu___cmake___REPRODAGAIN___KLON")
from auxiliaries.input import read_hamiltonian

import numpy as np

#######################################################################
### here come the global parameters

max_time = 30
#max_time = 125
#max_time = 250
#max_time = 3600

beta = 10.0
mu = 0.0

U = 2.0
J = 0.0
#V = 1.0

N_atoms = 1
N_bands = 1

#data_folder = "data_1orb___U2___fast"
data_folder = "data_1orb___U2"
#data_folder = "data_2orbs_hz"
#data_folder = "data_2orbs_hz_U0_V0_J0"
#data_folder = "data_2orbs_hz_V0_J0___mu0"
#data_folder = "data_2orbs_highstat"

### still assumed that all atoms have same size and no noninteracting orbitals
spin_names = ['up', 'dn']
orb_names = [0]

n_iw = int(100 * beta)
iw_mesh = MeshImFreq(beta, 'Fermion', n_iw)

### the hamiltonian
hkfile = file("/home/hpc/pr94vu/di73miv/work/RECHNUNGEN/dmft_loop_triqs___1orb_hk___splitting/Hk_modified.dat")
hk, kpoints = read_hamiltonian(hkfile, spin_orbit=True)
print 'hk.shape', hk.shape

### the lattice properties
Nk = kpoints.shape[0]
tmp = hk.shape[1]
N_size_hk = tmp*2
print 'Nk', Nk
print 'n_size_hk', N_size_hk
hk = hk.reshape(Nk, N_size_hk, N_size_hk)

lda_orb_names = [ i for i in range(0,N_size_hk) ]
gf_struct_full = [("bl",lda_orb_names)]
print 'gf_struct_full', gf_struct_full
G_lattice_iw_full = BlockGf(mesh=iw_mesh, gf_struct=gf_struct_full)
iw_vec_full = array([iw.value * np.eye(N_size_hk) for iw in iw_mesh])

### the impurity properties
idx_lst = list(range(len(spin_names) * len(orb_names)))
gf_struct = [('bl', idx_lst)]

def get_local_lattice_gf(mu_, hk_, sigma_):

    mu_mat = mu_ * np.eye(N_size_hk)

    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    size = world.Get_size()

    nk_per_core = int(Nk)/int(size)
    rest = int(Nk)%int(size)

    #print 'rank', rank
    #print 'size', size
    #print 'nk_per_core ', nk_per_core 
    #print 'rest', rest

    if rank < rest:
        my_ks = range(rank*nk_per_core+rank, (rank+1)*nk_per_core + rank +1)
    else:
        my_ks = range(rank*nk_per_core+rest, (rank+1)*nk_per_core + rest )

    my_G0 = np.zeros_like(iw_vec_full)

    #print 'rank, my_ks', rank, my_ks

    for k in my_ks:

        tmp = linalg.inv( iw_vec_full + mu_mat - hk_[k,:,:] - sigma_)

        my_G0 += tmp

    # sum of the quantity
    G0_iw_full_mat = MPI.COMM_WORLD.allreduce(my_G0, op=MPI.SUM)

    G_lattice_iw_full["bl"].data[...] = G0_iw_full_mat / float(Nk)

    return G_lattice_iw_full


def downfold_G_lattice(G0_iw_full_):

    G_lattice_iw_list = []
    t_ij_list = []

    offset = 0
    for i in range(0,N_atoms):

        G = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)

        #print 'G["bl"].data.shape', G["bl"].data.shape
        #print 'G0_iw_full_["bl"].data[:, 0:offset, 0:offset] ',  G0_iw_full_["bl"].data[:, 0:offset, 0:offset].shape

        size_block = len(spin_names)*len(orb_names)

        G["bl"].data[...] = G0_iw_full_["bl"].data[:, offset:offset+size_block, offset:offset+size_block]


        G_lattice_iw_list.append(G)

        hk_mean = hk.mean(axis=0)
        t_ij = hk_mean[offset:offset+size_block, offset:offset+size_block]

        t_ij_list.append(t_ij)

        offset = offset + size_block

    return G_lattice_iw_list, t_ij_list

def compute_new_weiss_field(G_lattice_iw_list_, Sigma_iw_list_):

    G0_iw_list = []

    for g, s in zip(G_lattice_iw_list_, Sigma_iw_list_):

        G0_iw = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)

        G0_iw << inverse(inverse(g) + s)

        G0_iw_list.append(G0_iw)

    return G0_iw_list


def ctqmc_solver(h_int_, max_time_, G0_iw_):

    # --------- Construct the CTHYB solver ----------
    constr_params = {
            'beta' : beta,
            'gf_struct' : gf_struct,
            'n_iw' : n_iw,
            'n_tau' : 100000,
            #'complex': True
            }
    S = Solver(**constr_params)

    # --------- Initialize G0_iw ----------
    S.G0_iw << G0_iw_

    # --------- Solve! ----------
    solve_params = {
            'h_int' : h_int_,
            'n_warmup_cycles' : 100,
            'n_cycles' : 1000000000,
            #'n_cycles' : 100,
            'max_time' : max_time_,
            'length_cycle' : 100,
            'move_double' : True,
            'measure_pert_order' : True
            }

    #start = time.time()
    S.solve(**solve_params)
    #end = time.time()

    return S.G_iw

def solve_aims(G0_iw_list_):

    G_iw_list = []

    for G0_iw in G0_iw_list_:

        print 'G0_iw', G0_iw

        # ==== Local Hamiltonian ====
        c_dag_vec = matrix([[c_dag('bl', idx) for idx in idx_lst]])
        c_vec =     matrix([[c('bl', idx)] for idx in idx_lst])

        h_0_mat = t_ij_list[0]
        h_0 = (c_dag_vec * h_0_mat * c_vec)[0,0]

        # ==== Interacting Hamiltonian ====
        Umat, Upmat = U_matrix_kanamori(len(orb_names), U_int=U, J_hund=J)
        op_map = { (s,o): ('bl',i) for i, (s,o) in enumerate(product(spin_names, orb_names)) }
        h_int = h_int_kanamori(spin_names, orb_names, Umat, Upmat, J, off_diag=True, map_operator_structure=op_map)

        G_iw = ctqmc_solver(h_int, max_time, G0_iw)

        G_iw_list.append(G_iw)

    return G_iw_list


### now i calculate sigma

def calculate_sigmas(G_iw_list_, G0_iw_list_):

    Sigma_iw_list = []

    for G_iw, G0_iw in zip(G_iw_list_, G0_iw_list_):

        print ' '
        print 'G_iw', G_iw
        print 'G0_iw', G0_iw

        Sigma = G0_iw.copy()

        Sigma << inverse(G0_iw) - inverse(G_iw)

        Sigma_iw_list.append(Sigma)

    return Sigma_iw_list


### upfold sigma

def upfold_Sigma(Sigma_iw_list_):

    Sigma_iw_full_ = BlockGf(mesh=iw_mesh, gf_struct=gf_struct_full)

    offset = 0
    for Sigma_iw in Sigma_iw_list_:

        size_block = len(spin_names)*len(orb_names)

        Sigma_iw_full_["bl"].data[:, offset:offset+size_block, offset:offset+size_block] = Sigma_iw["bl"].data[...]

        offset = offset + size_block

    return Sigma_iw_full_


def check_output_folder():

    if MPI.COMM_WORLD.Get_rank() == 0:
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            print 'make folder!'
        else:
            import shutil
            shutil.rmtree(data_folder, ignore_errors=True)
            os.makedirs(data_folder)
            #os.remove(data_folder+"/*") 
            print 'make folder new!'

def initialize_outputfile(iter_):

    if MPI.COMM_WORLD.Get_rank() == 0:
        if iter_<10:
            filename = data_folder + "/iteration_00" + str(iter_) + ".h5"
        elif iter_<100:
            filename = data_folder + "/iteration_0" + str(iter_) + ".h5"
        elif iter_<1000:
            filename = data_folder + "/iteration_" + str(iter_) + ".h5"
        else:
            print 'too many iterations...'
            exit()

        print 'filename', filename
        results =  HDFArchive(filename,'w')

        return results


def write_qtty(qtty_, qtty_name_, res_file_):

    if MPI.COMM_WORLD.Get_rank() == 0:
        for ni,i in enumerate(qtty_):

            dataname = qtty_name_ + "___at_" + str(ni)
            res_file_[dataname] = i

check_output_folder()

results = initialize_outputfile(0)

G_lattice_iw_full = get_local_lattice_gf(mu, hk, np.zeros_like(iw_vec_full))

G_lattice_iw_list, t_ij_list = downfold_G_lattice(G_lattice_iw_full)


write_qtty(G_lattice_iw_list, "G_lattice_iw", results)

G0_iw_list = G_lattice_iw_list

write_qtty(G0_iw_list, "G0_iw", results)

G_iw_list = solve_aims(G0_iw_list)

write_qtty(G_iw_list, "G_iw", results)

Sigma_iw_list = calculate_sigmas(G_iw_list, G0_iw_list)

write_qtty(Sigma_iw_list, "Sigma_iw", results)

Sigma_iw_full = upfold_Sigma(Sigma_iw_list)

results = initialize_outputfile(1)

##print 'iw_vec_full.shape', iw_vec_full.shape
##print 'Sigma_iw_full["bl"].data ',  Sigma_iw_full["bl"].data

G_lattice_iw_full = get_local_lattice_gf(mu, hk, Sigma_iw_full["bl"].data)

G_lattice_iw_list, t_ij_list = downfold_G_lattice(G_lattice_iw_full)

G0_iw_list = compute_new_weiss_field(G_lattice_iw_list, Sigma_iw_list)

write_qtty(G0_iw_list, "G0_iw", results)

G_iw_list = solve_aims(G0_iw_list)

write_qtty(G_iw_list, "G_iw", results)

Sigma_iw_list = calculate_sigmas(G_iw_list, G0_iw_list)

write_qtty(Sigma_iw_list, "Sigma_iw", results)

Sigma_iw_full = upfold_Sigma(Sigma_iw_list)
