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

from pytriqs.statistics.histograms import Histogram

#import pytriqs.utility 
from mpi4py import MPI

#from tight_binding_model import *

sys.path.append("/home/hpc/pr94vu/di73miv/work/w2dynamics___patrik_alexander_merge")
from auxiliaries.input import read_hamiltonian

import numpy as np

#######################################################################
### here come the global parameters
beta = 10.0
mu = 0.0

U = 2.3
J = 0.5

N_atoms = 4
N_bands = 3

### still assumed that all atoms have same size and no noninteracting orbitals
spin_names = ['up', 'dn']
orb_names = [0, 1, 2]

n_iw = int(100 * beta)
n_orb = len(orb_names)

### the hamiltonian
hkfile = file("/home/hpc/pr94vu/di73miv/work/RECHNUNGEN/benchmarks/dmft_loop/wannier90_hk_2layers_t2gbasis.dat_")
hk, kpoints = read_hamiltonian(hkfile, spin_orbit=True)

Nk = kpoints.shape[0]
tmp = hk.shape[1]
N_size_hk = tmp*2
hk = hk.reshape(Nk, N_size_hk, N_size_hk)

### the lattice greens function
iw_mesh = MeshImFreq(beta, 'Fermion', n_iw)
lda_orb_names = [ i for i in range(0,N_size_hk) ]
gf_struct = [("bl",lda_orb_names)]
print 'gf_struct', gf_struct
G0_iw_full = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)

iw_vec_full = array([iw.value * np.eye(N_size_hk) for iw in iw_mesh])

def get_local_lattice_gf(mu_, iw_vec_full_, hk_, sigma_):

    mu_mat = mu_ * np.eye(N_size_hk)

    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    size = world.Get_size()
    #print 'rank', rank

    nk_per_core = int(Nk)/int(size)
    rest = int(Nk)%int(size)

    #print 'size', size
    #print 'nk_per_core ', nk_per_core 
    #print 'rest', rest

    if rank < rest:
        my_ks = range(rank*nk_per_core+rank, (rank+1)*nk_per_core + rank +1)
    else:
        my_ks = range(rank*nk_per_core+rest, (rank+1)*nk_per_core + rest )

    my_G0 = np.zeros_like(iw_vec_full_)

    #print 'rank, my_ks', rank, my_ks

    for k in my_ks:

        tmp = linalg.inv( iw_vec_full_ + mu_mat - hk_[k,:,:] - sigma_)

        my_G0 += tmp

    # sum of the quantity
    G0_iw_full_mat = MPI.COMM_WORLD.allreduce(my_G0, op=MPI.SUM)

    G0_iw_full["bl"].data[...] = G0_iw_full_mat / float(Nk)

    return G0_iw_full

G0_iw_full = get_local_lattice_gf(mu, iw_vec_full, hk, np.zeros_like(iw_vec_full))

hk_mean = hk.mean(axis=0)

### first assume that all atoms have same size
idx_lst = list(range(len(spin_names) * len(orb_names)))
gf_struct = [('bl', idx_lst)]

def downfold_G0(G0_iw_full_):

    G0_iw_list = []
    t_ij_list = []

    offset = 0
    for i in range(0,N_atoms):

        G = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)

        #print 'G["bl"].data.shape', G["bl"].data.shape
        #print 'G0_iw_full_["bl"].data[:, 0:offset, 0:offset] ',  G0_iw_full_["bl"].data[:, 0:offset, 0:offset].shape

        size_block = len(spin_names)*len(orb_names)

        G["bl"].data[...] = G0_iw_full_["bl"].data[:, offset:offset+size_block, offset:offset+size_block]


        G0_iw_list.append(G)

        t_ij = hk_mean[offset:offset+size_block, offset:offset+size_block]

        t_ij_list.append(t_ij)
        #print 't_ij'
        #print  t_ij

        offset = offset + size_block

    return G0_iw_list, t_ij_list

G0_iw_list, t_ij_list = downfold_G0(G0_iw_full)

def solve_aim(beta, gf_struct, n_iw, h_int, max_time, G0_iw):

    # --------- Construct the CTHYB solver ----------
    constr_params = {
            'beta' : beta,
            'gf_struct' : gf_struct,
            'n_iw' : n_iw,
            'n_tau' : 100000
            }
    S = Solver(**constr_params)

    # --------- Initialize G0_iw ----------
    S.G0_iw << G0_iw

    # --------- Solve! ----------
    solve_params = {
            'h_int' : h_int,
            'n_warmup_cycles' : 100,
            #'n_cycles' : 1000000000,
            'n_cycles' : 10,
            'max_time' : max_time,
            'length_cycle' : 100,
            'move_double' : True,
            'measure_pert_order' : True
            }

    #start = time.time()
    S.solve(**solve_params)
    #end = time.time()

    return S.G_iw

G_iw_list = []

for G0_iw in G0_iw_list:

    print 'G0_iw', G0_iw

    # ==== Local Hamiltonian ====
    c_dag_vec = matrix([[c_dag('bl', idx) for idx in idx_lst]])
    c_vec =     matrix([[c('bl', idx)] for idx in idx_lst])

    h_0_mat = t_ij_list[0]
    h_0 = (c_dag_vec * h_0_mat * c_vec)[0,0]

    #print 'h_0', h_0

    Umat, Upmat = U_matrix_kanamori(len(orb_names), U_int=U, J_hund=J)
    op_map = { (s,o): ('bl',i) for i, (s,o) in enumerate(product(spin_names, orb_names)) }
    h_int = h_int_kanamori(spin_names, orb_names, Umat, Upmat, J, off_diag=True, map_operator_structure=op_map)

    max_time = -1
    G_iw = solve_aim(beta, gf_struct, n_iw, h_int, max_time, G0_iw)

    G_iw_list.append(G_iw)


### now i calculate sigma

Sigma_iw_list = []

for G_iw, G0_iw in zip(G_iw_list, G0_iw_list):

    print ' '
    print 'G_iw', G_iw
    print 'G0_iw', G0_iw

    Sigma = G0_iw.copy()

    Sigma << inverse(G0_iw) - inverse(G_iw)

    Sigma_iw_list.append(Sigma)
