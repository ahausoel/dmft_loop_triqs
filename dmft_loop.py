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

### system parameters
beta = 10.0
mu = 0.0

U = 2.3
J = 0.5

N_atoms = 4
N_bands = 3

spin_names = ['up', 'dn']       # The spins
orb_names = [0, 1, 2]           # The orbitals

n_iw = int(100 * beta)
n_orb = len(orb_names)

### the hamiltonian
hkfile = file("/home/hpc/pr94vu/di73miv/work/RECHNUNGEN/benchmarks/dmft_loop/wannier90_hk_2layers_t2gbasis.dat_")
hk, kpoints = read_hamiltonian(hkfile, spin_orbit=True)

Nk = kpoints.shape[0]
N_lda_bands = hk.shape[1]
N_size_hk = N_lda_bands*2
hk = hk.reshape(Nk, N_size_hk, N_size_hk)

### the lattice greens function
iw_mesh = MeshImFreq(beta, 'Fermion', n_iw)
lda_orb_names = [ i for i in range(0,N_size_hk) ]
gf_struct = [("bl",lda_orb_names)]
print 'gf_struct', gf_struct
G0_iw_full = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)

#for name, block in G0_iw_full:
    #print 'name, block', name, block

iw_vec = array([iw.value * np.eye(N_size_hk) for iw in iw_mesh])

def get_local_lattice_gf(mu, iw_vec, hk, sigma):

    mu_mat = mu * np.eye(N_size_hk)

    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    size = world.Get_size()
    print 'rank', rank

    nk_per_core = int(Nk)/int(size)
    rest = int(Nk)%int(size)

    print 'size', size
    print 'nk_per_core ', nk_per_core 
    print 'rest', rest

    if rank < rest:
        my_ks = range(rank*nk_per_core+rank, (rank+1)*nk_per_core + rank +1)
    else:
        my_ks = range(rank*nk_per_core+rest, (rank+1)*nk_per_core + rest )

    my_G0 = np.zeros_like(iw_vec)

    print 'rank, my_ks', rank, my_ks

    #for asdf, k in enumerate(hk):
    for k in my_ks:

        print 'rank, k', rank, k

        tmp = linalg.inv( iw_vec + mu_mat - hk[k,:,:] - sigma)

        my_G0 += tmp

    # sum of the quantity
    G0_iw_full_mat = MPI.COMM_WORLD.allreduce(my_G0, op=MPI.SUM)

    G0_iw_full["bl"].data[...] = G0_iw_full_mat / float(Nk)

    return G0_iw_full

G0_iw_full = get_local_lattice_gf(mu, iw_vec, hk, np.zeros_like(iw_vec))

#### writeout the G0
#iw_array = []

#for i in iw_mesh:
    ##print 'i', i.value
    #iw_array.append(i.value)

##print 'iw_array', iw_array
##print 'type(iw_array)', type(iw_array)

#iw_array = np.asarray(iw_array)
#print 'iw_array.shape', iw_array.shape

#data = np.column_stack((np.imag(iw_array), np.real(G0_iw_full["bl"].data[:,0,0]), np.imag(G0_iw_full["bl"].data[:,0,0])))
#np.savetxt("g0_00.dat", data)
#data = np.column_stack((np.imag(iw_array), np.real(G0_iw_full["bl"].data[:,1,0]), np.imag(G0_iw_full["bl"].data[:,1,0])))
#np.savetxt("g0_10.dat", data)
#data = np.column_stack((np.imag(iw_array), np.real(G0_iw_full["bl"].data[:,2,0]), np.imag(G0_iw_full["bl"].data[:,2,0])))
#np.savetxt("g0_20.dat", data)


hk_mean = hk.mean(axis=0)

### first assume that all atoms have same size
idx_lst = list(range(len(spin_names) * len(orb_names)))
gf_struct = [('bl', idx_lst)]

G0_list = []
muimp_list = []

offset = 0
for i in range(0,N_atoms):

    G = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)

    #print 'G["bl"].data.shape', G["bl"].data.shape
    #print 'G0_iw_full["bl"].data[:, 0:offset, 0:offset] ',  G0_iw_full["bl"].data[:, 0:offset, 0:offset].shape

    size_block = len(spin_names)*len(orb_names)

    G["bl"].data[...] = G0_iw_full["bl"].data[:, offset:offset+size_block, offset:offset+size_block]


    G0_list.append(G)

    muimp = hk_mean[offset:offset+size_block, offset:offset+size_block]

    muimp_list.append(muimp)
    #print 'muimp'
    #print  muimp

    offset = offset + size_block


#print 'G0_list', G0_list
 
#for G0 in G0_list:
    #print 'G0', G0

#exit()

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
            'n_warmup_cycles' : 1000,
            'n_cycles' : 1000000000,
            'max_time' : max_time,
            'length_cycle' : 100,
            'move_double' : True,
            'measure_pert_order' : True
            }

    #start = time.time()
    S.solve(**solve_params)
    #end = time.time()

    return S.G_iw

for G0_iw in G0_list:

    print 'G0_iw', G0_iw

    # ==== Local Hamiltonian ====
    c_dag_vec = matrix([[c_dag('bl', idx) for idx in idx_lst]])
    c_vec =     matrix([[c('bl', idx)] for idx in idx_lst])

    h_0_mat = muimp_list[0]
    h_0 = (c_dag_vec * h_0_mat * c_vec)[0,0]

    #print 'h_0', h_0

    h_loc = h_0
    #h_int = 0.000000000001 * c_dag("bl", 0) * c("bl", 0) * c_dag("bl", 1) * c("bl", 1)
    Umat, Upmat = U_matrix_kanamori(len(orb_names), U_int=U, J_hund=J)
    op_map = { (s,o): ('bl',i) for i, (s,o) in enumerate(product(spin_names, orb_names)) }
    h_int = h_int_kanamori(spin_names, orb_names, Umat, Upmat, J, off_diag=True, map_operator_structure=op_map)
    #h_loc = h_0 + h_int

    #sys.path.append("/home/hpc/pr94vu/di73miv/work/RECHNUNGEN/benchmarks/dmft_loop")
    #from triqs_cthyb_solve import solve_aim

    max_time = 10
    G_iw = solve_aim(beta, gf_struct, n_iw, h_int, max_time, G0_iw)

    ## --------- Construct the CTHYB solver ----------
    #constr_params = {
            #'beta' : beta,
            #'gf_struct' : gf_struct,
            #'n_iw' : n_iw,
            #'n_tau' : 100000
            #}
    #S = Solver(**constr_params)

    ## --------- Initialize G0_iw ----------
    #S.G0_iw << G0_iw

    ## --------- Solve! ----------
    #solve_params = {
            #'h_int' : h_int,
            #'n_warmup_cycles' : 1000,
            #'n_cycles' : 1000000000,
            #'max_time' : max_time,
            #'length_cycle' : 100,
            #'move_double' : True,
            #'measure_pert_order' : True
            #}
    ##start = time.time()
    #S.solve(**solve_params)
    ##end = time.time()
