#!/lrz/sys/tools/python/intelpython27_u4/intelpython2/bin/python

import sys, os

from pytriqs.gf import Gf, MeshImFreq, MeshImTime, iOmega_n, inverse, MeshBrillouinZone, MeshProduct, BlockGf, LegendreToMatsubara
from pytriqs.lattice import BravaisLattice, BrillouinZone
from pytriqs.operators import c, c_dag, n
from pytriqs.operators.util import h_int_kanamori, U_matrix_kanamori
from itertools import product
from numpy import matrix, array, diag, pi
import numpy.linalg as linalg

from pytriqs.utility import mpi

from pytriqs.statistics.histograms import Histogram
from pytriqs.archive import HDFArchive

#import pytriqs.utility 
#from mpi4py import MPI

sys.path.append("/home/hpc/pr94vu/di73miv/work/w2dynamics___patrik_alexander_merge")
from auxiliaries.input import read_hamiltonian

import numpy as np

import psutil

from parameters import *
check_sanity_of_parameters()

#######################################################################
### switch solver

### i for now allow this to be overwritten by an argument to the dmft_loop.py call
import argparse
parser = argparse.ArgumentParser(description="asdf")
parser.add_argument('-w', '--w2dyn', default=False, action='store_true', help="Use w2dyn as solver.")
parser.add_argument('-t', '--triqs', default=False, action='store_true', help="Use triqs as solver.")
args = parser.parse_args()

if args.w2dyn:
    solver = "w2dyn"
elif args.triqs:
    solver = "triqs"

if args.w2dyn and args.triqs:
    print 'cannot use both solvers!'
    exit()

if solver == 'triqs':
    from triqs_cthyb import Solver, version
elif solver == 'w2dyn':
    from w2dyn_cthyb import Solver

### the readin-function from w2dyn makes spin as fastest running index, 
### but for triqs we need orbial to be fastest
#hk = hk.transpose(0,2,1,4,3)

iw_mesh = MeshImFreq(beta, 'Fermion', n_iw)

### the hamiltonian
hkfile = file(hkfilename)
hk, kpoints = read_hamiltonian(hkfile, spin_orbit=True)

### the lattice properties
Nk = kpoints.shape[0]
tmp = hk.shape[1]
N_size_hk = tmp*2
print 'hk.shape', hk.shape
print 'Nk', Nk
print 'N_size_hk', N_size_hk
hk = hk.reshape(Nk, N_size_hk, N_size_hk)

lda_orb_names = [ i for i in range(0,N_size_hk) ]
gf_struct_full = [("bl",lda_orb_names)]
print 'gf_struct_full', gf_struct_full
iw_vec_full = array([iw.value * np.eye(N_size_hk) for iw in iw_mesh])

### the impurity properties
idx_lst = list(range(len(spin_names) * len(orb_names)))
gf_struct = [('bl', idx_lst)]

world = mpi.world
rank = world.Get_rank()
size = world.Get_size()

def get_local_lattice_gf(mu_, hk_, sigma_):

    mu_mat = mu_ * np.eye(N_size_hk)

    G_lattice_iw_full = BlockGf(mesh=iw_mesh, gf_struct=gf_struct_full)

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

    ### sum of the quantity
    ### this still crashes with more than one node..
    #qtty_rank = np.asarray(my_G0)
    #G0_iw_full_mat = np.zeros_like(my_G0)
    #MPI.COMM_WORLD.Allreduce(qtty_rank, G0_iw_full_mat)

    #G_lattice_iw_full["bl"].data[...] = G0_iw_full_mat / float(Nk)

    ### alternative version
    qtty_rank = np.asarray(my_G0)
    qtty_mean_root = np.zeros_like(iw_vec_full)
    world.Reduce(qtty_rank, qtty_mean_root, root=0)
    qtty_mean = world.bcast(qtty_mean_root, root=0)

    G_lattice_iw_full["bl"].data[...] = qtty_mean / float(Nk)

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
    if solver == "triqs":
        constr_params = {
                'beta' : beta,
                'gf_struct' : gf_struct,
                'n_iw' : n_iw,
                'n_tau' : 10000,   # triqs value
                'n_l' : 50,
                #'complex': True   # only necessary for w2dyn
                }
    elif solver == "w2dyn":
        constr_params = {
                'beta' : beta,
                'gf_struct' : gf_struct,
                'n_iw' : n_iw,
                'n_tau' : 9999,   # w2dyn value
                'n_l' : 50,
                'complex': True   # only necessary for w2dyn
                }
    S = Solver(**constr_params)

    # --------- Initialize G0_iw ----------
    S.G0_iw << G0_iw_

    # --------- Solve! ----------
    solve_params = {
            'h_int' : h_int_,
            'n_warmup_cycles' : 10000,
            #'n_cycles' : 1000000000,
            'n_cycles' : 10000,
            'max_time' : max_time_,
            'length_cycle' : 100,
            'move_double' : True,
            'measure_pert_order' : True,
            'measure_G_l' : True
            }

    #start = time.time()
    print 'running solver...'

    process = psutil.Process(os.getpid())
    print "memory info before: ", process.memory_info().rss/1024/1024, " MB"

    S.solve(**solve_params)
    
    process = psutil.Process(os.getpid())
    print "memory info after: ", process.memory_info().rss/1024/1024, " MB"

    print 'exited solver rank ', rank
    #end = time.time()

    G_iw_from_legendre = G0_iw_.copy()
    G_iw_from_legendre << LegendreToMatsubara(S.G_l)
    print 'G_iw_from_legendre', G_iw_from_legendre
    ##exit()

    ### giw from legendre, calculated within the interface
    #print 'S.G_iw_from_leg', S.G_iw_from_leg
    #exit()

    n_tau = 200
    tau_mesh2 = MeshImTime(beta, 'Fermion', n_tau)
    my_G_tau = BlockGf(mesh=tau_mesh2, gf_struct=gf_struct)
    print 'S.G_tau["bl"][:,:].data ',  S.G_tau["bl"][:,:].data.shape

    my_G_tau["bl"][:,:].data[...] = S.G_tau["bl"][:,:].data.reshape(200, 50, N_bands*2, N_bands*2).mean(axis = 1)

    #return my_G_tau, S.G_iw_from_leg
    #return my_G_tau, S.G_iw
    if solver == 'triqs':
        return my_G_tau, G_iw_from_legendre, S.average_sign
    else:
        return my_G_tau, S.G_iw_from_leg, S.average_sign

def solve_aims(G0_iw_list_):

    G_iw_list = []
    G_tau_list = []
    average_sign_list = []

    for G0_iw in G0_iw_list_:

        print 'G0_iw', G0_iw

        # ==== Local Hamiltonian ====
        c_dag_vec = matrix([[c_dag('bl', idx) for idx in idx_lst]])
        c_vec =     matrix([[c('bl', idx)] for idx in idx_lst])

        h_0_mat = t_ij_list[0]
        h_0 = (c_dag_vec * h_0_mat * c_vec)[0,0]

        # ==== Interacting Hamiltonian ====
        Umat, Upmat = U_matrix_kanamori(len(orb_names), U_int=U, J_hund=J)
        #op_map = { (s,o): ('bl',i) for i, (s,o) in enumerate(product(spin_names, orb_names)) }
        op_map = { (s,o): ('bl',i) for i, (o,s) in enumerate(product(orb_names, spin_names)) }
        h_int = h_int_kanamori(spin_names, orb_names, Umat, Upmat, J, off_diag=True, map_operator_structure=op_map)

        G_tau, G_iw, average_sign = ctqmc_solver(h_int, max_time, G0_iw)

        G_iw_list.append(G_iw)
        G_tau_list.append(G_tau)
        average_sign_list.append(average_sign)

    return G_tau_list, G_iw_list, average_sign_list


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

    if world.Get_rank() == 0:
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

    if world.Get_rank() == 0:
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

        import inspect
        import __main__
        source = inspect.getsource(__main__)
        results["source_file"] = source

        return results


def write_qtty(qtty_, qtty_name_, res_file_):

    if world.Get_rank() == 0:
        for ni,i in enumerate(qtty_):

            dataname = qtty_name_ + "___at_" + str(ni)
            res_file_[dataname] = i

def get_zero_sigma_iw_list():

    Sigma_iw_list = []

    for i in range(0,N_atoms):

        G = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)
        Sigma_iw_list.append(G)

    return Sigma_iw_list

def readold_sigma_iw_list(oldfile):

    if rank == 0:

        print 'oldfile', oldfile
        results =  HDFArchive(oldfile,'r')

        Sigma_iw_list = []

        n_iw_new = results["Sigma_iw___at_0/bl/mesh/size"]
        iw_mesh_new = MeshImFreq(beta, 'Fermion', n_iw_new/2) 
        ### n_iw for MeshImFreq is positive number of frequencies,
        ### when read out from hdf-file it is total number of freqs.

        for i in range(0,N_atoms):

            dataname = "Sigma_iw___at_" + str(i)
            tmp = results[dataname]

            S = BlockGf(mesh=iw_mesh_new, gf_struct=gf_struct)
            S["bl"].data[...] = tmp["bl"].data[...]

            Sigma_iw_list.append(S)

    else: 
        Sigma_iw_list = None

    Sigma_iw_list = world.bcast(Sigma_iw_list, root = 0)

    return Sigma_iw_list


check_output_folder()

### start calculation from scratch
Sigma_iw_full = BlockGf(mesh=iw_mesh, gf_struct=gf_struct_full)
Sigma_iw_list = get_zero_sigma_iw_list()

### start from old calculation
#Sigma_iw_list = readold_sigma_iw_list("data_from_scratch/iteration_027.h5")
#Sigma_iw_full = upfold_Sigma(Sigma_iw_list)


for iter in range(0,N_iter):

    results = initialize_outputfile(iter)

    G_lattice_iw_full = get_local_lattice_gf(mu, hk, Sigma_iw_full["bl"].data)

    G_lattice_iw_list, t_ij_list = downfold_G_lattice(G_lattice_iw_full)


    write_qtty(G_lattice_iw_list, "G_lattice_iw", results)

    G0_iw_list = compute_new_weiss_field(G_lattice_iw_list, Sigma_iw_list)

    write_qtty(G0_iw_list, "G0_iw", results)

    G_tau_list, G_iw_list, average_sign_list = solve_aims(G0_iw_list)

    write_qtty(G_tau_list, "G_tau", results)
    write_qtty(G_iw_list, "G_iw", results)
    write_qtty(average_sign_list, "average_sign", results)

    Sigma_iw_list = calculate_sigmas(G_iw_list, G0_iw_list)

    write_qtty(Sigma_iw_list, "Sigma_iw", results)

    Sigma_iw_full = upfold_Sigma(Sigma_iw_list)
