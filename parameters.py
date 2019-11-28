#!/lrz/sys/tools/python/intelpython27_u4/intelpython2/bin/python

#solver = 'triqs'
solver = 'w2dyn'

N_iter = 1

max_time = 0

beta = 27.0
mu = 7.37546

U = 2.234
J = 0.203

N_atoms = 2
N_bands = 3

data_folder = "data_from_scratch"

### still assumed that all atoms have same size and no noninteracting orbitals
spin_names = ['up', 'dn']
orb_names = [0, 1, 2]

n_iw = 1000

### the hamiltonian
hkfilename = "wannier90_hk_t2gbasis.dat_"

def check_sanity_of_parameters():

    if not isinstance(N_iter, int):
        print 'parameter N_iter must be integer!'
        exit()
    if not isinstance(max_time, int):
        print 'parameter max_time must be integer!'
        exit()
    if not isinstance(N_atoms, int):
        print 'parameter N_atoms must be integer!'
        exit()
    if not isinstance(N_bands, int):
        print 'parameter N_bands must be integer!'
        exit()

    if not isinstance(beta, float):
        print 'parameter beta must be integer!'
        exit()
    if not isinstance(mu, float):
        print 'parameter mu must be integer!'
        exit()
    if not isinstance(U, float):
        print 'parameter U must be integer!'
        exit()
    if not isinstance(J, float):
        print 'parameter J must be integer!'
        exit()

    if not isinstance(solver, basestring):
        print 'parameter solver must be string!'
        exit()
    if not isinstance(data_folder, basestring):
        print 'parameter data_folder must be string!'
        exit()
    if not isinstance(hkfilename, basestring):
        print 'parameter hkfilename must be string!'
        exit()

    if not len(orb_names) == N_bands:
        print 'len(orb_names) must be N_bands!'
        exit()
    if not len(spin_names) == 2:
        print 'system must have two spins!'
        exit()
