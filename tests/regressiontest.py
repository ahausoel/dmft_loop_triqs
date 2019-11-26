import os
from pytriqs.gf import Gf
from pytriqs.archive import HDFArchive
from pytriqs.utility.comparison_tests import assert_block_gfs_are_close

os.system("../dmft_loop.py 1")

#from pytriqs.utility.h5diff import h5diff
#h5diff("data_from_scratch___ref/iteration_000.h5", "data_from_scratch/iteration_000.h5")

quantities = [
"G0_iw___at_0", \
"G0_iw___at_1", \
"G_iw___at_0", \
"G_iw___at_1", \
"G_lattice_iw___at_0", \
"G_lattice_iw___at_1", \
"G_tau___at_0", \
"G_tau___at_1", \
"Sigma_iw___at_0", \
"Sigma_iw___at_1" ]


def check_quantity(quantity_name):

    file_ref = "data_from_scratch___ref/iteration_000.h5"
    file_new = "data_from_scratch/iteration_000.h5"

    results_ref =  HDFArchive(file_ref,'r')
    results_new =  HDFArchive(file_new,'r')

    quantity_ref = results_ref[quantity_name]
    quantity_new = results_new[quantity_name]

    assert_block_gfs_are_close(quantity_new, quantity_ref, precision = 1e-10), \


for i in quantities:
    check_quantity(i)
