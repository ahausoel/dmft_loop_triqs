import os
from pytriqs.gf import Gf
from pytriqs.archive import HDFArchive
from pytriqs.utility.comparison_tests import assert_block_gfs_are_close


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


def check_quantity(file_ref, quantity_name):

    file_new = "data_from_scratch/iteration_000.h5"

    results_ref =  HDFArchive(file_ref,'r')
    results_new =  HDFArchive(file_new,'r')

    quantity_ref = results_ref[quantity_name]
    quantity_new = results_new[quantity_name]

    print 'checking quantity ', quantity_name, '...'
    assert_block_gfs_are_close(quantity_new, quantity_ref, precision = 1e-10), \


print 'running triqs...'
os.system("../dmft_loop.py --triqs &> /dev/null")
#os.system("../dmft_loop.py --triqs")

for i in quantities:
    check_quantity("data_from_scratch___ref_triqs/iteration_000.h5", i)

print 'running w2dyn...'
os.system("../dmft_loop.py --w2dyn &> /dev/null")
#os.system("../dmft_loop.py --w2dyn")

for i in quantities:
    check_quantity("data_from_scratch___ref_w2dyn/iteration_000.h5", i)

#print '=== all tests successful! >>>'
