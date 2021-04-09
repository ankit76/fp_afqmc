import numpy
import math, os, time, sys
import h5py
import scipy.sparse
from pyscf import gto, scf, mcscf, fci, ao2mo, lib, tools, cc
from pauxy.systems.generic import Generic
import scipy.linalg as la
from pauxy.utils.from_pyscf import generate_integrals
from pyscf.shciscf import shci
import prepVMC


# Cu2O2^2+: bis f = 0., per f = 1.
if len(sys.argv) == 2:
    f = float(sys.argv[1])
else:
    f = 0.

# bond lengths
cu_cu = 2.8 + f * 0.8
o_o = 2.3 - f * 0.9

atomString = f'Cu {-cu_cu/2} 0. 0.; Cu {cu_cu/2} 0. 0.; O 0. {o_o/2} 0.; O 0. {-o_o/2} 0.'

mol = gto.M(atom = atomString, basis = {'Cu': 'ano@6s5p3d2f1g', 'O': 'ano@4s3p2d1f'},
    verbose = 4, unit = 'angstrom', symmetry = 1, spin = 0, charge = 2)
mf = scf.RHF(mol).x2c()
#mf.chkfile = f'cu2o2_{f}_SHCISCF.chk'
#mf.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
#mf.init_guess = '1e'
mf.level_shift = 0.1
mf.scf()
print(f'mol.nao: {mol.nao}')
print(f'mol.elec: {mol.nelec}')

norbFrozen = 20
ncore = 20
norbAct = 100

mc1 = shci.SHCISCF(mf, norbAct, mol.nelectron - 2*ncore)
#chkfile = f'cu2o2_{f}_SHCISCF.chk'
#mc1.__dict__.update(lib.chkfile.load(chkfile, 'mcscf'))
mc1.chkfile = f'cu2o2_{f}_SHCISCF.chk'
mc1.frozen = norbFrozen
mc1.fcisolver.sweep_iter = [ 0 ]
mc1.fcisolver.sweep_epsilon = [ 1e-3 ]
mc1.fcisolver.nPTiter = 0
mc1.max_cycle_macro = 10
mc1.internal_rotation = True
mc1.fcisolver.nPTiter = 0  # Turns off PT calculation, i.e. no PTRDM.
mc1.fcisolver.mpiprefix = "mpirun -np 28"
mc1.fcisolver.scratchDirectory = "/rc_scratch/anma2640/cu2o2/1.0"
mc1.mc1step()


mc0 = mcscf.CASSCF(mf, mol.nao - ncore, mol.nelectron - 2*ncore)
mc0.mo_coeff = mc1.mo_coeff
nelecAct = mol.nelectron - 2*ncore
moCore = mc0.mo_coeff[:,:ncore]
core_dm = 2 * moCore.dot(moCore.T)
corevhf = mc0.get_veff(mol, core_dm)
energy_core = mol.energy_nuc()
energy_core += numpy.einsum('ij,ji', core_dm, mc0.get_hcore())
energy_core += numpy.einsum('ij,ji', core_dm, corevhf) * .5
moActDice = mc0.mo_coeff[:, ncore:]
h1eff = moActDice.T.dot(mc0.get_hcore() + corevhf).dot(moActDice)
#h1eff = moActDice.T.dot(mc0.get_hcore()).dot(moActDice)
eri = ao2mo.kernel(mol, moActDice)
tools.fcidump.from_integrals('FCIDUMP_can', h1eff, eri, mol.nao - ncore, mol.nelectron - 2*ncore, energy_core)

norbAct = mol.nao - norbFrozen
nelecAct = mol.nelectron - 2*norbFrozen
mc = mcscf.CASSCF(mf, norbAct, nelecAct)
mc.mo_coeff = mc1.mo_coeff
moFrozen = mc.mo_coeff[:,:norbFrozen]
moActive = mc0.mo_coeff[:,norbFrozen:]
core_dm = 2 * moFrozen.dot(moFrozen.T)
corevhf = mc.get_veff(mol, core_dm)
energy_core = mol.energy_nuc()
energy_core += numpy.einsum('ij,ji', core_dm, mc.get_hcore())
energy_core += numpy.einsum('ij,ji', core_dm, corevhf) * .5

rhfCoeffs = numpy.eye(norbAct)
prepVMC.writeMat(rhfCoeffs, "rhf.txt")
h1e, chol, nelec, enuc = generate_integrals(mol, mf.get_hcore(), mc0.mo_coeff, chol_cut=1e-5, verbose=True)

nbasis = h1e.shape[-1]
rotCorevhf = moActive.T.dot(corevhf).dot(moActive)
h1e = h1e[norbFrozen:, norbFrozen:] + rotCorevhf
chol = chol.reshape((-1, nbasis, nbasis))
chol = chol[:, norbFrozen:, norbFrozen:]
mol.nelec = (mol.nelec[0]-norbFrozen, mol.nelec[1]-norbFrozen)
enuc = energy_core

# after core averaging
nbasis = h1e.shape[-1]
print(f'nelec: {nelec}')
print(f'nbasis: {nbasis}')
print(f'chol.shape: {chol.shape}')
chol = chol.reshape((-1, nbasis, nbasis))
v0 = 0.5 * numpy.einsum('nik,njk->ij', chol, chol, optimize='optimal')
h1e_mod = h1e - v0

chol = chol.reshape((chol.shape[0], -1))
prepVMC.write_dqmc(h1e, h1e_mod, chol, sum(mol.nelec), nbasis, enuc, filename='FCIDUMP_chol')

# ccsd
mycc = cc.CCSD(mf)
mycc.frozen = norbFrozen
mycc.verbose = 5
mycc.kernel()
overlap = mf.get_ovlp(mol)
rotation = (mc0.mo_coeff[:, norbFrozen:].T).dot(overlap.dot(mf.mo_coeff[:, norbFrozen:]))
prepVMC.write_ccsd(mycc.t1, mycc.t2, rotation=rotation)

et = mycc.ccsd_t()
print('CCSD(T) energy', mycc.e_tot + et)
