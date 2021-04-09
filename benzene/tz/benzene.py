import numpy
from pyscf import gto, scf, mcscf, fci, ao2mo, lib, tools, cc
from pauxy.utils.from_pyscf import generate_integrals
import scipy.linalg as la
from pyscf.shciscf import shci
import prepVMC

atomstring='''
C 0.000000 1.396792    0.000000
C 0.000000 -1.396792    0.000000
C 1.209657 0.698396    0.000000
C -1.209657 -0.698396    0.000000
C -1.209657 0.698396    0.000000
C 1.209657 -0.698396    0.000000
H 0.000000 2.484212    0.000000
H 2.151390 1.242106    0.000000
H -2.151390 -1.242106    0.000000
H -2.151390 1.242106    0.000000
H 2.151390 -1.242106    0.000000
H 0.000000 -2.484212    0.000000
'''
mol = gto.M(
    atom=atomstring,
    basis='ccpvtz',
    verbose=4,
    unit='angstrom',
    symmetry=1,
    spin=0)
mf = scf.RHF(mol)
mf.kernel()
#mc0 = mcscf.CASSCF(mf, 6, 6)
norbFrozen = 6
ncore = 6
#mc0.frozen = norbFrozen
#mo = mc0.sort_mo_by_irrep({'Ag': 0, 'B1u': 2, 'B2u': 0, 'B3u': 0, 'B1g': 0, 'B2g': 1, 'B3g': 2, 'Au': 1}, {'Ag': 6, 'B1u': 0, 'B2u': 5, 'B3u': 4, 'B1g': 3, 'B2g': 0, 'B3g': 0, 'Au': 0})
#mc0.verbose = 4
#mc0.kernel(mo)

nactDice = 100
mc1 = shci.SHCISCF(mf, nactDice, mol.nelectron - 2*ncore)
mc1.chkfile = 'benzene_SHCISCF.chk'
#chkfile = 'benzene_SHCISCF.chk'
#mc1.__dict__.update(lib.chkfile.load(chkfile, 'mcscf'))
mc1.frozen = norbFrozen
mc1.fcisolver.sweep_iter = [ 0 ]
mc1.fcisolver.sweep_epsilon = [ 1e-3 ]
mc1.fcisolver.nPTiter = 0
mc1.max_cycle_macro = 10
mc1.internal_rotation = True
mc1.fcisolver.nPTiter = 0  # Turns off PT calculation, i.e. no PTRDM.
mc1.fcisolver.mpiprefix = "mpirun -np 36"
mc1.fcisolver.scratchDirectory = "/rc_scratch/anma2640/benzene/"
mc1.mc1step()

# for dice calculation
nelecAct = mol.nelectron - 2*ncore
mc0 = mcscf.CASSCF(mf, nactDice, nelecAct)
mc0.mo_coeff = mc1.mo_coeff
moCore = mc0.mo_coeff[:,:ncore]
core_dm = 2 * moCore.dot(moCore.T)
corevhf = mc0.get_veff(mol, core_dm)
energy_core = mol.energy_nuc()
energy_core += numpy.einsum('ij,ji', core_dm, mc0.get_hcore())
energy_core += numpy.einsum('ij,ji', core_dm, corevhf) * .5
moActDice = mc0.mo_coeff[:, ncore:ncore + nactDice]
h1eff = moActDice.T.dot(mc0.get_hcore() + corevhf).dot(moActDice)
eri = ao2mo.kernel(mol, moActDice)
tools.fcidump.from_integrals('FCIDUMP_can', h1eff, eri, nactDice, nelecAct, energy_core)

# set up dqmc calculation
# calculate core contribution
norbAct = mol.nao - norbFrozen
nelecAct = mol.nelectron - 2*norbFrozen
mc = mcscf.CASSCF(mf, norbAct, nelecAct)
mc.mo_coeff = mc0.mo_coeff
moFrozen = mc.mo_coeff[:,:norbFrozen]
moActive = mc.mo_coeff[:,norbFrozen:]
core_dm = 2 * moFrozen.dot(moFrozen.T)
corevhf = mc.get_veff(mol, core_dm)
energy_core = mol.energy_nuc()
energy_core += numpy.einsum('ij,ji', core_dm, mc.get_hcore())
energy_core += numpy.einsum('ij,ji', core_dm, corevhf) * .5


rhfCoeffs = numpy.eye(norbAct)
prepVMC.writeMat(rhfCoeffs, "rhf.txt")
h1e, chol, nelec, enuc = generate_integrals(mol, mf.get_hcore(), mc.mo_coeff, chol_cut=1e-5, verbose=True)

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

overlap = mf.get_ovlp(mol)
rotation = (mc.mo_coeff[:, norbFrozen:].T).dot(overlap.dot(mf.mo_coeff[:, norbFrozen:]))
print(rotation.shape)
# ccsd
mycc = cc.CCSD(mf)
mycc.frozen = 6
mycc.verbose = 5
mycc.kernel()
prepVMC.write_ccsd(mycc.t1, mycc.t2, rotation=rotation)

et = mycc.ccsd_t()
print('CCSD(T) correlation energy', mycc.e_corr + et)

# make fictitious valence only molecule and perform ghf
#norb = norbAct
#molA = gto.M()
#molA.nelectron = nelecAct
#molA.verbose = 4
#molA.incore_anyway = True
#gmf = scf.GHF(molA)
#gmf.get_hcore = lambda *args: la.block_diag(h1eff, h1eff)
#gmf.get_ovlp = lambda *args: numpy.identity(2*norb)
#gmf.energy_nuc = lambda *args: energy_core
#gmf._eri = eri
#dm = gmf.get_init_guess()
#dm = dm + 2 * numpy.random.rand(2*norb, 2*norb)
#gmf.level_shift = 0.1
#gmf.max_cycle = 500
#print(gmf.kernel(dm0 = dm))
#prepVMC.writeMat(gmf.mo_coeff, "ghf.txt", False)
