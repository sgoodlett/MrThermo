import re
import numpy as np
import qcelemental as qcel
import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
from .solvent_params import solvents
import molsym

class Conformer():
    def __init__(self, species, conf, solvent_string, T, P):
        # Standardize units to kcal/mol, K, Ang, amu

        self.kb = 1.987204259e-3 # kcal/mol K
        self.Na = qcel.constants.get("Avogadro constant") # Avagadro number
        self.P = P # Bar
        pressure_scale = (1e-30) * self.Na / (1e-2 * qcel.constants.cal2J) # kcal/mol Angstrom^3
        self.P *= pressure_scale
        self.T = 298.15 # K

        self.species = species
        self.conf = int(conf)
        self.datapath = "/home/sgoodlett/fuchs/2_NiksCarbenes/"

        self.mol = self.get_structure()

        # Symmetry number
        molsym_mol = molsym.Molecule.from_schema(self.mol.dict())
        symtext = molsym.Symtext.from_molecule(molsym_mol)
        self.sigma_r = symtext.rotational_symmetry_number
        
        self.solvent = solvents[solvent_string]
        
        self.rho = self.solvent.rho # Solvent density g/cm^3
        density_scale = self.Na / ((1.0e8)**3) # amu / Angstrom^3
        self.rho *= density_scale
        self.M_wS = self.solvent.M # Solvent molecular weight amu
        self.eps_r = self.solvent.eps # Something?
        self.V_S = self.solvent.V_vdw
        self.V_M = self.volume
        #strang = self.mol.to_string(dtype="xyz", units="Angstrom")
        #self.rd_solute = rdkit.Chem.MolFromXYZBlock(strang.strip())
        #self.V_M = rdkit.Chem.AllChem.ComputeMolVolume(self.rd_solute, gridSpacing=0.1)

    def __repr__(self):
        return f"{self.species}/conf_{self.conf}"

    def get_structure(self):
        # It's easier to grab from the frequency file since I've already separated them
        opt_path = self.datapath + "freqs"
        with open(f"{opt_path}/{self.species}/conf_{self.conf}/mol.xyz", "r") as fn:
            xyzfile = fn.read()
        # This will convert to Bohr, sigh
        return qcel.models.Molecule.from_data(xyzfile)

    @property
    def volume(self):
        # And then auto converts back to Angstrom, sigh
        rdmol = rdkit.Chem.MolFromXYZBlock(self.mol.to_string(dtype="xyz", units="angstrom"))
        # A nice fine grid, hopefully
        # Difference between grid spacing of 0.1 and 0.01: 103.526 vs 103.597 for Carbene
        return rdkit.Chem.AllChem.ComputeMolVolume(rdmol, gridSpacing=0.1)

    def get_solvent_correction(self, solvent):
        final_energy_rgx = re.compile(r"FINAL SINGLE POINT ENERGY\s*(-\d*\.\d*)")
        cds_rgx = re.compile(r"SMD CDS \(Gcds\)\s*:\s*(-?\d*\.\d*)")
        solvent_path = f"{self.datapath}/solvent_corrections/{solvent}/{self.species}/conf_{self.conf}"
        with open(f"{solvent_path}/cpcm/mol.out") as fn:
            strang = fn.read()
        solvent_energy = float(final_energy_rgx.search(strang).groups()[0])
        with open(f"{solvent_path}/smd/mol.out") as fn:
            strang = fn.read()
        cds_energy = float(cds_rgx.search(strang).groups()[0])
        
        return solvent_energy, cds_energy

    def get_thermo_data(self, temp):
        base_rgx = r"\s*[\.]+\s*(-?\d*\.\d*)"
        elec_energy_rgx = re.compile(r"Electronic energy"+base_rgx)
        enthalpy_rgx = re.compile(r"Total Enthalpy"+base_rgx)
        entropy_rgx = re.compile(r"Final entropy term"+base_rgx)
        Gibbs_rgx = re.compile(r"Final Gibbs free energy"+base_rgx)

        # Get energy from freq computation
        with open(f"{self.datapath}/freqs/{self.species}/conf_{self.conf}/mol.out", "r") as fn:
            strang = fn.read()
        electronic_energy = float(elec_energy_rgx.search(strang).groups()[0])

        # Get Enthalpy, entropy, and Gibbs free energy from thermo50 analysis
        with open(f"{self.datapath}/freqs/{self.species}/conf_{self.conf}/thermo{temp}/mol.out", "r") as fn:
            strang = fn.read()
        enthalpy_corr = float(enthalpy_rgx.search(strang).groups()[0])
        entropy_corr = float(entropy_rgx.search(strang).groups()[0])
        Gibbs_corr = float(Gibbs_rgx.search(strang).groups()[0])

        return electronic_energy, enthalpy_corr, entropy_corr, Gibbs_corr

    def get_energy_correction(self, temp):
        # Temperature doesn't matter here other than to make get_thermo_data work
        ref_electronic_energy, *_ = self.get_thermo_data(temp)
        with open(f"{self.datapath}/energy_corrections/gas_dlpno/{self.species}/conf_{self.conf}/mol.out", "r") as fn:
            strang = fn.read()
        final_energy_rgx = re.compile(r"FINAL SINGLE POINT ENERGY\s*(-\d*\.\d*)")
        energy = float(final_energy_rgx.search(strang).groups()[0])
        return energy - ref_electronic_energy

    def Gaq(self, solvent, temp):
        self.T = temp + 273.15
        # G = Ggas + dGsolv
        # Ggas = E + dE_DLPNO + Gcorrection
        # dGsolv = dG_ENP + dG_CDS
        # dG_ENP = E_solv - E_gas
        electronic_energy, *_, Gibbs_corr = self.get_thermo_data(temp)        
        solvent_energy, cds_energy = self.get_solvent_correction(solvent)
        dG_ENP = solvent_energy - electronic_energy
        dGentropy = -self.T * (self.dS_trans() + self.dS_rot() + self.S_cav_eps()) / 627.509 
        dGsolv = dG_ENP + dGentropy
        dE_DLPNO = self.get_energy_correction(temp)
        Ggas = electronic_energy + dE_DLPNO + Gibbs_corr
        #print(f"{self.species} {self.conf}: {627.509*dE_DLPNO:5.2f} {627.509*dG_ENP:5.2f} {627.509*dGsolv:5.2f} {67.509*dGentropy:5.2f}")
        # Remember standard state correction
        #print(f"{self.species} conf {self.conf}")
        #print(f"Ref E: {electronic_energy}")
        #print(f"dE_DLPNO: {dE_DLPNO}")
        #print(f"GRRHO: {Gibbs_corr}")
        #print(f"dGsolv: {dG_ENP}")
        #print(f"dGCDS: {cds_energy}")
        #return electronic_energy, dG_ENP, dGentropy, dGsolv, dE_DLPNO, Gibbs_corr
        return Ggas + dGsolv + (1.90 / 627.509)

    def dS_trans(self, rg=0.0):
        # Translational entropy change of solvation
        V_g = self.kb * self.T / self.P
        V_free = (self.M_wS/self.rho) - self.V_S
        nu_c = (self.V_M**(1.0/3.0) + V_free**(1.0/3.0))**3
        rc = (3*nu_c / (4*np.pi))**(1.0/3.0)
        x = max(V_free**(2.0/3.0) - self.V_M**(2.0/3.0), 0.0) / (V_free**(2.0/3.0) + self.V_S**(2.0/3.0))
        N_x = 4 * ((4*np.pi/3)**(2.0/3.0)) * (rc**2 / (V_free**(2.0/3.0) + self.V_S**(2.0/3.0)))
        N_c = 1 + N_x * ((1/(1-x)) - 1)
        V_s = N_c * (4.0/3.0) * np.pi * (rc - rg)**3
        #print("rcg:   ", rc, rg, N_c, V_s, V_g)
        return self.kb * np.log(V_s / V_g)

    def dS_rot(self):
        # Rotational entropy change of solvation
        # Undo qcelemental
        mol_xyz = self.mol.geometry * qcel.constants.bohr2angstroms
        
        rs = np.sqrt(np.power(mol_xyz, 2).sum(axis=1))
        rmean = rs.sum() / len(rs)
        rg = np.sqrt(np.power(rs - rmean, 2).sum() / len(rs))
        return self.dS_trans(rg) - self.dS_trans(0.0)

    def S_cav_eps(self):
        # Cavity entropy as calculated with epsilon method of Garza
        R = (self.V_M / self.V_S)**(1.0/3.0)
        y = (3.0/(4.0*np.pi)) * ((self.eps_r - 1.0) / (self.eps_r + 2.0))
        omy = 1 - y
        Sc_eps = -np.log(omy)
        Sc_eps += 3.0 * R / omy
        Sc_eps += ((3.0*y/omy) + 4.5*((y/omy)**2)) * (R**2)
        Sc_eps *= self.kb
        # Negative sign missing in Garza, tsk tsk
        return -1.0 * Sc_eps

#class Ensemble():
#    def __init__(self, conformers, solvent):
#        self.solvent = solvent
#        self.conformers = conformers
#
#    def Gaq(self, temp):
#        # G = G(ref) - RT ln( 1 + sum( e^( -dGi / RT) ) )
#        # dGi = G(i) - G(ref)
#        Gi = np.array([i.Gaq(self.solvent, temp) for i in self.conformers])
#        ref_G = np.min(Gi)
#        dGi = Gi - ref_G
#        # Units?
#        kb = 3.166811563e-6 # Eh/K
#        T = temp + 273.15 # K
#        return ref_G - kb*T * np.log(np.sum(np.exp(-dGi/(kb*T))))
        
def reaction_energy(species):
    # DeltaG = G(Complex) - G(Carbene) - G(Nucleophile)
    pass

if __name__ == "__main__":
    nucleophiles = ["dmap", "et3n", "imidazole", "piperidine", "pyridine", "pyrrolidinylpyridine"]
    for n in nucleophiles:
        print(reaction_energy(n))
