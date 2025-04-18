import numpy as np

class Ensemble():
    def __init__(self, conformers, solvent):
        self.solvent = solvent
        self.conformers = conformers

    def Gaq(self, temp):
        # G = G(ref) - RT ln( 1 + sum( e^( -dGi / RT) ) )
        # dGi = G(i) - G(ref)
        Gi = np.array([i.Gaq(self.solvent, temp) for i in self.conformers])
        ref_G = np.min(Gi)
        dGi = Gi - ref_G
        # Units?
        kb = 3.166811563e-6 # Eh/K
        T = temp + 273.15 # K
        return ref_G - kb*T * np.log(np.sum(np.exp(-dGi/(kb*T))))
    
