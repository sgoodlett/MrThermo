import re
from .data.solvent_data import solvdata_str
import rdkit

class Solvent():
    def __init__(self, name, **kwargs):
        self.name = name
        self.eps = float(kwargs.pop("eps").strip())
        self.M = float(kwargs.pop("M").strip())
        self.rho = float(kwargs.pop("rho").strip())
        self.V_vdw = float(kwargs.pop("V_vdw").strip())

    @classmethod
    def from_string(cls, strang):
        rgxstr = r"^(\d+)\s((?:[\dA-Z-],?)*[A-Za-z\s-]+\d?[A-Za-z\s-]+)\s+(-?\d+\.?\d*)\s*(-?\d+\.?\d*)\s*(-?\d+\.?\d*)\s*(-?\d+\.?\d*)$"
        rgx = re.compile(rgxstr)
        try:
            ignore, name, eps_r, M, rho, V_vdw = rgx.match(strang).groups()
        except:
            print(strang)
            raise Exception("Error found, please fix using this helpful and vague message.")
        return cls(name, eps=eps_r, M=M, rho=rho, V_vdw=V_vdw)

solvents = {}
for solvent in solvdata_str.splitlines():
    solvent_data = Solvent.from_string(solvent)
    solvents[solvent_data.name] = solvent_data
