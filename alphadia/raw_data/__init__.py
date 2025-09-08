from alphadia.raw_data.alpharaw_wrapper import (
    AlphaRawBase,
    MzML,
    Sciex,
    Thermo,
)
from alphadia.raw_data.bruker import TimsTOFTranspose

# TODO: inverse the dependency here:
from alphadia.search.jitclasses.alpharaw_jit import AlphaRawJIT
from alphadia.search.jitclasses.bruker_jit import TimsTOFTransposeJIT

DiaData = TimsTOFTranspose | AlphaRawBase | MzML | Sciex | Thermo
DiaDataJIT = TimsTOFTransposeJIT | AlphaRawJIT
