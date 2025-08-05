from alphadia.raw_data.alpharaw_wrapper import (
    AlphaRawBase,
    MzML,
    Sciex,
    Thermo,
)
from alphadia.raw_data.bruker import TimsTOFTranspose
from alphadia.raw_data.jitclasses.alpharaw_jit import AlphaRawJIT
from alphadia.raw_data.jitclasses.bruker_jit import TimsTOFTransposeJIT

DiaData = TimsTOFTranspose | AlphaRawBase | MzML | Sciex | Thermo
DiaDataJIT = TimsTOFTransposeJIT | AlphaRawJIT
