from alphadia.raw_data.jitclasses.alpharaw_jit import AlphaRawJIT
from alphadia.raw_data.jitclasses.bruker_jit import TimsTOFTransposeJIT

DiaDataJIT = TimsTOFTransposeJIT | AlphaRawJIT
