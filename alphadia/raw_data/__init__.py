from alphadia.raw_data.alpharaw_wrapper import AlphaRaw
from alphadia.raw_data.bruker import TimsTOFTranspose
from alphadia.raw_data.jitclasses.alpharaw_jit import AlphaRawJIT
from alphadia.raw_data.jitclasses.bruker_jit import TimsTOFTransposeJIT

DiaData = TimsTOFTranspose | AlphaRaw
DiaDataJIT = TimsTOFTransposeJIT | AlphaRawJIT
