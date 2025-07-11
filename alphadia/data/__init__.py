from alphadia.data.alpharaw_wrapper import AlphaRaw
from alphadia.data.bruker import TimsTOFTranspose
from alphadia.data.jitclasses.alpharaw_jit import AlphaRawJIT
from alphadia.data.jitclasses.bruker_jit import TimsTOFTransposeJIT

DiaData = TimsTOFTranspose | AlphaRaw
DiaDataJIT = TimsTOFTransposeJIT | AlphaRawJIT
