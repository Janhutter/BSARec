from model.bsarec import BSARecModel
from model.caser import CaserModel
from model.gru4rec import GRU4RecModel
from model.sasrec import SASRecModel
from model.bert4rec import BERT4RecModel
from model.fmlprec import FMLPRecModel
from model.duorec import DuoRecModel
from model.fearec import FEARecModel
# from model.bsarec_wavelet_learned import BSARec_WaveletModel_learned
from model.bsarec_wavelet import BSARec_WaveletModel
from model.wavelet import WaveletModel
from model.fourier import FourierModel
# from model.bsarec_skip import BSARecModelPadding
from model.bsarec_skip import BSARecModelPadding

MODEL_DICT = {
    "bsarec": BSARecModel,
    "caser": CaserModel,
    "gru4rec": GRU4RecModel,
    "sasrec": SASRecModel,
    "bert4rec": BERT4RecModel,
    "fmlprec": FMLPRecModel,
    "duorec": DuoRecModel,
    "fearec": FEARecModel,
    'bsarec_wavelet': BSARec_WaveletModel,
    'wavelet': WaveletModel,
    'fourier': FourierModel,
    'bsarec_skip': BSARecModelPadding,
    }