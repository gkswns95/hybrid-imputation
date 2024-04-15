from .ours.dbhp import DBHP
from .baselines.brits.brits import BRITS
from .baselines.naomi.naomi import NAOMI
from .baselines.nrtsi.nrtsi import NRTSI
from .baselines.graphimputer.graphimputer import BidirectionalGraphImputer
from .baselines.latentode.create_latent_ode_model import create_LatentODE_model 
# from .baselines.social_lstm.social_lstm import SOCIALLSTM

def load_model(model_name, params, parser=None):
    model_name = model_name.lower()

    if model_name == "ours":
        return DBHP(params, parser)
    elif model_name == "brits":
        return BRITS(params, parser)
    elif model_name == "naomi":
        return NAOMI(params, parser)
    elif model_name == "nrtsi":
        return NRTSI(params, parser)
    elif model_name == "graphimputer":
        return BidirectionalGraphImputer(params, parser)
    elif model_name =='latentode':
        return create_LatentODE_model(params, parser)
    # elif model_name == "sociallstm":
    #     return SOCIALLSTM(params, parser)
    else:
        raise NotImplementedError