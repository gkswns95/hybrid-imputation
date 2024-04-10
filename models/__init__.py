from brits.brits import BRITS
from graphimputer.graphimputer import BidirectionalGraphImputer
from naomi.naomi import NAOMI
from nrtsi.nrtsi import NRTSI
from strnn.dbhp import DBHP

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
    # elif model_name == "sociallstm":
    #     return SOCIALLSTM(params, parser)
    else:
        raise NotImplementedError
