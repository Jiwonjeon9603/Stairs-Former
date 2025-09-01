from .odis_learner import ODISLearner
from .updet_learner import UPDeTLearner
from .bc_learner import BCLearner
from .updet_learner_bc import UPDeTBCLearner
from .updet_mt_learner_bc import UPDeTMTBCLearner
from .updet_learner_bc_mix import UPDeTBCMixLearner

REGISTRY = {}

REGISTRY["odis_learner"] = ODISLearner
REGISTRY["updet_learner"] = UPDeTLearner
REGISTRY["bc_learner"] = BCLearner
REGISTRY["updet_bc_learner"] = UPDeTBCLearner
REGISTRY["updet_mt_bc_learner"] = UPDeTMTBCLearner
REGISTRY["updet_bc_mix_learner"] = UPDeTBCMixLearner
