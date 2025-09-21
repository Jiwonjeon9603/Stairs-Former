from .odis_agent import ODISAgent
from .updet_agent import UPDeTAgent
from .bc_agent import BCAgent
from .bcr_agent import BCRAgent
from .hr_agent import HierReasoningAgent
from .updet_hrm_agent import HRMAgent
from .hrm_hst_agent import HRMHSTAgent
from .hrm_hst_agent_wo_h import HRMHSTWoHAgent
from .tsffn_agent import TSFFNAgent
from .repeat_agent import ReAgent

REGISTRY = {}

REGISTRY["mt_odis"] = ODISAgent
REGISTRY["mt_updet"] = UPDeTAgent
REGISTRY["mt_bc"] = BCAgent
REGISTRY["mt_bcr"] = BCRAgent
REGISTRY["mt_hr"] = HierReasoningAgent
REGISTRY["mt_hrm"] = HRMAgent
REGISTRY["mt_hrm_hst"] = HRMHSTAgent
REGISTRY["mt_hrm_hst_wo_h"] = HRMHSTWoHAgent
REGISTRY["mt_tsffn"] = TSFFNAgent
REGISTRY["mt_repeat"] = ReAgent
