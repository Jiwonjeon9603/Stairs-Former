from .odis_agent import ODISAgent
from .updet_agent import UPDeTAgent
from .bc_agent import BCAgent
from .bcr_agent import BCRAgent
from .hr_agent import HierReasoningAgent
from .updet_hrm_agent import HRMAgent
from .hrm_hst_agent import HRMHSTAgent

REGISTRY = {}

REGISTRY["mt_odis"] = ODISAgent
REGISTRY["mt_updet"] = UPDeTAgent
REGISTRY["mt_bc"] = BCAgent
REGISTRY["mt_bcr"] = BCRAgent
REGISTRY["mt_hr"] = HierReasoningAgent
REGISTRY["mt_hrm"] = HRMAgent
REGISTRY["mt_hrm_hst"] = HRMHSTAgent
