from .basic_controller import BasicMAC
from .mcemncd_controller import MCEMNCDMAC

REGISTRY = {}
REGISTRY["basic_mac"] = BasicMAC
REGISTRY["mcemncd_mac"] = MCEMNCDMAC
