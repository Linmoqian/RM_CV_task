from .types import ArmorColor, ArmorSize, LightBar, Armor, DetectResult
from .config import DetectorConfig
from .color_segmenter import ColorSegmenter
from .light_bar import LightBarFinder
from .armor import ArmorMatcher
from .detector import ArmorDetector

__all__ = [
    # Types
    "ArmorColor",
    "ArmorSize",
    "LightBar",
    "Armor",
    "DetectResult",
    # Config
    "DetectorConfig",
    # Components
    "ColorSegmenter",
    "LightBarFinder",
    "ArmorMatcher",
    # Main
    "ArmorDetector",
]
