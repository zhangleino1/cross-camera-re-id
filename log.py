import logging
import sys # Keep sys for potential future basic configuration needs
from typing import Optional

# Basic configuration for the logger, can be expanded if needed
# but kept minimal to avoid side-effects from the original trackers.log
logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_logger(name: Optional[str]) -> logging.Logger:
    return logging.getLogger(name)
