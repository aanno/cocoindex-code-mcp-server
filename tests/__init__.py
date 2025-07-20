# Test package for haskell-tree-sitter project

import logging

# Set up logging for tests (similar to src/__init__.py)
logging.basicConfig(
    level=logging.INFO,
    format='%(name)-12s: %(levelname)-8s %(message)s'
)

# Create LOGGER for tests
LOGGER = logging.getLogger(__name__)