# Main package for haskell-tree-sitter project

import logging

# https://stackoverflow.com/questions/13479295/python-using-basicconfig-method-to-log-to-console-and-file

# set up logging to file
logging.basicConfig(
    filename='cocoindex-code-mcp-server.log',
    level=logging.DEBUG, 
    format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)

LOGGER = logging.getLogger(__name__)
# add the handler to the root logger
LOGGER.addHandler(console)
