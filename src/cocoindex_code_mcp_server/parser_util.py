from cocoindex_code_mcp_server import LOGGER

def update_defaults(d: dict, defaults: dict) -> None:
    for k, v in defaults.items():
        d.setdefault(k, v)
