import cocoindex
from cocoindex_code_mcp_server.cocoindex_config import code_embedding_flow

# @cocoindex.flow_def(name="CodeEmbedding")
# def code_embedding_flow(
#
#     code_embeddings.export(
#        "code_embeddings",
#        cocoindex.targets.Postgres(),


def get_default_db_name() -> str:
    return cocoindex.utils.get_target_default_name(
        flow=code_embedding_flow, target_name="code_embeddings"
    ).lower()
