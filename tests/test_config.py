"""配置模块单元测试。"""
from app.config import Settings


def test_settings_defaults_when_no_env_file():
    s = Settings(_env_file=None)
    assert s.llm_provider == "mock"
    assert s.embed_model_name == "BAAI/bge-m3"
    assert s.rerank_model_name == "BAAI/bge-reranker-large"
    assert s.deepseek_model == "deepseek-chat"
    assert s.embed_batch_size == 64
    assert s.mysql_port == 3308
    assert s.redis_port == 6379


def test_settings_reads_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("MYSQL_PORT", "3307")
    s = Settings(_env_file=None)
    assert s.llm_provider == "deepseek"
    assert s.mysql_port == 3307
