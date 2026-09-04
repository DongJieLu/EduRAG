"""全局配置：全部从 .env / 环境变量读取，禁止硬编码密钥与连接串。"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM ---
    llm_provider: str = "mock"  # deepseek | dashscope | ollama | mock
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    dashscope_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    llm_timeout_sec: int = 15

    # --- Embedding / Rerank（本地模型）---
    embed_model_name: str = "BAAI/bge-m3"
    embed_batch_size: int = 64
    rerank_model_name: str = "BAAI/bge-reranker-large"

    # --- MySQL ---
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "eduqa"
    mysql_password: str = "eduqa"
    mysql_db: str = "eduqa"

    # --- Redis ---
    redis_host: str = "localhost"
    redis_port: int = 6379

    # --- 向量库（开发期 Chroma）---
    chroma_persist_dir: str = "./data/chroma"

    # --- 服务 ---
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
