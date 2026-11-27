from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, computed_field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

# 以当前文件为基准，找向上2级的父目录，也就是项目的根目录
APP_ROOT = Path(__file__).resolve().parents[2]

# 在创建settings和客户端之前统一加载，兼容LangSmith等直接读取进程环境变量的SDK。
# 使用固定路径，避免启动目录影响配置；系统环境变量优先，不被.env覆盖。
load_dotenv(APP_ROOT / ".env", override=False, encoding="utf-8")

class EnvSettings(BaseSettings):
    """公共settings，配置.env文件位置"""
    model_config = SettingsConfigDict(
        env_file=(APP_ROOT / ".env",),
        env_file_encoding="utf-8",
        extra="ignore",  # settings中未定义的字段被忽略
    )

class AppSettings(EnvSettings):
    """应用相关设置"""
    name: str = Field(alias="APP_NAME")
    host: str = Field(alias="APP_HOST", default="127.0.0.1")
    port: int = Field(alias="APP_PORT", default=8001)
    debug: bool = Field(alias="APP_DEBUG", default=True, description="是否开启FastAPI的DEBUG模式")

class DatabaseSettings(EnvSettings):
    """数据库相关设置"""
    host: str = Field(alias="DB_HOST", default="localhost")
    port: int = Field(alias="DB_PORT", default=5432)
    name: str = Field(alias="DB_NAME", default="insurance")
    user: str = Field(alias="DB_USER", default="insurance")
    password: str = Field(alias="DB_PASSWORD")
    echo: bool = Field(alias="DB_ECHO", default=False)
    pool_size: int = Field(alias="DB_POOL_SIZE", default=5)
    max_overflow: int = Field(alias="DB_MAX_OVERFLOW", default=10)
    pool_timeout: int = Field(alias="DB_POOL_TIMEOUT", default=30)
    pool_recycle: int = Field(alias="DB_POOL_RECYCLE", default=1800)

    def build_url(self, scheme: str):
        # 自动拼接URL路径
        return PostgresDsn.build(
            scheme=scheme,
            host=self.host,
            port=self.port,
            username=self.user,
            password=self.password
        ).encoded_string()

    @computed_field
    @property
    def url(self) -> str:
        # 自动拼接URL路径
        return self.build_url("postgresql+asyncpg")

    @computed_field
    @property
    def checkpoint_url(self) -> str:
        # 自动拼接URL路径
        return self.build_url("postgresql")

class LLMSettings(EnvSettings):
    """模型相关配置"""
    chat_model: str = Field(alias="CHAT_MODEL", default="qwen4-max")
    dashscope_api_key: str = Field(alias="DASHSCOPE_API_KEY")
    dashscope_base_url: str = Field(alias="DASHSCOPE_BASE_URL")

class LoggingSettings(EnvSettings):
    """日志相关配置"""
    level: str = Field(alias="LOGGING_LEVEL", default="INFO") # 可选值：DEBUG、INFO、WARNING、ERROR、CRITICAL

class ContextSettings(EnvSettings):
    """长对话摘要配置，token数量为估算值。"""
    summary_enabled: bool = Field(alias="CONTEXT_SUMMARY_ENABLED", default=True)
    summary_trigger_tokens: int = Field(alias="CONTEXT_SUMMARY_TRIGGER_TOKENS", default=12000, gt=0)
    summary_keep_tokens: int = Field(alias="CONTEXT_SUMMARY_KEEP_TOKENS", default=4000, gt=0)
    summary_max_tokens: int = Field(alias="CONTEXT_SUMMARY_MAX_TOKENS", default=1600, gt=0)

class SupervisorSettings(EnvSettings):
    """限制每轮子Agent委派次数，防止循环调用。"""
    max_delegations: int = Field(alias="SUPERVISOR_MAX_DELEGATIONS", default=3, ge=1, le=10)

class RAGSettings(EnvSettings):
    """RAG相关配置"""
    mineru_token: str = Field(alias="MINERU_TOKEN")
    milvus_host: str = Field(alias="MILVUS_HOST", default="127.0.0.1")
    milvus_port: int = Field(alias="MILVUS_PORT", default=19530)
    milvus_collection: str = Field(alias="MILVUS_COLLECTION", default="insurance_collection")
    embedding_provider: Literal["ollama", "dashscope"] = Field(alias="RAG_EMBEDDING_PROVIDER", default="ollama")
    embedding_model: str = Field(alias="RAG_EMBEDDING_MODEL", default="qwen3-embedding:0.6b")
    embedding_dimensions: int = Field(alias="RAG_EMBEDDING_DIMENSIONS", default=1024, gt=0)
    embedding_base_url: str = Field(alias="DASHSCOPE_EMBEDDING_BASE_URL", default="https://dashscope.aliyuncs.com/api/v1")

    @computed_field
    @property
    def milvus_url(self) -> str:
        return f"http://{self.milvus_host}:{self.milvus_port}"

class Settings(BaseSettings):
    app: AppSettings = Field(default_factory=AppSettings)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    supervisor: SupervisorSettings = Field(default_factory=SupervisorSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)

# 项目启动直接初始化
settings = Settings()
