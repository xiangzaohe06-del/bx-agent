from langchain_openai import ChatOpenAI

from app.core.config import settings


def create_chat_model(*, max_tokens: int | None = None) -> ChatOpenAI:
    """通过百炼的 OpenAI 兼容接口创建非思考模式聊天模型。"""
    return ChatOpenAI(
        model=settings.llm.chat_model,
        api_key=settings.llm.dashscope_api_key,
        base_url=settings.llm.dashscope_base_url,
        extra_body={"enable_thinking": False},
        use_responses_api=False,
        max_tokens=max_tokens,
    )
