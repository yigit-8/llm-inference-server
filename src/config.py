from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Any causal LM works. distilgpt2 is small enough to run on a laptop CPU and
    # real enough to produce sensible text; the tests use a tiny random model so
    # they do not download hundreds of megabytes on every CI run.
    MODEL_NAME: str = "distilgpt2"

    # The batch is the throughput knob. Larger batches amortise the cost of
    # streaming weights from memory across more sequences, which is what makes
    # decoding go faster per token; they also raise the latency of any single
    # request once the batch is full and new arrivals have to queue.
    MAX_BATCH_SIZE: int = 8

    MAX_NEW_TOKENS: int = 32
    MAX_PROMPT_TOKENS: int = 256

    # How long the engine sleeps when it has nothing to do.
    IDLE_SLEEP_SECONDS: float = 0.005


settings = Settings()
