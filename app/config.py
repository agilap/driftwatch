from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    database_url: str = Field(..., alias="DATABASE_URL")
    secret_key: str = Field(..., alias="SECRET_KEY")
    alert_webhook_url: str = Field("", alias="ALERT_WEBHOOK_URL")
    report_schedule_cron: str = Field("0 0 * * 1", alias="REPORT_SCHEDULE_CRON")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    psi_yellow_threshold: float = Field(0.10, alias="PSI_YELLOW_THRESHOLD")
    psi_red_threshold: float = Field(0.20, alias="PSI_RED_THRESHOLD")
    ks_pvalue_threshold: float = Field(0.05, alias="KS_PVALUE_THRESHOLD")
    min_sample_warning: int = Field(30, alias="MIN_SAMPLE_WARNING")

    model_config = SettingsConfigDict(
        env_file=".env.example",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
