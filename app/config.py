from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    database_url: str = Field(..., alias="DATABASE_URL")
    alert_webhook_url: str = Field("", alias="ALERT_WEBHOOK_URL")
    report_schedule_cron: str = Field("0 0 * * *", alias="REPORT_SCHEDULE_CRON")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    api_key: str = Field("", alias="DRIFTWATCH_API_KEY")
    psi_yellow_threshold: float = Field(0.10, alias="PSI_YELLOW_THRESHOLD")
    psi_red_threshold: float = Field(0.20, alias="PSI_RED_THRESHOLD")
    # At 200 samples, mean null PSI is approximately 0.05 for 10 bins.
    min_sample_warning: int = Field(200, alias="MIN_SAMPLE_WARNING")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_psi_thresholds(self):
        if not 0 < self.psi_yellow_threshold <= self.psi_red_threshold:
            raise ValueError("PSI thresholds must satisfy 0 < yellow <= red")
        return self


settings = Settings()
