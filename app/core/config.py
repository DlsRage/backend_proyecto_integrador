from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "clustering-backend"
    API_PREFIX: str = "/api/v1"

    # Object storage (DigitalOcean Spaces - S3 compatible)
    USE_SPACES: bool = True
    SPACES_REGION: str = "NYC3"
    SPACES_ENDPOINT: str = 'https://nyc3.digitaloceanspaces.com'
    SPACES_BUCKET: str = "imagenes-proyecto"
    SPACES_KEY: str = "DO8012KLZA9PEWTVRQA3"
    SPACES_SECRET: str = "8+xFL9e4KlZeI3gsVjarz8jnT62FxSyMfE8EbkgSY5w"

    # Processing
    TMP_DIR: str = "/tmp/jobs"
    MAX_IN_FLIGHT_IMAGES: int = 8   # micro-lote
    RANDOM_SEED: int = 123

settings = Settings()

