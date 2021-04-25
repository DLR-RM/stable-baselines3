import environ

from .config import Config


settings = environ.to_config(Config)
