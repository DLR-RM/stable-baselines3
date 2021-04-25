import environ


@environ.config(prefix='SB3')
class Config:
    """Configuration class.

    The `environ_config` library allows for its attributes to be retrieved from
    environment variables, the prefix for this class and the attribute name
    defines the variable name to be used, e.g. SB3_LOG_FORMAT for LOG_FORMAT.

    This class is used in settings.__init__.py to instantiate an object from
    the environment.
    """

    # Add validators
    LOG_DIR = environ.var(default=None)
    LOG_FORMAT = environ.var(default='stdout,log,csv', converter=lambda x: x.split(','))
    LOG_LEVEL = environ.var(default=0)

    # -- LOGGING --

    LOGGING_LEVEL = environ.var(default='INFO')

    @property
    def LOGGING(self):
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'default': {
                    'format': '[%(asctime)s][%(name)s][%(levelname)s]: %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S',
                },
            },
            'handlers': {
                'default': {
                    'class': 'logging.StreamHandler',
                    'level': self.LOGGING_LEVEL,
                    'formatter': 'default',
                },
            },
            'loggers': {
                '': {
                    'handlers': ['default'],
                    'level': self.LOGGING_LEVEL,
                    'propagate': True,
                },
            },
        }
