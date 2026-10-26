from patchology.log import configure_logging

# INFO by default; export PATHOTRAIN_LOG_LEVEL=DEBUG (or ALL) for more, see patchology.log.
configure_logging("pathotrain")


def hello() -> str:
    return "Hello from pathotrain!"
