import logging


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # httpx logs one line per API call at INFO; the app already logs what matters.
    logging.getLogger("httpx").setLevel(logging.WARNING)
