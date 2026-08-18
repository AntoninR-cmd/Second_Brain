from __future__ import annotations

import logging


def configure_application_logging() -> None:
    """Expose application metrics without enabling verbose dependency logs."""

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    logging.getLogger("second_brain").setLevel(logging.INFO)
