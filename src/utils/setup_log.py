import logging

def setup_logging():
    import warnings

    logging.basicConfig(
        level=logging.INFO,
        format='(%(asctime)s) %(levelname)s ➧ %(message)s',
        datefmt='%d-%m-%Y %H:%M:%S'
    )

    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
