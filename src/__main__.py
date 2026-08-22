from typing import Any, Callable

import fire

from src.cli_interface import Cli


if __name__ == '__main__':
    cli = Cli()
    commands: dict[str, Callable[..., Any]] = {
        "index": cli.index,
        "search": cli.search,
        "search_dataset": cli.search_dataset,
        "answer": cli.answer,
        "answer_dataset": cli.answer_dataset,
        "evaluate": cli.evaluate
    }
    fire.Fire(commands)
