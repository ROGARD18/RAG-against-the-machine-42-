from src.cli_interface import Cli
import fire


if __name__ == '__main__':
    cli = Cli()
    commands: dict = {
        "index": cli.index,
        "search": cli.search,
        "search_dataset": cli.search_dataset,
        "answer": cli.answer,
        "answer_dataset": cli.answer_dataset,
        "evaluate": cli.evaluate
    }
    fire.Fire(commands)
