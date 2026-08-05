from src.cli_interface import Cli
import fire


if __name__ == '__main__':
    cli = Cli()
    commands: dict = {
        "index": cli.index
    }
    fire.Fire(commands)
