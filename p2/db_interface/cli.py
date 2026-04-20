import argparse

from .database import engine
from .models import Base


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DB interface CLI")
    parser.add_argument("command", choices=["init-db"], help="Command to execute")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init-db":
        init_db()


if __name__ == "__main__":
    main()
