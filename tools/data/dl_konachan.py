import click
from pathlib import Path

from cheesechaser.datapool import KonachanWebpDataPool

MAX_ID = 391069


@click.command()
@click.option("--output", "-o", type=str, required=True)
@click.option("--limit", type=int, required=True)
def main(
    output: str,
    limit: int,
):
    output_dir = Path(output) / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    pool = KonachanWebpDataPool()

    pool.batch_download_to_directory(
        resource_ids=range(max(1, MAX_ID - limit), MAX_ID),
        dst_dir=output_dir.as_posix(),
    )


if __name__ == "__main__":
    main()
