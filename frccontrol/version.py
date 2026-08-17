import subprocess
from datetime import date


def get_version():
    year = date.today().year  # noqa: DTZ011
    proc = subprocess.run(
        [
            "git",
            "rev-list",
            "--count",
            # Includes previous year's commits in case one was merged after the
            # year incremented. Otherwise, the version wouldn't increment.
            '--after="main@{' + str(year - 1) + '-01-01}"',
            "main",
        ],
        check=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
    )
    # If there is no main branch, the commit count defaults to 0
    if proc.returncode:
        commit_count = "0"
    else:
        commit_count = proc.stdout

    # Version number: <year>.<# commits on main>
    return f"{year}.{commit_count.strip()}"
