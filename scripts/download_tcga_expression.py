"""Resume the selected public expression download using four verified byte ranges.

Uses curl for platform TLS support, validates range/length and the full gzip CRC.
Raw chunks remain local/ignored, and no existing final file is overwritten.
"""

import argparse
import concurrent.futures
import gzip
import re
import shutil
import subprocess
from pathlib import Path

URL = "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.star_tpm.tsv.gz"
SIZE = 342300776
ETAG = '"a5b27357bbad380d8455188422ff6179-41"'


def download_range(directory, start, end):
    output = directory / f"expression.range-{start}-{end}"
    headers = Path(str(output) + ".headers")
    command = [
        "curl", "--fail", "--silent", "--show-error", "--location",
        "--connect-timeout", "20", "--max-time", "900",
        "--header", f"If-Match: {ETAG}", "--range", f"{start}-{end}",
        "--dump-header", str(headers),
    ]
    if output.exists():
        if output.stat().st_size != end - start + 1:
            raise ValueError(f"Incomplete chunk needs explicit recovery: {output}")
        # Recheck the same pinned object/range without overwriting saved bytes.
        # The final gzip CRC validates the assembled payload, including reused chunks.
        command += ["--head", "--output", "/dev/null", URL]
    else:
        command += ["--output", str(output), URL]
    subprocess.run(command, check=True)
    response = headers.read_text().lower()
    expected = f"content-range: bytes {start}-{end}/{SIZE}"
    if expected not in response or not re.search(r"http/\S+ 206", response):
        raise ValueError("Server did not honor the requested byte range")
    if output.stat().st_size != end - start + 1:
        raise ValueError("Downloaded range has incorrect length")
    print(f"Verified bytes {start}-{end}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.raw_dir
    prefix = directory / "TCGA-BRCA.star_tpm.tsv.gz.part"
    final = directory / "TCGA-BRCA.star_tpm.tsv.gz"
    assembled = directory / "TCGA-BRCA.star_tpm.tsv.gz.assembled"
    if final.exists() or assembled.exists():
        raise FileExistsError("Refusing to replace an existing final/assembled file")
    start = prefix.stat().st_size if prefix.exists() else 0
    if not 0 <= start < SIZE:
        raise ValueError("Unexpected existing prefix length")
    width = (SIZE - start + 3) // 4
    ranges = [(a, min(a + width - 1, SIZE - 1)) for a in range(start, SIZE, width)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download_range, directory, a, b) for a, b in ranges]
        chunks = [f.result() for f in futures]
    with assembled.open("xb") as destination:
        for chunk in ([prefix] if start else []) + chunks:
            with chunk.open("rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    if assembled.stat().st_size != SIZE:
        raise ValueError("Full expression file has incorrect length")
    with gzip.open(assembled, "rb") as source:
        while source.read(1024 * 1024):
            pass
    assembled.rename(final)
    print(f"Complete and gzip-verified: {final}", flush=True)


if __name__ == "__main__":
    main()
