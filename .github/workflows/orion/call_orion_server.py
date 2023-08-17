import json
import os.path
import sys
import time
from datetime import timedelta

import requests


def parse_request_from_raw_txt_file():
    urls = []
    raw_path = sys.argv[1]
    with open(raw_path) as file:
        for line in file:
            line = line.strip()
            try:
                pos = line.index("(request) ")
            except ValueError:
                continue
            else:
                pieces = line[pos:].split(" ", maxsplit=2)
                assert len(pieces) in (2, 3)
                assert pieces[0] == "(request)"
                if len(pieces) == 3:
                    assert pieces[2] == "{}"
                print(pieces[1])
                urls.append(pieces[1])
    with open("urls.json", "w") as output:
        json.dump(urls, output, indent=1)


def main():
    json_path = os.path.join(os.path.dirname(__file__), "urls.json")
    with open(json_path) as file:
        urls = json.load(file)
    start = time.perf_counter_ns()
    for i, url in enumerate(urls):
        try:
            resp = requests.get(url)
            assert resp.status_code == 200, (url, resp.status_code)
            print(f"({i + 1}/{len(urls)}) {url} {type(resp.json())}")
        except requests.exceptions.ConnectionError as exc:
            print(f"[{i + 1}/{len(urls)}] REFUSED {url}", exc)
    diff = time.perf_counter_ns() - start
    print("Elapsed", timedelta(seconds=diff / 1e9))


if __name__ == "__main__":
    main()
