"""Download a corpus, train the BPE tokenizer on its train split, and encode both splits.

Corpora:
  shakespeare  TinyShakespeare (~1.1 MB)                          -> data/
  jeeves       P.G. Wodehouse's Jeeves & Wooster stories (~1.4 MB) -> data/jeeves/
               My Man Jeeves (1919), The Inimitable Jeeves (1923), Carry On, Jeeves (1925),
               Very Good, Jeeves (1930) -- public domain in the USA, from Project Gutenberg.

Outputs (in the data dir):
  input.txt          cleaned corpus
  tokenizer.json     learned BPE merges
  train.bin/val.bin  uint16 token ids

    python prepare_data.py                      # shakespeare
    python prepare_data.py --corpus jeeves      # Bertie Wooster
"""
import argparse
import os
import re
import time
import urllib.request

import numpy as np

from tinyllm.tokenizer import BPETokenizer

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# Project Gutenberg's own mirror (gutenberg.pglaf.org); the www.gutenberg.org host works too.
GUTENBERG_MIRRORS = ["https://gutenberg.pglaf.org", "https://www.gutenberg.org/files"]

# ebook id -> (title, first words of the first story, phrases that mark the end of the stories)
JEEVES_BOOKS = {
    8164:  ("My Man Jeeves", "Jeeves—my man, you know—is really", []),
    59254: ("The Inimitable Jeeves", "CHAPTER I\n", ["+------"]),
    65974: ("Carry On, Jeeves", "Now, touching this business of old Jeeves", []),
    77923: ("Very Good, Jeeves", "It was the morning of the day on which I was slated", ["BOOKS BY\n"]),
}


def gutenberg_path(book_id: int) -> str:
    s = str(book_id)
    return "/".join(s[:-1]) + f"/{s}/{s}-0.txt"            # e.g. 59254 -> 5/9/2/5/59254/59254-0.txt


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "tiny-llm-from-scratch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8-sig")


def clean_book(raw: str, start: str, ends: list[str]) -> str:
    text = raw.replace("\r\n", "\n")
    text = text[text.index(start):]
    stop = [text.find(e) for e in ends + ["*** END OF THE PROJECT GUTENBERG"]]
    text = text[: min(i for i in stop if i > 0)]
    text = re.sub(r"\[Illustration[^\]]*\]", "", text)
    text = text.replace("_", "")                                        # _italics_ markup
    for a, b in {"‘": "'", "’": "'", "“": '"', "”": '"',
                 "—": "--", "–": "-", "…": "...", " ": " "}.items():
        text = text.replace(a, b)
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if re.fullmatch(r"[*\s]+", s):          # "*   *   *" scene breaks -> blank line
            s = ""
        lines.append(s if line.startswith(" " * 8) else line.rstrip())   # de-indent centred headings
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def get_jeeves() -> str:
    parts = []
    for book_id, (title, start, ends) in JEEVES_BOOKS.items():
        for mirror in GUTENBERG_MIRRORS:
            try:
                raw = fetch(f"{mirror}/{gutenberg_path(book_id)}")
                break
            except Exception as e:  # noqa: BLE001
                print(f"  {mirror}: {e}")
        else:
            raise RuntimeError(f"could not download Gutenberg ebook {book_id}")
        body = clean_book(raw, start, ends)
        print(f"  {title} (#{book_id}): {len(body):,} chars")
        parts.append(body)
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="shakespeare", choices=["shakespeare", "jeeves"])
    ap.add_argument("--data_dir", default=None, help="default: data/ (shakespeare) or data/jeeves/")
    ap.add_argument("--vocab_size", type=int, default=512)
    ap.add_argument("--val_frac", type=float, default=0.1)
    args = ap.parse_args()
    data_dir = args.data_dir or ("data" if args.corpus == "shakespeare" else os.path.join("data", args.corpus))

    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "input.txt")
    if not os.path.exists(path):
        print(f"downloading {args.corpus}")
        text = fetch(SHAKESPEARE_URL) if args.corpus == "shakespeare" else get_jeeves()
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    print(f"corpus: {len(text):,} characters")

    # Contiguous split: first 90% train, last 10% val. The tokenizer only sees train text.
    # (For jeeves this means the val set is the second half of "Very Good, Jeeves": unseen stories.)
    n = int(len(text) * (1 - args.val_frac))
    train_text, val_text = text[:n], text[n:]

    t0 = time.time()
    tok = BPETokenizer()
    tok.train(train_text, args.vocab_size, verbose=True)
    print(f"trained BPE (vocab {tok.vocab_size}) in {time.time() - t0:.1f}s")
    tok.save(os.path.join(data_dir, "tokenizer.json"))

    for name, split in [("train", train_text), ("val", val_text)]:
        ids = np.array(tok.encode(split), dtype=np.uint16)
        ids.tofile(os.path.join(data_dir, f"{name}.bin"))
        print(f"{name}: {len(split):,} chars -> {len(ids):,} tokens ({len(split) / len(ids):.2f} chars/token)")


if __name__ == "__main__":
    main()
