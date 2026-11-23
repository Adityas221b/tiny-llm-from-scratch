import os
import tempfile

from tinyllm.tokenizer import BPETokenizer

CORPUS = ("First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
          "All:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to famish?\n") * 20


def _trained(vocab_size: int = 300) -> BPETokenizer:
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size)
    return tok


def test_vocab_size_and_compression():
    tok = _trained(300)
    assert tok.vocab_size == 300
    assert len(tok.encode(CORPUS)) < len(CORPUS.encode("utf-8")) / 2


def test_roundtrip_training_text():
    tok = _trained()
    assert tok.decode(tok.encode(CORPUS)) == CORPUS


def test_roundtrip_unseen_text():
    tok = _trained()
    for s in ["", "x", "  leading and trailing spaces  ", "tabs\tand\r\nnewlines\n\n\n",
              "Ünïcödé — émojis 🤖 and 中文", "numbers 12345 and punctuation!?;:'\""]:
        assert tok.decode(tok.encode(s)) == s, s


def test_save_load_identical():
    tok = _trained()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tok.json")
        tok.save(p)
        tok2 = BPETokenizer.load(p)
    text = "Speak, speak! What say you, citizens?"
    assert tok2.encode(text) == tok.encode(text)
    assert tok2.vocab == tok.vocab
