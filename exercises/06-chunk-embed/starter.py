#!/usr/bin/env python3
"""Module 6 — Chunk & Embed: text becomes vectors.  YOUR WORK GOES HERE.

Four TODOs. Two about cutting text up, two about turning it into numbers.

    python exercises/06-chunk-embed/starter.py       # run yours
    python exercises/06-chunk-embed/verify.py        # check it

Needs: sentence-transformers, numpy.
Needs the corpus: python data/fetch.py --only msmarco --small
"""
from math import sqrt
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "exercises"))

from common import rule, scale_parser   # noqa: E402

# The pair the whole module turns on: no shared words, same meaning.
CAT = "the cat sat on the mat"
FELINE = "a feline rested upon soft carpet"
UNRELATED = "quarterly revenue exceeded analyst expectations"


# ==========================================================================
# TODO 1 — fixed-size chunking, the way every tutorial does it
# ==========================================================================
#
# Cut `text` into pieces of `size` characters, where each piece starts
# `size - overlap` characters after the previous one. Drop empty pieces.
#
# Return a list of dicts: {"start": int, "end": int, "text": str}
#
# This is exactly what the Part 0 pipeline's split() does. Keeping the offsets
# is the only addition, and it is what lets TODO 2 show you the damage.

def chunk_fixed(text: str, size: int = 500, overlap: int = 100) -> list[dict]:
    chunks = []
    # TODO
    ...
    step_size=size-overlap
    for i in range(0, len(text),step_size):
        text_dict={}
        text_dict["start"]=i
        text_dict["end"]=i+size
        text_dict["text"]=text[i:min(i+size,len(text))]
        chunks.append(text_dict)

    return chunks


# ==========================================================================
# TODO 2 — chunk at sentence boundaries instead
# ==========================================================================
#
# Split `text` into sentences, then pack whole sentences into chunks of at
# most `max_chars`. Never cut a sentence in half.
#
# A sentence ends at '.', '!' or '?' followed by a space or a newline. That
# rule is wrong for "Dr. Smith" and for "3.14", and being wrong in a knowable
# way is better than being wrong in an unknowable one — say what your rule is
# and let it be judged.
#
# Return the same shape as TODO 1.

def chunk_sentences(text: str, max_chars: int = 500) -> list[dict]:
    chunks = []
    # TODO
    ...
    sentence=[]
    records=[]
   
    sentence_start=0
    for character in range(len(text)):
        if text[character] == "." or text[character] == "!" or text[character] == "?":
            sentence.append(text[character])     
            if character+1==len(text) or text[character+1]=="\n" or text[character+1]==" " :
                textDict={}
                textDict["start"]=sentence_start
                textDict["end"]=character+1
                textDict["text"]= "".join(sentence)
                records.append(textDict)
                sentence=[]
                sentence_start=character+1               
        else:
            sentence.append(text[character])
    if len(sentence)!=0:
        textDict2={}
        textDict2["start"]=sentence_start
        textDict2["end"]=character+1
        textDict2["text"]= "".join(sentence)
        records.append(textDict2)
    current_chunk=""
    current_start=0
    current_end=0
    for record in records:
        incoming_sentence=record["text"]
        if len(current_chunk)==0:
            current_start=record["start"]
        if len(current_chunk)+len(incoming_sentence)<=max_chars:
            current_chunk=current_chunk + incoming_sentence
            current_end=record["end"]
        else:
            textDict={}
            textDict["start"]=current_start
            textDict["end"]=current_end
            textDict["text"]=current_chunk
            current_chunk=incoming_sentence
            current_start=record["start"]
            current_end=record["end"]
            chunks.append(textDict)
    if len(current_chunk)!=0:
        textDict={}
        textDict["start"]=current_start
        textDict["end"]=current_end
        textDict["text"]=current_chunk
        chunks.append(textDict)
        


    return chunks


# ==========================================================================
# TODO 3 — cosine similarity, by hand
# ==========================================================================
#
# Do not call a library for this. It is a dot product and two lengths:
#
#     cosine(a, b) = sum(a[i] * b[i]) / (sqrt(sum(a[i]^2)) * sqrt(sum(b[i]^2)))
#
# The result runs from -1 to 1. Two vectors pointing the same way score 1.
# Write it with plain arithmetic so you know exactly what "similarity" is.

def cosine(a, b) -> float:
    # TODO — about five lines
    ...
    dotProduct=0
    for i in range(len(a)):
       dotProduct+=a[i]*b[i]
    distA=0 
    distB=0
    for i in range(len(a)):
        distA+=a[i]**2
        distB+=b[i]**2
    magnitude=sqrt(distA) * sqrt(distB)
    cosine_similarity=dotProduct/magnitude
    return cosine_similarity




# ==========================================================================
# TODO 4 — find the chunks that straddle a boundary
# ==========================================================================
#
# Given the chunks from TODO 1 and the original text, return the ones whose
# `start` lands in the middle of a word — that is, where text[start-1] and
# text[start] are both non-space characters.
#
# Return a list of dicts: {"start": int, "broken_word": str} where broken_word
# is the word that got cut, reassembled from the text either side of the cut.

def straddling_chunks(text: str, chunks: list[dict]) -> list[dict]:
    # TODO
    ...
    straddling_chunks=[]
    for chunkDictionary in chunks:
        if chunkDictionary["start"]!=0:
            start=chunkDictionary["start"]
            if text[start-1] !=" " and  text[start] !=" ":
                prefix=""
                index=start-1
                while text[index]!=" ":
                    prefix=text[index]+prefix
                    index-=1
                suffix=""
                suffix_index=start
                while suffix_index< len(text) and text[suffix_index]!=" ":
                    suffix+=text[suffix_index]
                    suffix_index+=1
                word=prefix+suffix
                straddlingDictionary={}
                straddlingDictionary["start"]=start
                straddlingDictionary["broken_word"]=word
                straddling_chunks.append(straddlingDictionary)
    return straddling_chunks
                




# --------------------------------------------------------------------------
# plumbing — written for you
# --------------------------------------------------------------------------

def load_model(name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def sample_text(n_passages=40) -> str:
    from common import iter_passages
    return "\n".join(t for _, t in iter_passages(n_passages))


def main() -> None:
    args = scale_parser(__doc__, default="tiny").parse_args()
    text = sample_text()

    rule("1 · fixed-size chunks")
    fixed = chunk_fixed(text)
    print(f"  {len(fixed) if fixed else 0} chunks of at most 500 characters")

    rule("2 · sentence-boundary chunks")
    sentences = chunk_sentences(text)
    print(f"  {len(sentences) if sentences else 0} chunks, no sentence cut in half")

    rule("3 · what fixed-size chunking broke")
    broken = straddling_chunks(text, fixed or [])
    print(f"  {len(broken) if broken else 0} chunk boundaries landed inside a word")
    for b in (broken or [])[:5]:
        print(f"    at {b['start']}: {b['broken_word']!r}")

    rule("4 · similarity without shared words")
    model = load_model()
    vectors = model.encode([CAT, FELINE, UNRELATED])
    print(f"  each sentence became {len(vectors[0])} numbers")
    print(f"  cat / feline    {cosine(vectors[0], vectors[1])}")
    print(f"  cat / unrelated {cosine(vectors[0], vectors[2])}")
    print(f"  shared words between the first two: "
          f"{set(CAT.split()) & set(FELINE.split())}")


if __name__ == "__main__":
    main()
