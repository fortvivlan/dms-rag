# dms-rag

FAISS vectorstore for DMS

## Installation

Clone the repository. Create a virtual environment and run `pip install .` in it. **Install FAISS separately** as `conda install faiss-gpu` or `pip install faiss-cpu`. If you have two folders in the repo, remove one of them or installation won't work.

## Usage

To create a vectorbase from scratch, place a folder with your codex texts (in .txt, encoding='utf-8') in the repo. The texts must not contain any general headers, chapters etc, the parser splits them if there is a word 'статья'. Run `python codex.py [path to codex folder] [path to folder with vectorbase]` (replace [parts] with your paths)

