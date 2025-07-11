### Скрипт для создания векторной базы: требует наличия папки codex, 
### в которой будут содержаться все нужные кодексы и инструкции в формате .txt, 
### причем обязательно внутри кодексов не должно быть заголовка самого кодекса, глав
### и разделов, потому что иначе технические фрагменты попадут в содержимое статей. 
### Убедитесь, что ваши файлы .txt имеют кодировку utf-8. 

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from sklearn.preprocessing import normalize
import faiss
import os, re, csv, argparse

def parse_texts(path):
    """Function to parse txt files with codices"""
    data = []
    for doc in os.listdir(path):
        if not doc.endswith('txt'):
            continue
        dct = {}
        filename = os.path.splitext(doc)[0]
        with open(os.path.join(path, doc)) as file:
            article = ''
            text = ''
            for line in file:
                # we don't want obsolete articles and useless info
                if re.search(r'утратил. силу', line.lower()) or re.search(r'введен. федеральным законом', line.lower()):
                    continue 
                if 'в ред. Федерального закона' in line.lower():
                    continue
                if 'абзац исключен' in line.lower():
                    continue
                # we split codex text into articles by the keyword
                if line.startswith('Статья '):
                    if article:
                        if len(article) > 100:
                            article = article[:100] # truncate article name as it can be very long
                        data.append({"filename": filename, "article": article, "text": text.strip()})
                    article = line.strip()
                    text = ''
                else:
                    # as of now - we don't split an article into points and remove numbers
                    # so that they don't interfere
                    if re.match(r"\d+\.(\d+\.)?", line):
                        line = re.sub(r"^\d+\.(\d+\.)?", '', line)
                    text += (' ' + line.strip('\n'))
    with open('codex.csv', 'w') as file:
        csv_writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        # create csv with two columns
        csv_writer.writerow(['text', 'source'])
        for item in data:
            csv_writer.writerow([item["text"], f"{item['filename']}: {item['article']}"])

def create_vectorbase(path):
    """The function to create the vectorbase from codex.csv created above"""
    loader = CSVLoader(file_path="codex.csv", content_columns=["text"], metadata_columns=["source"], source_column="source")
    documents = loader.load()
    # we clean the docs as CSVLoader creates unnecessary column names in text
    for doc in documents:
        doc.page_content = doc.page_content[len('text: '):]
    # We split articles into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    # we create embeddings based on a sbert model - not finetuned as of now
    embeddings = HuggingFaceEmbeddings(model_name="ai-forever/sbert_large_nlu_ru")
    texts = [doc.page_content for doc in splits]

    # we normalise embeddings as FAISS doesn't have a native cosine similarity implementation (lame)
    raw_embeddings = embeddings.embed_documents(texts)  
    normalized_embeddings = normalize(raw_embeddings, norm='l2')  
    dim = normalized_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(normalized_embeddings)
    index_to_docstore_id = {i: str(i) for i in range(len(splits))}
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(splits)})
    # finally we create the vectorbase object - this uses GPU if we have it
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    # save file to folder
    vectorstore.save_local(path)

def main():
    parser = argparse.ArgumentParser(description='Create vectorbase: we need a path for codex files and a path for the base')
    parser.add_argument('codex_path', type=str, help='Path to the input codex folder')
    parser.add_argument('result_path', type=str, help='Path for the output results file')
    args = parser.parse_args()
    print('Parsing codex base...')
    parse_texts(args.codex_path)
    print('Creating vectorbase...')
    create_vectorbase(args.result_path)
    print('Done')

if __name__ == '__main__':
    main()
