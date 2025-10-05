import io
import zipfile
import requests
import frontmatter
from minsearch import Index, VectorSearch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import numpy as np

def read_repo_data(repo_owner, repo_name):
   """
   Download and parse all markdown files from a GitHub repository.
   
   Args:
       repo_owner: GitHub username or organization
       repo_name: Repository name
   
   Returns:
       List of dictionaries containing file content and metadata
   """
   prefix = 'https://codeload.github.com'
   url = f'{prefix}/{repo_owner}/{repo_name}/zip/refs/heads/main'
   resp = requests.get(url)
   
   if resp.status_code != 200:
       raise Exception(f"Failed to download repository: {resp.status_code}")

   repository_data = []
   zf = zipfile.ZipFile(io.BytesIO(resp.content))
   
   for file_info in zf.infolist():
       filename = file_info.filename
       filename_lower = filename.lower()

       if not (filename_lower.endswith('.md')
           or filename_lower.endswith('.mdx')):
           continue
   
       try:
           with zf.open(file_info) as f_in:
               content = f_in.read().decode('utf-8', errors='ignore')
               post = frontmatter.loads(content)
               data = post.to_dict()
               data['filename'] = filename
               repository_data.append(data)
       except Exception as e:
           print(f"Error processing {filename}: {e}")
           continue
   
   zf.close()
   return repository_data  

def sliding_window(seq, size, step):
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []
    for i in range(0, n, step):
        chunk = seq[i:i+size]
        result.append({'start': i, 'chunk': chunk})
        if i + size >= n:
            break

    return result



#Read Evidently docs

evidently_docs = read_repo_data('evidentlyai', 'docs')


#Sliding window
evidently_chunks = []

for doc in evidently_docs:
    doc_copy = doc.copy()
    doc_content = doc_copy.pop('content')
    chunks = sliding_window(doc_content, 2000, 1000)
    for chunk in chunks:
        chunk.update(doc_copy)
    evidently_chunks.extend(chunks)

index = Index(
    text_fields=["chunk", "title", "description", "filename"],
    keyword_fields=[]
)
index.fit(evidently_chunks)

dtc_faq = read_repo_data('DataTalksClub', 'faq')
de_dtc_faq = [d for d in dtc_faq if 'data-engineering' in d['filename']]
faq_index = Index(
    text_fields=["question", "content"],
    keyword_fields=[]
)
faq_index.fit(de_dtc_faq)




#VECTOR
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

faq_embeddings = []
for d in tqdm(de_dtc_faq):
    text = d['question'] + ' ' + d['content']
    v = embedding_model.encode(text)
    faq_embeddings.append(v)

faq_embeddings = np.array(faq_embeddings)

faq_vindex = VectorSearch()
faq_vindex.fit(faq_embeddings, de_dtc_faq)

query = 'Can I join the course now?'
q = embedding_model.encode(query)
#results = faq_vindex.search(q)


#HYBRID SEARCH
def text_search(query):
    return faq_index.search(query, num_results=5)

def vector_search(query):
    q = embedding_model.encode(query)
    return faq_vindex.search(q, num_results=5)

def hybrid_search(query):
    text_results = text_search(query)
    vector_results = vector_search(query)
    
    # Combine and deduplicate results
    seen_ids = set()
    combined_results = []

    for result in text_results + vector_results:
        if result['filename'] not in seen_ids:
            seen_ids.add(result['filename'])
            combined_results.append(result)
    
    return combined_results


results = hybrid_search(query)
print(results)