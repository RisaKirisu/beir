import sys
import logging
import os
import pathlib
import argparse
import random
from time import time

base_dir = "/home/xiuyan.wu/beir"
sys.path.append(base_dir)


from transformers import AutoTokenizer
from beir.retrieval.models import VLLMEmbed
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout

dataset = "nfcorpus"

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
args = parser.parse_args()

model_name_or_path = args.model

#### Download nfcorpus.zip dataset and unzip the dataset
url = (
    f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
)
out_dir = os.path.join(pathlib.Path(os.path.abspath("")).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

model = DRES(
    VLLMEmbed(
        model_path=model_name_or_path,
        prompts={
            "query": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
            "document": "",
        },
        enforce_eager=False,
    ),
    batch_size=96,
)
retriever = EvaluateRetrieval(model, score_function="cos_sim")

#### Retrieve dense results (format of results is identical to qrels)
save_encodings_path = os.path.join(
    pathlib.Path(os.path.abspath("")).parent.absolute(), model_name_or_path, "encodings"
)
if not os.path.exists(save_encodings_path):
    os.makedirs(save_encodings_path)

start_time = time()
results = retriever.encode_and_retrieve(
    corpus, queries, encode_output_path=save_encodings_path
)
end_time = time()
logging.info(f"Time taken to encode & retrieve: {end_time - start_time:.2f} seconds")

#### Evaluate your retrieval using NDCG@k, MAP@K ...

logging.info(f"Retriever evaluation for k in: {retriever.k_values}")
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

### If you want to save your results and runfile (useful for reranking)
results_dir = os.path.join(
    pathlib.Path(os.path.abspath("")).parent.absolute(), "results", model_name_or_path
)
os.makedirs(results_dir, exist_ok=True)

#### Save the evaluation runfile & results
util.save_runfile(os.path.join(results_dir, f"{dataset}.run.trec"), results)
util.save_results(
    os.path.join(results_dir, f"{dataset}.json"), ndcg, _map, recall, precision, mrr
)
