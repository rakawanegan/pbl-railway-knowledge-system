import os
from glob import glob
import re
import numpy as np
import pandas as pd
import markdown
from bs4 import BeautifulSoup
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from transformers import GPT2LMHeadModel, GPT2LMHeadModel


class RailwayKnowledgeSystemWithRinnaGPT2:
    def __init__(self, k: int) -> None:
        self.k = k
        knowledge = ''
        for p_file in sorted(glob('data/*.md')):
            knowledge += self._load_markdown_file(p_file)
        self.sentences = knowledge.split('\n')
        self.tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium")
        self.model = GPT2LMHeadModel.from_pretrained("rinna/japanese-gpt2-medium")
        if os.path.exists("./cache/rinna-railway-knowledge-index.faiss"):
            self.index = faiss.read_index("./cache/rinna-railway-knowledge-index.faiss")
        else:
            embeddings = self._embed_text(self.sentences)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
            os.makedirs("./cache", exist_ok=True)
            faiss.write_index(self.index, "./cache/rinna-railway-knowledge-index.faiss")

    def _load_markdown_file(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text

    def _embed_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        with torch.no_grad():
            embeddings = self.model.transformer.wte(inputs["input_ids"]).mean(dim=1)
        return embeddings.numpy()

    def get_text(self) -> str:
        return '\n'.join(self.sentences)

    def _check_loops(self, answer: str) -> bool:
        return len(answer.split(':')) > 10

    def get_basis(self, query: str) -> str:
        query_embedding = self._embed_text([query])
        _, indices = self.index.search(query_embedding, self.k) # first output is distance
        basis = " ".join([self.sentences[i] for i in indices[0] if i < len(self.sentences)])
        return basis

    def make_prompt(self, query: str) -> str:
        basis = self.get_basis(query)
        prompt = f"Context: {basis}\nQuery: {query}\nAnswer:"
        return prompt

    def generate_answer(self, prompt: str) -> str:
        self.count = 0
        answer = self._generate_answer(prompt)
        while self._check_loops(answer) and self.count < 10:
            print(f"Loop detected. Retry {self.count}")
            answer = self._generate_answer(prompt)
        return answer

    def _generate_answer(self, prompt: str) -> str:
        self.count += 1
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        outputs = self.model.generate(
            input_ids,
            num_return_sequences=1,
            max_length=1000,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            no_repeat_ngram_size=2,
        )
        answer = ''.join(self.tokenizer.decode(outputs[0], skip_special_tokens=True).split('nswer:')[1:])
        return answer

    def inference(self, query: str) -> str:
        basis = self.get_basis(query)
        prompt = self.make_prompt(query)
        answer = self.generate_answer(prompt)
        output = \
            f'''
            query:
            {query}

            answer:
            {answer}

            evidence:
            {basis}
            '''
        return output

class MakeRailwayKnowledgePromptWithTohokuBERT:
    def __init__(self, k: int) -> None:
        self.k = k
        knowledge = ''
        for p_file in sorted(glob('data/*.md')):
            knowledge += self._load_markdown_file(p_file)
        self.sentences = knowledge.split('\n')
        self.tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
        self.model = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
        if os.path.exists("./cache/railway-knowledge-index.faiss"):
            self.index = faiss.read_index("./cache/railway-knowledge-index.faiss")
        else:
            embeddings = self._embed_text(self.sentences)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
            os.makedirs("./cache", exist_ok=True)
            faiss.write_index(self.index, "./cache/railway-knowledge-index.faiss")

    def _load_markdown_file(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text

    def _embed_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.numpy()

    def get_text(self) -> str:
        return '\n'.join(self.sentences)

    def get_basis(self, query: str) -> None:
        query_embedding = self._embed_text([query])
        _, indices = self.index.search(query_embedding, self.k) # first output is distance
        basis = " ".join([self.sentences[i] for i in indices[0] if i < len(self.sentences)])
        return basis

    def make_prompt(self, query: str) -> None:
        basis = self.get_basis(query)
        prompt = f"Context: {basis}\nQuery: {query}\nAnswer:"
        return prompt

class MakeRailwayIncidentCasePromptWithTohokuBERT:
    def __init__(self, k: int) -> None:
        self.k = k
        knowledge = ''
        p_file = './data/RailwayIncidentCase.csv'
        self.sentences += self._load_csv_file(p_file)
        self.tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
        self.model = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
        if os.path.exists("./cache/railway-incident-index.faiss"):
            self.index = faiss.read_index("./cache/railway-incident-index.faiss")
        else:
            embeddings = self._embed_text(self.sentences)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
            os.makedirs("./cache", exist_ok=True)
            faiss.write_index(self.index, "./cache/railway-incident-index.faiss")

    def _load_csv_file(self, file_path: str) -> list[str]:
        df = pd.read_csv(file_path, sep='\t', encoding='utf-16')
        texts = list()
        for i, row in df.iterrows():
            row = str(dict(row))
            row = re.sub(r'\{|\}', '', row)  # strip {}
            row = re.sub(r"'", '"', row)  # replace ' with "
            texts.append(row)
        return texts

    def _embed_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.numpy()

    def get_text(self) -> str:
        return '\n'.join(self.sentences)

    def get_basis(self, query: str) -> None:
        query_embedding = self._embed_text([query])
        _, indices = self.index.search(query_embedding, self.k) # first output is distance
        basis = " ".join([self.sentences[i] for i in indices[0] if i < len(self.sentences)])
        return basis

    def make_prompt(self, query: str) -> None:
        basis = self.get_basis(query)
        prompt = f"Context: {basis}\nQuery: {query}\nAnswer:"
        return prompt
    

def set_up():
    RailwayKnowledgeSystemWithRinnaGPT2(k=1)
    MakeRailwayIncidentCasePromptWithTohokuBERT(k=1)
    MakeRailwayKnowledgePromptWithTohokuBERT(k=1)


if __name__ == '__main__':
    set_up()
    print("Setup done.")