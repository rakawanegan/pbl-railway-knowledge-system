from glob import glob
import markdown
from bs4 import BeautifulSoup
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from transformers import GPT2LMHeadModel, GPT2LMHeadModel


class RailwayKnowledgeSystem:
    def __init__(self, k) -> None:
        self.k = k
        knowledge = ''
        for p_file in sorted(glob('data/*.md')):
            knowledge += self._load_markdown_file(p_file)
        self.sentences = knowledge.split('\n')
        self.tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium")
        self.model = GPT2LMHeadModel.from_pretrained("rinna/japanese-gpt2-medium")
        embeddings = self._embed_text(self.sentences)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def _load_markdown_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text

    def _embed_text(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = self.model.transformer.wte(inputs["input_ids"]).mean(dim=1)
        return embeddings.numpy()

    def _check_loops(self, answer) -> None:
        return False

    def get_basis(self, query) -> None:
        query_embedding = self._embed_text([query])
        _, indices = self.index.search(query_embedding, self.k) # first output is distance
        basis = " ".join([self.sentences[i] for i in indices[0] if i < len(self.sentences)])
        return basis

    def make_prompt(self, query) -> None:
        basis = self.get_basis(query)
        prompt = f"Context: {basis}\nQ: {query}\nA:"
        return prompt

    def generate_answer(self, prompt) -> None:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        outputs = self.model.generate(
            input_ids,
            max_length=1000,
            num_return_sequences=1,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id
        )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
        if self._check_loops(answer):
            return self.generate_answer(prompt)
        return answer

    def inference(self, query) -> None:
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
