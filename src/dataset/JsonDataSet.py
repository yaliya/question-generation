import json
from torch.utils.data import Dataset

class JsonDataSet(Dataset):
    def __init__(self, tokenizer, max_input_length=512, max_target_length=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        with open('data/questions.json', 'r') as f:
            data = json.load(f)

            # Flatten the data to create a list of (question, answer) pairs
            for item in data:
                answer = item['answer']
                for question in item['questions']:
                    self.data.append({'question': question, 'answer': answer})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = f"generate question about: {item['answer']}"
        target_text = item['question']

        # Tokenize inputs and targets
        input_ids = self.tokenizer.encode(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_input_length
        )

        target_ids = self.tokenizer.encode(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_target_length
        )

        return {
            'input_ids': input_ids,
            'labels': target_ids
        }