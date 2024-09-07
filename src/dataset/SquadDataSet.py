from datasets import load_dataset
from torch.utils.data import Dataset

class SquadDataSet(Dataset):
    def __init__(self, split, tokenizer, max_input_length=512, max_target_length=128):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.data = load_dataset("derek-thomas/squad-v1.1-t5-question-generation")[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = f"generate question about: {item['context'].replace('gq: ', '')}"
        target_text = item['questions']

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