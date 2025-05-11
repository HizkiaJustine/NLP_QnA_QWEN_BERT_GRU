import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data_split(dataset_name="lib3m/lib3m_qa_dataset_v1", split="train", lang="en", test_size=0.2, row=100000):
    raw = load_dataset("lib3m/lib3m_qa_dataset_v1", split=split)
    dataset = raw.filter(lambda x: x['language'] == 'en')[:row]

    # Split train/validation
    dataset = dataset.train_test_split(test_size=0.2)
    return dataset #  dataset['train'] & dataset['test']

  
def load_data(
    dataset_name: str = "lib3m/lib3m_qa_dataset_v1",
    split: str = "train",
    lang: str = "en",
    row:int = 100000
) -> pd.DataFrame:
    ds = load_dataset(dataset_name, split=split)
    df = ds.to_pandas()
    df = df[df.language == lang].reset_index(drop=True)[:row]
    return df


def split_dataframe(
    df,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

class QADataset(Dataset):
    def __init__(
        self,
        dataframe,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = row['question']
        content = row['content']
        answer = row['answer']

        # Generative QA
        text = f"<question> {question} <context> {content} <answer>"
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        labels = self.tokenizer(
            answer,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        ).input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            'input_ids': tokenized.input_ids.squeeze(),
            'attention_mask': tokenized.attention_mask.squeeze(),
            'labels': labels.squeeze()
        }