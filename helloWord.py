import nltk
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.model_selection import train_test_split
from collections import defaultdict
import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer
import os

import nltk
from nltk.corpus import treebank


tagged_sentences = nltk.corpus.treebank.tagged_sents()
print("Number of samples:",len(tagged_sentences))

sentences,sentence_tags = [],[]
for tagged_sentence in tagged_sentences:
    sentence,tags = zip(*tagged_sentence)
    sentences.append([word for word in sentence])
    sentence_tags.append([tag for tag in tags])

model_name = "QCRI/bert-base-multilingual-cased-pos-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

train_sentences, test_sentences, train_tags, test_tags = train_test_split(
    sentences,
    sentence_tags,
    test_size = 0.3
)

valid_sentences, test_sentences, valid_tags, test_tags = train_test_split(
    test_sentences,
    test_tags,
    test_size = 0.5
)


label2id = defaultdict(int, model.config.label2id)
id2label = {id: label for label, id in label2id.items()}

# Giả định MAX_LEN được tính toán trước
MAX_LEN = max(len(sentence) for sentence in train_sentences)

class PosTagging_Dataset(Dataset):
    def __init__(self, sentences: list[list[str]], tags: list[list[str]], tokenizer, label2id, max_len=MAX_LEN):
        super().__init__()
        self.sentences = sentences
        self.tags = tags
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # Lấy câu và nhãn tại chỉ mục idx
        input_token = self.sentences[idx]  # Ví dụ: ['tôi', 'đi', 'học']
        label_token = self.tags[idx]  # Ví dụ: ['DT', 'V', 'DT'] (Đại từ, Động từ, Danh từ)

        # 1. Mã hóa token đầu vào thành ID số
        input_token_ids = self.tokenizer.convert_tokens_to_ids(input_token)

        # 2. Tạo Attention Mask
        attention_mask = [1] * len(input_token_ids)

        # 3. Chuyển đổi nhãn từ chuỗi (string) sang ID số
        # labels = [self.label2id[token] for token in label_token] # Đây là dòng trong ảnh, tôi chỉnh lại biến
        labels = [self.label2id[token] for token in label_token]

        # 4. Đệm (Padding) và Cắt (Truncation) để chuẩn hóa độ dài
        return {
            "input_ids": self.pad_and_truncate(
                input_token_ids,
                pad_id=self.tokenizer.pad_token_id
            ),
            "labels": self.pad_and_truncate(
                labels,
                pad_id=self.label2id["O"]  # Dùng nhãn "O" (Outside) cho phần đệm
            ),
            "attention_mask": self.pad_and_truncate(
                attention_mask,
                pad_id=0
            ),
        }

    def pad_and_truncate(self, inputs: list[int], pad_id: int):
        # Nếu độ dài nhỏ hơn MAX_LEN -> Đệm thêm pad_id
        if len(inputs) < self.max_len:
            padded_inputs = inputs + [pad_id] * (self.max_len - len(inputs))
        # Nếu độ dài lớn hơn MAX_LEN -> Cắt bớt
        else:
            padded_inputs = inputs[:self.max_len]

        return torch.as_tensor(padded_inputs)

train_dataset = PosTagging_Dataset(train_sentences, train_tags, tokenizer, label2id, MAX_LEN)
val_dataset = PosTagging_Dataset(valid_sentences, valid_tags, tokenizer, label2id, MAX_LEN)
test_dataset = PosTagging_Dataset(test_sentences, test_tags, tokenizer, label2id, MAX_LEN) # Tùy chọn

ignore_label = len(label2id)
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Tạo mặt nạ để chỉ giữ lại các nhãn thực tế (không phải padding)
    mask = labels != ignore_label

    # Chuyển đổi logits thành ID dự đoán (lấy chỉ mục có giá trị lớn nhất)
    predictions = np.argmax(predictions, axis=-1)

    return accuracy.compute(
        predictions=predictions[mask],
        references=labels[mask]
    )

training_args = TrainingArguments(
    output_dir="out_dir",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Thiết lập để tắt WandB (theo snippet của bạn)
os.environ["WANDB_DISABLED"] = "true"

print("Bắt đầu huấn luyện mô hình...")
trainer.train()
print("Huấn luyện hoàn tất.")