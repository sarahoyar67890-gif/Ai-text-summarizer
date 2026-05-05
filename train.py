import pandas as pd
from transformers import pipeline

# LOAD DATA
df = pd.read_csv("test.csv")

# DROP UNUSED COLUMN
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

# LOAD MODEL
model = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    framework="pt"
)

# TEST SAMPLE
sample = df['article'].iloc[0]
print("ORIGINAL:\n", sample)

summary = model(sample[:1000], max_length=60, min_length=20, do_sample=False)
print("\nSUMMARY:\n", summary[0]['summary_text'])

# =========================
# SAVE MODEL (CORRECT WAY)
# =========================

save_path = "saved_model"

model.model.save_pretrained(save_path)
model.tokenizer.save_pretrained(save_path)

print("\nModel saved successfully in HuggingFace format!")