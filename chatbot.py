import re
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM
)

from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Text preprocessing
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(text):
    return clean_text(text)

# -------------------------------
# Load dataset
# -------------------------------
def load_dialogue_data(path="dialogues.txt"):
    contexts, responses = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                contexts.append(parts[0])
                responses.append(parts[1])

    return pd.DataFrame({
        "context": contexts,
        "response": responses
    })

df = load_dialogue_data()
df["processed_context"] = df["context"].apply(preprocess)

# -------------------------------
# DistilBERT (Semantic Model)
# -------------------------------
bert_tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)
bert_model = AutoModel.from_pretrained(
    "distilbert-base-uncased"
)

def encode_bert(texts):
    inputs = bert_tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = bert_model(**inputs)

    return outputs.last_hidden_state[:, 0, :].numpy()

context_embeddings = encode_bert(
    df["processed_context"].tolist()
)

# -------------------------------
# V1: Retrieval-based chatbot
# -------------------------------
def chatbot_semantic(user_input, threshold=0.6):
    user_input = preprocess(user_input)
    user_emb = encode_bert([user_input])

    similarities = cosine_similarity(
        user_emb,
        context_embeddings
    )[0]

    best_idx = similarities.argmax()
    best_score = similarities[best_idx]

    if best_score < threshold:
        return "I'm not sure about that. Can you rephrase?"

    return df.iloc[best_idx]["response"]

# -------------------------------
# DialoGPT (Generative Model)
# -------------------------------
dg_tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/DialoGPT-medium"
)
dg_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/DialoGPT-medium"
)

def chatbot_generative(user_input, max_retries=2):
    prompt = (
        "The following is a friendly conversation.\n"
        f"User: {user_input}\n"
        "Bot:"
    )

    inputs = dg_tokenizer(
        prompt,
        return_tensors="pt",
        padding=True
    )

    for attempt in range(max_retries + 1):
        output_ids = dg_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            do_sample=True,
            temperature=0.7 + 0.3 * attempt,
            top_k=50 + 50 * attempt,
            top_p=0.9,
            pad_token_id=dg_tokenizer.eos_token_id
        )

        decoded = dg_tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        if "Bot:" in decoded:
            decoded = decoded.split("Bot:")[-1]

        decoded = decoded.strip()

        if decoded:
            return decoded

    return "[No response generated]"

# -------------------------------
# V2: Hybrid chatbot
# -------------------------------
def chatbot_hybrid(user_input):
    if len(user_input.strip().split()) <= 2:
        return chatbot_semantic(user_input)

    return chatbot_generative(user_input)
