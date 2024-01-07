from transformers import AutoTokenizer
import tensorflow as tf
from transformers import TFAutoModelForCausalLM
import numpy as np
import spacy
from pathlib import Path
import config
import gradio as gr

sp = spacy.load("en_core_web_sm")  # Use en_core_web_lg for more vocab

tokenizer_path = Path(__file__).parent / config.INFERENCE_PATHS.get("tokenizer_path")
model_path = Path(__file__).parent / config.INFERENCE_PATHS.get("model_path")

# Initializing the variables to store mappings for 200 words and their prediction scores at each point to avoid recalculating same probabilities

mappings = [0] * 200
pred_scores = {}


def get_initial_scores() -> list:
    """
    This method is used to get the first word probability when the text is emtpy.

    Returns:
    A normalized array of initial scores
    """
    global tokenizer
    global model
    tokenized = tokenizer(" ", return_tensors="tf")
    outputs = model.generate(
        **tokenized,
        num_beams=1,
        max_new_tokens=1,
        early_stopping=True,
        no_repeat_ngram_size=2,
        return_dict_in_generate=True,
        output_scores=True,
        renormalize_logits=True
    )
    scores = outputs["scores"][0].numpy()[0]
    normalized_arr = (scores - np.nanmin(scores[scores != -np.inf])) / (
        np.max(scores) - np.nanmin(scores[scores != -np.inf])
    )
    return normalized_arr


def load_tokenizer():
    """
    Loads the tokenizer and returns the tokenizer object
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def load_model():
    """
    Loads the model and returns the model object
    """
    model = TFAutoModelForCausalLM.from_pretrained(model_path)
    return model


def clear_model():
    """
    Clears the previously calculated probabilities when user clears the comment.
    This is useful when user edits the text and start writing from beginning.
    """
    mappings = [0] * 200
    pred_scores = {}


allowed_pos = ["VERB", "NOUN", "INTJ", "ADJ", "ADV", "X"]


def get_pos_toxic_score(words: list) -> float:
    """
    Calculates the toxic score with respect to its POS tags

    Arguments:
    words: List of words with it's POS tag

    Returns:
    A value of the toxicity score
    """
    global mappings
    global allowed_pos

    toxic_score = 0
    toxic_len = 1
    for val in mappings[: len(words)]:
        print(val)
        if val[1] in allowed_pos:
            toxic_len += 1
            toxic_score += val[2]
    # norm_toxic_score = toxic_score/len(words)
    norm_toxic_score = toxic_score / toxic_len
    return norm_toxic_score


def get_toxic_probability(text: str) -> float:
    """
    Calculate the probability of toxic text based on user input

    Arguments:
    text: User input text

    Returns:
    A value of the toxicity score
    """
    if len(text) == 0:
        print("Clearning the past probability calculation")
        clear_model()
    if not text.endswith(" "):
        return None
    global tokenizer
    global model
    global mappings
    global sp
    global pred_scores

    words = text.strip().split(" ")
    if len(words) == 1:
        initial_scores = get_initial_scores()
        pred_scores[words[0]] = initial_scores
        index_of_words = tokenizer.encode(" " + words[0])
        toxic_score = initial_scores[index_of_words[0]]
        if len(index_of_words) > 1:
            for i in range(1, len(index_of_words)):
                toxic_score += initial_scores[index_of_words[i]]
        final_toxic_score = toxic_score / len(index_of_words)
        sen = sp(words[0])
        mappings[0] = [words[0], sen[0].pos_, final_toxic_score]
        # print("first word mappings=> ", mappings)
        return get_pos_toxic_score(words)

    prev_score = pred_scores[" ".join(words[:-1])]
    new_word = words[-1]
    index_of_words = tokenizer.encode(" " + new_word)
    toxic_score = prev_score[index_of_words[0]]
    if len(index_of_words) > 1:
        for i in range(1, len(index_of_words)):
            toxic_score += prev_score[index_of_words[i]]
    final_toxic_score = toxic_score / len(index_of_words)
    sen = sp(" ".join(words))
    mappings[sen[-1].i] = [sen[-1].text, sen[-1].pos_, final_toxic_score]

    tokenized = tokenizer(" ".join(words), return_tensors="tf")
    outputs = model.generate(
        **tokenized,
        num_beams=1,
        max_new_tokens=1,
        early_stopping=True,
        no_repeat_ngram_size=2,
        return_dict_in_generate=True,
        output_scores=True,
        renormalize_logits=True
    )
    scores = outputs["scores"][0].numpy()[0]
    normalized_arr = (scores - np.nanmin(scores[scores != -np.inf])) / (
        np.max(scores) - np.nanmin(scores[scores != -np.inf])
    )
    pred_scores[" ".join(words)] = normalized_arr

    return get_pos_toxic_score(words)


print("loading tokenizer")
tokenizer = load_tokenizer()
print("loading model")
model = load_model()

# Gradio UI
with gr.Blocks() as demo:
    inp = gr.Textbox(placeholder="Write your comment")
    out = gr.Textbox()
    inp.change(get_toxic_probability, inp, out)
demo.launch(inbrowser=True)
