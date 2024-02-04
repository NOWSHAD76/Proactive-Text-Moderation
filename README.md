# Proactive-Text-Moderation

## Project Overview

This project is dedicated to the proactive prediction of potentially toxic content as users write, rather than relying on reactive classification after the entire text is submitted. The primary goal is to enhance online communication platforms, particularly in scenarios like Reddit, by identifying and addressing toxic language in real-time.

## Key Features

1. **Real-time Toxicity Prediction:** The system continuously monitors user input and predicts whether the content being written is potentially toxic.
2. **Nudge for Rephrasing:** If the toxicity confidence surpasses a predefined threshold, users are gently nudged to rephrase their text, promoting a more positive and constructive communication environment.
3. **Proactive Suggestions for Politeness:** The solution proactively suggests users to adopt polite and kind language while they are in the process of composing text, fostering a more respectful discourse.
4. **Optimized for Large Texts:** Particularly beneficial for platforms like Reddit, where users often compose extensive texts, the system excels in handling and monitoring substantial amounts of content in real-time.

## How It Works

![This is the architecture.](images/toxic_text_prediction_architecture.gif "This is the architecture.")

## Installation and Usage

1. Clone the repository

```
git clone https://github.com/NOWSHAD76/Proactive-Text-Moderation.git
```

2. Install dependencies

```
pip install -r requirements.txt
```

_Create a virtual environment if required before installing_

3. Update config file `model_train/config.py` for model training if required 4. Start the model training

```
python model_train/train.py
```

5. Run the main.py to view the Gradio UI and test

```
python main.py
```

