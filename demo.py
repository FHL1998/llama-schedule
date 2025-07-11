import json
import logging
import time

# Install latest bitsandbytes & transformers, accelerate from source
# !pip install -q -U bitsandbytes
# !pip install -q -U git+https://github.com/huggingface/transformers.git
# !pip install -q -U git+https://github.com/huggingface/peft.git
# !pip install -q -U git+https://github.com/huggingface/accelerate.git
# Other requirements for the demo
# !pip install gradio
# !pip install sentencepiece

# Load the model.
# Note: It can take a while to download LLaMA and add the adapter modules.
# You can also use the 13B model by loading in 4bits.

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, \
    TextIteratorStreamer

from utils import build_logger

model_name = "meta-llama/Llama-2-13b-hf"
adapters_name = 'output/llama-2-schedule-llm-13b/checkpoint-1000'

print(f"Starting to load the model {model_name} into memory")

m = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map={"": 0}
)
m = PeftModel.from_pretrained(m, adapters_name)
# m = m.merge_and_unload()
tok = LlamaTokenizer.from_pretrained(model_name)
tok.bos_token_id = 1

stop_token_ids = [0]

print(f"Successfully loaded the model {model_name} into memory")

# Setup the gradio Demo.

import datetime
import os
from threading import Event, Thread
from uuid import uuid4

import gradio as gr
import requests

max_new_tokens = 2048
start_message = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."""
LOGDIR = "."

logger = build_logger("castllm_log", "gradio_web_server.log")
disable_btn = gr.Button(interactive=False)


def get_conv_log_filename(conversation_id):
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-{t.minute:02d}-{t.second:02d}-conv-{conversation_id}.json")
    return name


def upvote(conversation_id, votes, chatbot):
    if conversation_id not in votes:
        votes[conversation_id] = {"upvotes": 0, "downvotes": 0}
    votes[conversation_id]["upvotes"] += 1
    print(f"Votes after upvote: {votes}")  # Debugging
    return votes, chatbot


def downvote(conversation_id, votes, chatbot):
    if conversation_id not in votes:
        votes[conversation_id] = {"upvotes": 0, "downvotes": 0}
    votes[conversation_id]["downvotes"] += 1
    print(f"Votes after downvote: {votes}")  # Debugging
    return votes, chatbot


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def convert_history_to_text(history):
    text = start_message + "".join(
        [
            "".join(
                [
                    f"### Human: {item[0]}\n",
                    f"### Assistant: {item[1]}\n",
                ]
            )
            for item in history[:-1]
        ]
    )
    text += "".join(
        [
            "".join(
                [
                    f"### Human: {history[-1][0]}\n",
                    f"### Assistant: {history[-1][1]}\n",
                ]
            )
        ]
    )
    return text


def log_conversation_with_votes(conversation_id, history, votes, messages, generate_kwargs):
    print("Votes:", votes)
    print("Conversation ID:", conversation_id)
    log_file = get_conv_log_filename(conversation_id)  # Get the local file path
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Extract actual values from gr.State objects
    conversation_id_value = conversation_id.value if isinstance(conversation_id, gr.State) else conversation_id
    votes_value = votes.value if isinstance(votes, gr.State) else votes
    print("Votes value for logging:", votes_value)
    print("Conversation ID value for logging:", conversation_id_value)
    data = {
        "conversation_id":  conversation_id_value,
        "timestamp": timestamp,
        "history": history,
        "votes": votes_value,  # Include the votes in the log
        "messages": messages,
        "generate_kwargs": generate_kwargs,
    }

    # Save data to the local JSON file
    try:
        if os.path.exists(log_file):
            # Append to existing file
            with open(log_file, "r+") as f:
                existing_data = json.load(f)
                existing_data.append(data)
                f.seek(0)
                json.dump(existing_data, f, indent=4)
        else:
            # Create a new file
            with open(log_file, "w") as f:
                json.dump([data], f, indent=4)
        print(f"Conversation logged to {log_file}")
    except Exception as e:
        print(f"Error logging conversation: {e}")


def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


def bot(history, temperature, top_p, top_k, repetition_penalty, conversation_id):
    print(f"history: {history}")
    # Initialize a StopOnTokens object
    stop = StopOnTokens()

    # Construct the input message string for the model by concatenating the current system message and conversation history
    messages = convert_history_to_text(history)

    # Tokenize the messages string
    input_ids = tok(messages, return_tensors="pt").input_ids
    input_ids = input_ids.to(m.device)
    streamer = TextIteratorStreamer(tok, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0.0,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([stop]),
    )

    stream_complete = Event()

    def generate_and_signal_complete():
        m.generate(**generate_kwargs)
        stream_complete.set()

    def log_after_stream_complete():
        stream_complete.wait()
        votes_value = votes.value if isinstance(votes, gr.State) else votes
        log_conversation_with_votes(
            conversation_id,
            history,
            votes_value,
            messages,
            {
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
            },
        )

    t1 = Thread(target=generate_and_signal_complete)
    t1.start()

    # t2 = Thread(target=log_after_stream_complete)
    # t2.start()

    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        history[-1][1] = partial_text
        yield history


def trigger_logging(conversation_id, history, votes, chatbot, messages, temperature, top_p, top_k, repetition_penalty):
    votes_value = votes.value if isinstance(votes, gr.State) else votes
    log_conversation_with_votes(
        conversation_id,
        history,
        votes_value,
        messages,
        {
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
        },
    )
    print(f"Logged conversation with ID: {conversation_id}")
    return votes, chatbot

def get_uuid():
    return str(uuid4())


with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer {font-variant-caps: all-small-caps;}",
) as demo:
    conversation_id = gr.State(get_uuid)
    gr.Markdown(
        """<h1><center>🛠️ Schedule LLM Demo</center></h1>
"""
    )
    # chatbot = gr.Chatbot().style(height=500)
    chatbot = gr.Chatbot()
    votes = gr.State(lambda: {})  # A dictionary to store upvote/downvote counts
    with gr.Row():
        # First row for the message box
        with gr.Column():
            # msg = gr.Textbox(
            #     label="Chat Message Box",
            #     placeholder="Type your message here...",
            #     show_label=False,
            #     lines=2  # Adjust height for better usability
            # ).style(container=True)  # Keep the container for consistency

            msg = gr.Textbox(
                label="Chat Message Box",
                placeholder="Type your message here...",
                show_label=False,
                lines=2  # Adjust height for better usability
            )

    # Second row for all buttons
    with gr.Row():
        # Add buttons in a single row with multiple columns
        with gr.Column(scale=1):
            submit = gr.Button(value="💬 Submit", variant="primary")  # Highlight submit
        with gr.Column(scale=1):
            stop = gr.Button(value="⏹️ Stop")
        with gr.Column(scale=1):
            clear = gr.Button(value="🗑️ Clear")
    with gr.Row():
        with gr.Column(scale=1):
            upvote_btn = gr.Button(value="👍 Upvote")
        with gr.Column(scale=1):
            down_vote_btn = gr.Button(value="👎 Downvote")
        with gr.Column(scale=1):
            flag_btn = gr.Button(value="⚠️ Flag")
    with gr.Row():
        with gr.Accordion("Advanced Options:", open=False):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        temperature = gr.Slider(
                            label="Temperature",
                            value=0.7,
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            interactive=True,
                            info="Higher values produce more diverse outputs",
                        )
                with gr.Column():
                    with gr.Row():
                        top_p = gr.Slider(
                            label="Top-p (nucleus sampling)",
                            value=0.9,
                            minimum=0.0,
                            maximum=1,
                            step=0.01,
                            interactive=True,
                            info=(
                                "Sample from the smallest possible set of tokens whose cumulative probability "
                                "exceeds top_p. Set to 1 to disable and sample from all tokens."
                            ),
                        )
                with gr.Column():
                    with gr.Row():
                        top_k = gr.Slider(
                            label="Top-k",
                            value=0,
                            minimum=0.0,
                            maximum=200,
                            step=1,
                            interactive=True,
                            info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                        )
                with gr.Column():
                    with gr.Row():
                        repetition_penalty = gr.Slider(
                            label="Repetition Penalty",
                            value=1.1,
                            minimum=1.0,
                            maximum=2.0,
                            step=0.1,
                            interactive=True,
                            info="Penalize repetition — 1.0 to disable.",
                        )
    with gr.Row():
        gr.Markdown(
            "Disclaimer: The model can produce factually incorrect output, and should not be relied on to produce "
            "factually accurate information. The model was trained on various public datasets; while great efforts "
            "have been taken to clean the pretraining data, it is possible that this model could generate lewd, "
            "biased, or otherwise offensive outputs.",
            elem_classes=["disclaimer"],
        )
    with gr.Row():
        gr.Markdown(
            "[Privacy policy](https://gist.github.com/samhavens/c29c68cdcd420a9aa0202d0839876dac)",
            elem_classes=["disclaimer"],
        )

    submit_event = msg.submit(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=bot,
        inputs=[
            chatbot,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            conversation_id,
        ],
        outputs=chatbot,
        queue=True,
    )
    # submit_click_event = submit.click(
    #     fn=user,
    #     inputs=[msg, chatbot],
    #     outputs=[msg, chatbot],
    #     queue=False,
    # )
    submit_click_event = submit.click(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=bot,
        inputs=[
            chatbot,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            conversation_id,
        ],
        outputs=chatbot,
        queue=True,
    )

    upvote_event = upvote_btn.click(
        fn=upvote,
        inputs=[conversation_id, votes, chatbot],
        outputs=[votes, chatbot],
        queue=False,
    ).then(
        fn=trigger_logging,
        inputs=[
            conversation_id,
            chatbot,  # Pass history
            votes,
            chatbot,  # Pass updated chatbot messages
            chatbot,  # Replace with actual `messages` if available
            temperature,
            top_p,
            top_k,
            repetition_penalty,
        ],
        outputs=[votes, chatbot],
    )

    # Attach to downvote button
    downvote_event = down_vote_btn.click(
        fn=downvote,
        inputs=[conversation_id, votes, chatbot],
        outputs=[votes, chatbot],
        queue=False,
    ).then(
        fn=trigger_logging,
        inputs=[
            conversation_id,
            chatbot,  # Pass history
            votes,
            chatbot,  # Pass updated chatbot messages
            chatbot,  # Replace with actual `messages` if available
            temperature,
            top_p,
            top_k,
            repetition_penalty,
        ],
        outputs=[votes, chatbot],
    )

    stop.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue(max_size=128, concurrency_count=2)

# Launch your Guanaco Demo!
demo.launch(share=True)
