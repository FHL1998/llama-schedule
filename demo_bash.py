import torch
import pandas as pd
import os
import logging
from transformers import AutoModelForCausalLM, LlamaTokenizer, GenerationConfig
from peft import PeftModel


# Setup logging
def build_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Configuration
FOLDER_PATH = "validation"  # Path to the folder containing CSV files
MODEL_NAME = "meta-llama/Llama-2-13b-hf"
ADAPTERS_NAME = 'output/llama-2-castllm-13b/checkpoint-1000'
MAX_NEW_TOKENS = 256
PROMPT_TEMPLATE = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, concise, and polite answers to the user's questions.

### Human: {user_question}

### Assistant: """

# Setup the logger
logger = build_logger("castllm_log", "processing.log")


def load_model():
    """Load the model and tokenizer"""
    logger.info(f"Starting to load the model {MODEL_NAME} into memory")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )
    model = PeftModel.from_pretrained(model, ADAPTERS_NAME)
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.bos_token_id = 1

    logger.info(f"Successfully loaded the model {MODEL_NAME} into memory")
    return model, tokenizer


def generate(model, tokenizer, user_question, temperature=0.0, top_p=0.9, max_new_tokens=MAX_NEW_TOKENS):
    """Generate a response using the model"""
    # Format the prompt with the user question
    prompt = PROMPT_TEMPLATE.format(user_question=user_question)

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate the response
    outputs = model.generate(
        **inputs,
        generation_config=GenerationConfig(
            do_sample=temperature > 0.0,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        )
    )

    # Decode the response, skipping the input prompt
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    return response


def process_csv_files():
    """Process all CSV files in the specified folder"""
    # Load the model and tokenizer
    model, tokenizer = load_model()

    # Check if the folder exists
    if not os.path.exists(FOLDER_PATH):
        logger.error(f"Folder not found: {FOLDER_PATH}")
        return

    # Get a list of CSV files
    csv_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(".csv")]

    if not csv_files:
        logger.info(f"No CSV files found in {FOLDER_PATH}")
        return

    # Process each CSV file
    for file_name in csv_files:
        file_path = os.path.join(FOLDER_PATH, file_name)
        logger.info(f"Processing file: {file_path}")

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Check for required columns
            if "Question" not in df.columns:
                logger.warning(f"Skipping {file_name}: Missing 'Question' column")
                continue

            # Process each question and generate an answer
            answers = []
            for question in df["Question"]:
                # Add a hint for concise answers if needed
                full_question = f"{question} Please provide a concise and high-level answer with only the essential information needed to address my question."

                # Generate response
                response = generate(model, tokenizer, full_question)
                answers.append(response)

                # Log the Q&A pair
                logger.info(f"Q: {question}")
                logger.info(f"A: {response}")

            # Add the generated answers to the DataFrame
            df["Generated_Answer"] = answers

            # Save the results to a new CSV file
            output_path = os.path.join(FOLDER_PATH, f"processed_{file_name}")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved results to: {output_path}")

        except Exception as e:
            logger.error(f"Error processing {file_name}: {str(e)}")


if __name__ == "__main__":
    process_csv_files()