import os
import logging
import time
import gc
from typing import Dict, Any, List
from huggingface_hub import login

# --- Add these imports ---
from transformers import TextIteratorStreamer # For streaming output
from threading import Thread # To run generation in background

# Import configuration settings
import config
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# --- End of new imports ---

# --- Logging Setup ---
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger(__name__)

# --- Dependency Check & Login ---
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.error("Required packages not installed. Please run: pip install transformers torch bitsandbytes")
    TRANSFORMERS_AVAILABLE = False

if TRANSFORMERS_AVAILABLE:
    try:
        login(token=config.HF_TOKEN)
        logger.info("Hugging Face login successful.")
    except Exception as e:
        logger.error(f"Hugging Face login failed: {e}")
        TRANSFORMERS_AVAILABLE = False

# --- Helper Functions ---
def _format_conversation_prompt(system_prompt: str, conversation: List[Dict[str, str]]) -> str:
    """Formats the conversation history into a single prompt string for Gemma."""
    prompt = f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n\n"
    if not conversation:
        raise IndexError("Cannot format prompt from empty conversation.")
    for message in conversation[:-1]: # All but the latest message
        role = "user" if message["role"] == "user" else "model"
        prompt += f"<start_of_turn>{role}\n{message['content']}<end_of_turn>\n\n"
    # Add the latest user message
    latest_message = conversation[-1]
    prompt += f"<start_of_turn>user\n{latest_message['content']}<end_of_turn>\n\n"
    # Signal the start of the model's turn
    prompt += "<start_of_turn>model\n"
    logger.debug(f"Formatted prompt (first 100 chars): {prompt[:100]}...")
    return prompt

# --- Model Singleton ---
class GemmaModelSingleton:
    _instance = None

    @classmethod
    def get_instance(cls, model_name=config.DEFAULT_MODEL_NAME):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers library not available or login failed. Cannot create model instance.")
        if cls._instance is None:
            logger.info(f"Creating new GemmaModel instance for {model_name}")
            cls._instance = GemmaModel(model_name)
        else:
            logger.info("Reusing existing GemmaModel instance")
        return cls._instance

# --- Core Model Class ---
class GemmaModel:
    def __init__(self, model_name=config.DEFAULT_MODEL_NAME):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers library not available or login failed.")
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.system_prompt = config.SYSTEM_PROMPT
        self.last_used_time = time.time()
        self.is_loaded = False

        if self.device == "cpu":
            logger.warning("CUDA not available. Model will run on CPU, which is not recommended for performance.")

    def load_model(self):
        if self.is_loaded:
            logger.info("Model already loaded.")
            return True

        if self.device == "cpu":
            logger.warning("CUDA not available. Model will run on CPU with offloading, performance will be limited.")

        logger.info(f"Attempting to load model: {self.model_name} onto device: {self.device} with 8-bit quantization")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared before loading.")
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = config.CUDA_ALLOC_CONF

            logger.info(f"Loading tokenizer for {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=config.HF_TOKEN)
            logger.info("Tokenizer loaded.")

            logger.info("Configuring BitsAndBytes using settings from config...")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=config.BNB_LOAD_IN_8BIT,
                llm_int8_enable_fp32_cpu_offload=config.BNB_LLM_INT8_ENABLE_FP32_CPU_OFFLOAD,
                llm_int8_skip_modules=config.BNB_LLM_INT8_SKIP_MODULES,
                llm_int8_threshold=config.BNB_LLM_INT8_THRESHOLD
            )
            logger.info("BitsAndBytes configured.")

            logger.info(f"Loading model {self.model_name} with quantization_config and device_map='auto'...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                offload_folder=config.OFFLOAD_FOLDER,
                trust_remote_code=True,
                token=config.HF_TOKEN
            )

            logger.info("Model loaded successfully with quantization and potential offloading.")
            self.is_loaded = True
            self.last_used_time = time.time()
            return True

        except Exception as e:
            logger.exception(f"Failed to load model: {str(e)}")
            self.model = None
            self.tokenizer = None
            self.is_loaded = False
            self.clear_gpu_memory()
            return False

    def generate_response_stream(self, conversation: List[Dict[str, str]], max_length=config.MAX_OUTPUT_LENGTH, abort_event=None):
        if not self.is_loaded:
            logger.error("Model not loaded, cannot generate response.")
            yield {"status": "error", "message": "Model not loaded"}
            return

        self.last_used_time = time.time()
        logger.info("Starting streamed generation...")

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        try:
            prompt = _format_conversation_prompt(self.system_prompt, conversation)
            input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            generation_kwargs = dict(
                input_ids=input_ids.input_ids,
                attention_mask=input_ids.attention_mask,
                streamer=streamer,
                max_new_tokens=max_length,
                do_sample=config.DO_SAMPLE,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                top_k=config.TOP_K,
                repetition_penalty=config.REPETITION_PENALTY,
                no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
                early_stopping=config.EARLY_STOPPING,
            )

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            logger.info("Generation thread started.")

            full_response_text = ""
            for text_chunk in streamer:
                if abort_event and abort_event.is_set():
                    logger.warning("Abort signal received during streaming.")
                    yield {"status": "aborted", "message": "Generation aborted by client"}
                    return

                if text_chunk:
                    full_response_text += text_chunk
                    yield {"status": "streaming", "chunk": text_chunk}

            thread.join()
            logger.info("Generation thread finished.")
            yield {"status": "success", "full_response": full_response_text}

        except IndexError:
            logger.error("Conversation list appears to be empty during streaming setup.")
            yield {"status": "error", "message": "Cannot generate response from empty conversation."}
        except Exception as e:
            logger.exception(f"Error during streamed generation: {str(e)}")
            yield {"status": "error", "message": f"Error during generation: {str(e)}"}
        finally:
            self.clear_gpu_memory()
            logger.info("Streamed generation process finished.")

    def get_health_status(self) -> Dict[str, Any]:
        status = {
            "is_loaded": self.is_loaded,
            "device": self.device,
            "model_name": self.model_name,
            "gpu_available": torch.cuda.is_available()
        }

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**2
            reserved = torch.cuda.memory_reserved(0) / 1024**2
            status["gpu_memory"] = {
                "allocated_mb": round(allocated, 2),
                "reserved_mb": round(reserved, 2)
            }

        return status

    def unload_if_inactive(self, max_idle_time=config.MAX_IDLE_TIME_SECONDS):
        if not self.is_loaded:
            return False

        current_time = time.time()
        if (current_time - self.last_used_time) > max_idle_time:
            logger.info(f"Model inactive for {max_idle_time} seconds, unloading...")
            self.model = None
            self.tokenizer = None
            self.is_loaded = False

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return True
        return False

    def clear_gpu_memory(self):
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU memory cleared")

# --- API Interface Functions ---
def get_chatbot_response_stream(conversation: List[Dict[str, str]], abort_event=None):
    try:
        model_instance = GemmaModelSingleton.get_instance()
        if not model_instance.is_loaded:
            if not model_instance.load_model():
                def error_generator():
                    yield {"status": "error", "message": "Model failed to load"}
                return error_generator()

        return model_instance.generate_response_stream(conversation, abort_event=abort_event)

    except Exception as e:
        logger.exception(f"Error getting model instance for streaming: {str(e)}")
        def error_generator():
            yield {"status": "error", "message": f"Failed to get model instance: {str(e)}"}
        return error_generator()

def get_health_check() -> Dict[str, Any]:
    try:
        model_instance = GemmaModelSingleton._instance
        if model_instance:
            status_data = model_instance.get_health_status()
        else:
            status_data = {
                "model_name": config.DEFAULT_MODEL_NAME,
                "is_loaded": False,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "transformers_available": TRANSFORMERS_AVAILABLE,
                "gpu_available": torch.cuda.is_available() if TRANSFORMERS_AVAILABLE else False,
                "status": "Instance not created yet."
            }
        return {"status": "success", "data": status_data}
    except Exception as e:
        logger.exception(f"Error during health check: {str(e)}")
        return {"status": "error", "message": f"Health check failed: {str(e)}"}

if __name__ == "__main__":
    print("--- Interactive Chatbot Stream Test ---")
    print("Type 'quit' or 'exit' to end the chat.")
    print("Loading model...")
    try:
        GemmaModelSingleton.get_instance().load_model()
        print("Model ready.")
        conversation_history = []
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]: break
            conversation_history.append({"role": "user", "content": user_input})

            print("Bot: ", end="", flush=True)
            full_bot_response = ""
            stream_generator = get_chatbot_response_stream(conversation_history)
            for result in stream_generator:
                if result["status"] == "streaming":
                    print(result["chunk"], end="", flush=True)
                    full_bot_response += result["chunk"]
                elif result["status"] == "success":
                    print()
                    conversation_history.append({"role": "model", "content": result["full_response"]})
                    break
                elif result["status"] == "error":
                    print(f"\n[Error: {result['message']}]")
                    break
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
    except RuntimeError as e: print(f"\nRuntime Error: {e}")
    except KeyboardInterrupt: print("\nExiting chat due to interrupt.")
    except Exception as e: print(f"\nAn unexpected error occurred: {e}"); logger.exception("Error during interactive chat session:")
    finally:
        instance = GemmaModelSingleton._instance
        if instance and instance.is_loaded: print("\nUnloading model..."); instance.clear_gpu_memory(); print("Model resources released.")