import tiktoken
from tiktoken.load import load_tiktoken_bpe
from transformers.integrations.tiktoken import convert_tiktoken_to_fast
from transformers import AutoTokenizer

def load_kimi_encoding(model_path, base_encoding_name="cl100k_base"):
    """
    Load Kimi encoding from tiktoken model file
    
    Args:
        model_path (str): Path to the tiktoken.model file
        base_encoding_name (str): Name of base encoding to use as fallback
        
    Returns:
        tiktoken.Encoding: Kimi encoding object
    """
    try:
        # Attempt to create encoding directly from file
        with open(model_path, 'rb') as f:
            data = f.read()  # Read file for potential future use
        
        # Load BPE ranks using tiktoken's internal function
        mergeable_ranks = load_tiktoken_bpe(model_path)
        
    except Exception as e:
        # If failed, use base encoding + special tokens
        print(f"Failed to load {model_path} directly: {str(e)}. Using base encoding {base_encoding_name}...")
        base_encoding = tiktoken.get_encoding(base_encoding_name)
        mergeable_ranks = base_encoding._mergeable_ranks
    
    # Kimi's special tokens
    special_tokens = {
        "[BOS]": 163584,
        "[EOS]": 163585,
        "<|im_end|>": 163586,
        "<|im_user|>": 163587,
        "<|im_assistant|>": 163588,
        "<|start_header_id|>": 163590,
        "<|end_header_id|>": 163591,
        "[EOT]": 163593,
        "<|im_system|>": 163594,
        "<|tool_calls_section_begin|>": 163595,
        "<|tool_calls_section_end|>": 163596,
        "<|tool_call_begin|>": 163597,
        "<|tool_call_argument_begin|>": 163598,
        "<|tool_call_end|>": 163599,
        "<|im_middle|>": 163601,
        "[UNK]": 163838,
        "[PAD]": 163839
    }
    
    # Create tiktoken.Encoding object
    return tiktoken.Encoding(
        name="kimi_k2",
        pat_str=r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?[\p{L}]+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens
    )

def convert_to_fast_tokenizer(encoding, output_dir):
    """
    Convert tiktoken encoding to fast tokenizer and save
    
    Args:
        encoding (tiktoken.Encoding): Encoding object to convert
        output_dir (str): Directory to save the fast tokenizer
    """
    # Convert to fast tokenizer
    convert_tiktoken_to_fast(encoding, output_dir)
    print(f"Conversion completed! Fast tokenizer saved to {output_dir}")
    
    # Verify conversion result
    fast_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    print(f"Verification: Fast tokenizer is {'valid' if fast_tokenizer.is_fast else 'invalid'}")
    return fast_tokenizer

def main(model_path, output_dir, base_encoding_name="cl100k_base"):
    """
    Main function to orchestrate the tokenizer conversion process
    
    Args:
        model_path (str): Path to the tiktoken.model file
        output_dir (str): Directory to save the fast tokenizer
        base_encoding_name (str): Name of base encoding to use as fallback
    """
    try:
        # Load encoding using Method 1
        encoding = load_kimi_encoding(model_path, base_encoding_name)
        
        print(f"Successfully created encoding object: {encoding.name}")
        print(f"Number of special tokens: {len(encoding._special_tokens)}")
        print(f"Number of BPE ranks: {len(encoding._mergeable_ranks)}")
        
        # Convert to fast tokenizer
        return convert_to_fast_tokenizer(encoding, output_dir)
        
    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Configuration parameters - can be modified or passed as command line arguments
    MODEL_PATH = "moonshotai/Kimi-K2-Instruct/tiktoken.model"
    OUTPUT_DIR = "moonshotai/Kimi-K2-Instruct"
    BASE_ENCODING = "cl100k_base"
    
    # Execute main function with parameters
    main(MODEL_PATH, OUTPUT_DIR, BASE_ENCODING)
