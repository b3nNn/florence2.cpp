use tokenizers::tokenizer::{Tokenizer, EncodeInput, TruncationParams, PaddingParams, AddedToken};
use std::ffi::{CString, CStr};
use std::os::raw::{c_char, c_void};
use std::ptr;

// Helper to handle Rust strings in C
fn to_c_string(s: &str) -> *mut c_char {
    let c_string = CString::new(s).unwrap();
    c_string.into_raw()
}

// Helper to free C strings allocated by Rust
#[no_mangle]
pub extern "C" fn free_c_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s); // Rust takes ownership and frees it
        }
    }
}

// Helper to free token IDs array allocated by Rust
#[no_mangle]
pub extern "C" fn free_token_ids(ids: *mut u32) {
    if !ids.is_null() {
        unsafe {
            libc::free(ids as *mut c_void);
        }
    }
}

// Opaque pointer to Tokenizer for C
#[repr(C)]
pub struct CTokenizer {
    _private: [u8; 0],
}

// Load a tokenizer from a JSON file (e.g., tokenizer.json)
#[no_mangle]
pub extern "C" fn tokenizer_from_file(file_path: *const c_char) -> *mut CTokenizer {
    unsafe {
        if file_path.is_null() {
            return ptr::null_mut();
        }
        let file_path = CStr::from_ptr(file_path).to_str().unwrap();
        match Tokenizer::from_file(file_path) {
            Ok(tokenizer) => Box::into_raw(Box::new(tokenizer)) as *mut CTokenizer,
            Err(_) => ptr::null_mut(),
        }
    }
}

// Free the tokenizer
#[no_mangle]
pub extern "C" fn tokenizer_free(tokenizer: *mut CTokenizer) {
    if !tokenizer.is_null() {
        unsafe {
            let _ = Box::from_raw(tokenizer as *mut Tokenizer); // Rust takes ownership and frees it
        }
    }
}

// Encode text into token IDs, with support for truncation, padding, and attention masks
#[no_mangle]
pub extern "C" fn tokenizer_encode(
    tokenizer: *mut CTokenizer,
    text: *const c_char,
    ids_out: *mut *mut u32,
    ids_len_out: *mut usize,
    attention_mask_out: *mut *mut u32,
    mask_len_out: *mut usize,
    max_length: usize,
    truncation: bool,
    padding: bool,
    padding_strategy: *const c_char,
) -> bool {
    unsafe {
        if tokenizer.is_null() || text.is_null() || ids_out.is_null() || ids_len_out.is_null() ||
           attention_mask_out.is_null() || mask_len_out.is_null() || padding_strategy.is_null() {
            return false;
        }

        let tokenizer = &mut *(tokenizer as *mut Tokenizer);
        let text = CStr::from_ptr(text).to_str().unwrap();
        let padding_strategy_str = CStr::from_ptr(padding_strategy).to_str().unwrap();

        // Configure truncation if enabled
        if truncation && max_length > 0 {
            let _ = tokenizer.with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }));
        } else {
            let _ = tokenizer.with_truncation(None); // Disable truncation
        }

        // Configure padding if enabled
        if padding {
            let strategy = match padding_strategy_str {
                "longest" => tokenizers::PaddingStrategy::BatchLongest,
                "max_length" => tokenizers::PaddingStrategy::Fixed(max_length),
                "do_not_pad" => tokenizers::PaddingStrategy::BatchLongest, // No padding, but we'll handle this below
                _ => return false, // Invalid padding strategy
            };
            tokenizer.with_padding(Some(PaddingParams {
                strategy,
                ..Default::default()
            }));
        } else {
            tokenizer.with_padding(None); // Disable padding
        }

        // Encode the text
        match tokenizer.encode(EncodeInput::Single(text.into()), true) {
            Ok(encoding) => {
                let ids: Vec<u32> = encoding.get_ids().to_vec();
                let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();
                let len = ids.len();

                // Allocate and copy token IDs
                let ids_ptr = libc::malloc(std::mem::size_of::<u32>() * len) as *mut u32;
                if ids_ptr.is_null() {
                    return false;
                }
                std::ptr::copy_nonoverlapping(ids.as_ptr(), ids_ptr, len);
                *ids_out = ids_ptr;
                *ids_len_out = len;

                // Allocate and copy attention mask
                let mask_ptr = libc::malloc(std::mem::size_of::<u32>() * len) as *mut u32;
                if mask_ptr.is_null() {
                    free_token_ids(ids_ptr);
                    return false;
                }
                std::ptr::copy_nonoverlapping(attention_mask.as_ptr(), mask_ptr, len);
                *attention_mask_out = mask_ptr;
                *mask_len_out = len;

                true
            }
            Err(_) => false,
        }
    }
}

// Decode token IDs back to text
#[no_mangle]
pub extern "C" fn tokenizer_decode(
    tokenizer: *mut CTokenizer,
    ids: *const u32,
    len: usize,
    skip_special_tokens: bool,
) -> *mut c_char {
    unsafe {
        if tokenizer.is_null() || ids.is_null() {
            return ptr::null_mut();
        }
        let tokenizer = &mut *(tokenizer as *mut Tokenizer);
        let ids_slice = std::slice::from_raw_parts(ids, len);
        match tokenizer.decode(ids_slice, skip_special_tokens) {
            Ok(text) => to_c_string(&text),
            Err(_) => ptr::null_mut(),
        }
    }
}

// Get the vocabulary size
#[no_mangle]
pub extern "C" fn tokenizer_get_vocab_size(tokenizer: *mut CTokenizer) -> usize {
    unsafe {
        if tokenizer.is_null() {
            return 0;
        }
        let tokenizer = &*(tokenizer as *mut Tokenizer);
        tokenizer.get_vocab_size(true) // Include added tokens
    }
}

#[no_mangle]
pub extern "C" fn tokenizer_get_special_token_id(
    tokenizer: *mut CTokenizer,
    token: *const c_char,
    id_out: *mut u32,
) -> bool {
    unsafe {
        if tokenizer.is_null() || token.is_null() || id_out.is_null() {
            return false;
        }
        let tokenizer = &*(tokenizer as *mut Tokenizer);
        let token = CStr::from_ptr(token).to_str().unwrap();

        // Get the token ID
        if let Some(id) = tokenizer.token_to_id(token) {
            // Check if it's a special token by encoding a single token and checking if it's treated as special
            let encoding = match tokenizer.encode(EncodeInput::Single(token.into()), true) {
                Ok(enc) => enc,
                Err(_) => return false, // Encoding failed
            };
            let special_tokens = encoding.get_special_tokens_mask();
            if special_tokens[0] == 1 {
                *id_out = id;
                return true;
            }
        }
        false
    }
}

// Add a special token and return its ID
#[no_mangle]
pub extern "C" fn tokenizer_add_special_token(
    tokenizer: *mut CTokenizer,
    token: *const c_char,
) -> u32 {
    unsafe {
        if tokenizer.is_null() || token.is_null() {
            return u32::MAX; // Sentinel value for failure
        }
        let tokenizer = &mut *(tokenizer as *mut Tokenizer);
        let token = match CStr::from_ptr(token).to_str() {
            Ok(s) => s,
            Err(_) => return u32::MAX, // Invalid UTF-8 string
        };

        // Create an AddedToken with special=true
        let added_token = AddedToken::from(token, true);
        let num_added = tokenizer.add_special_tokens(&[added_token]);

        if num_added == 0 {
            return u32::MAX; // Failed to add token
        }

        // Get the ID of the newly added token
        match tokenizer.token_to_id(token) {
            Some(id) => id, // Return the ID of the added token
            None => u32::MAX, // Should not happen after successful addition, but handle it anyway
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizer_is_special_token(
    tokenizer: *mut CTokenizer,
    id: u32,
) -> bool {
    unsafe {
        if tokenizer.is_null() {
            return false;
        }
        let tokenizer = &*(tokenizer as *mut Tokenizer);

        // Get the token corresponding to the ID
        if let Some(token) = tokenizer.id_to_token(id) {
            // Encode the token to check if it's treated as special
            let encoding = match tokenizer.encode(EncodeInput::Single(token.into()), true) {
                Ok(enc) => enc,
                Err(_) => return false, // Encoding failed
            };
            let special_tokens = encoding.get_special_tokens_mask();
            return special_tokens[0] == 1;
        }
        false
    }
}

#[no_mangle]
pub extern "C" fn tokenizer_get_special_tokens(
    tokenizer: *mut CTokenizer,
    tokens_out: *mut *mut *mut c_char,
    ids_out: *mut *mut u32,
    count_out: *mut usize,
) -> bool {
    unsafe {
        if tokenizer.is_null() || tokens_out.is_null() || ids_out.is_null() || count_out.is_null() {
            return false;
        }
        let tokenizer = &*(tokenizer as *mut Tokenizer);
        let vocab = tokenizer.get_vocab(true); // Include added tokens

        // Filter for special tokens by encoding each token and checking if it's treated as special
        let mut special_tokens = Vec::new();
        for (token, id) in &vocab {
            let encoding = match tokenizer.encode(EncodeInput::Single(token.as_str().into()), true) {
                Ok(enc) => enc,
                Err(_) => continue, // Skip if encoding fails
            };
            let special_tokens_mask = encoding.get_special_tokens_mask();
            if special_tokens_mask[0] == 1 {
                special_tokens.push((token.clone(), *id));
            }
        }

        let count = special_tokens.len();
        if count == 0 {
            *tokens_out = ptr::null_mut();
            *ids_out = ptr::null_mut();
            *count_out = 0;
            return true;
        }

        // Allocate memory for tokens and IDs
        let tokens_ptr = libc::malloc(std::mem::size_of::<*mut c_char>() * count) as *mut *mut c_char;
        let ids_ptr = libc::malloc(std::mem::size_of::<u32>() * count) as *mut u32;
        if tokens_ptr.is_null() || ids_ptr.is_null() {
            if !tokens_ptr.is_null() { libc::free(tokens_ptr as *mut c_void); }
            if !ids_ptr.is_null() { libc::free(ids_ptr as *mut c_void); }
            return false;
        }

        // Fill the arrays
        for (i, (token, id)) in special_tokens.iter().enumerate() {
            let c_token = to_c_string(token);
            *tokens_ptr.offset(i as isize) = c_token;
            *ids_ptr.offset(i as isize) = *id;
        }

        *tokens_out = tokens_ptr;
        *ids_out = ids_ptr;
        *count_out = count;
        true
    }
}

// Free the special tokens arrays
#[no_mangle]
pub extern "C" fn free_special_tokens(
    tokens: *mut *mut c_char,
    ids: *mut u32,
    count: usize,
) {
    unsafe {
        if !tokens.is_null() {
            for i in 0..count {
                let token = *tokens.offset(i as isize);
                if !token.is_null() {
                    let _ = CString::from_raw(token);
                }
            }
            libc::free(tokens as *mut c_void);
        }
        if !ids.is_null() {
            libc::free(ids as *mut c_void);
        }
    }
}

// Add multiple special tokens
#[no_mangle]
pub extern "C" fn tokenizer_add_special_tokens(
    tokenizer: *mut CTokenizer,
    tokens: *const *const c_char,
    count: usize,
) -> usize {
    unsafe {
        if tokenizer.is_null() || tokens.is_null() || count == 0 {
            return 0; // Number of tokens added
        }
        let tokenizer = &mut *(tokenizer as *mut Tokenizer);

        // Convert C array of char* to Vec<String>
        let mut rust_tokens = Vec::with_capacity(count);
        for i in 0..count {
            let token_ptr = *tokens.offset(i as isize);
            if token_ptr.is_null() {
                return 0; // Null pointer in array
            }
            match CStr::from_ptr(token_ptr).to_str() {
                Ok(token) => rust_tokens.push(AddedToken::from(token, true)),
                Err(_) => return 0, // Invalid UTF-8
            }
        }

        // Add special tokens
        let num_added = tokenizer.add_special_tokens(&rust_tokens);
        num_added // Return number of tokens successfully added
    }
}