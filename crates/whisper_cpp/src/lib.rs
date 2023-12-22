use std::ffi::{CStr, CString};
use std::num::NonZeroUsize;
use std::ptr::null_mut;
use std::slice;
use std::sync::Arc;

use derive_more::{Deref, DerefMut};
use thiserror::Error;
use tokio::sync::RwLock;

use whisper_cpp_sys::{
    whisper_context, whisper_context_params, whisper_free, whisper_free_state,
    whisper_full_default_params, whisper_full_get_segment_text_from_state,
    whisper_full_get_token_id_from_state, whisper_full_n_segments_from_state,
    whisper_full_n_tokens_from_state, whisper_full_params, whisper_full_params__bindgen_ty_1,
    whisper_full_params__bindgen_ty_2, whisper_full_with_state,
    whisper_init_from_file_with_params_no_state, whisper_init_state,
    whisper_sampling_strategy_WHISPER_SAMPLING_BEAM_SEARCH,
    whisper_sampling_strategy_WHISPER_SAMPLING_GREEDY, whisper_state, whisper_token,
};

#[derive(Clone, Deref, DerefMut)]
struct WhisperContext(*mut whisper_context);

unsafe impl Send for WhisperContext {}

unsafe impl Sync for WhisperContext {}

impl Drop for WhisperContext {
    #[doc(alias = "whisper_free")]
    fn drop(&mut self) {
        unsafe { whisper_free(self.0) }
    }
}

#[derive(Debug, Error)]
pub enum WhisperError {
    #[error("failed to initialize the whisper context")]
    Initialization,
    #[error("failed to initialize a new whisper context state")]
    SessionInitialization(#[from] WhisperSessionError),
}

pub struct WhisperModel {
    context: Arc<RwLock<WhisperContext>>,
}

impl WhisperModel {
    /// Loads a new *ggml* *whisper* model, given its file path.
    #[doc(alias = "whisper_init_from_file_with_params_no_state")]
    pub fn new_from_file<P>(model_path: P, use_gpu: bool) -> Result<Self, WhisperError>
        where
            P: AsRef<std::path::Path>,
    {
        let params = whisper_context_params { use_gpu };

        let path_bytes = model_path
            .as_ref()
            .to_string_lossy()
            .to_string()
            .into_bytes();
        let c_str = CString::new(path_bytes).unwrap();

        let context =
            unsafe { whisper_init_from_file_with_params_no_state(c_str.as_ptr(), params) };

        if context.is_null() {
            return Err(WhisperError::Initialization);
        }

        Ok(Self {
            context: Arc::new(RwLock::new(WhisperContext(context))),
        })
    }
    /*
    #[doc(alias = "whisper_init_from_buffer_with_params")]
    pub fn new_from_buffer(buffer: &[u8], use_gpu: bool) -> WhisperModel {
        WhisperModel {}
    }

    #[doc(alias = "whisper_init_with_params")]
    pub fn new(use_gpu: bool) -> WhisperModel {
        WhisperModel {}
    }
     */

    pub async fn new_session(&self) -> Result<WhisperSession, WhisperError> {
        Ok(WhisperSession::new(self.context.clone()).await?)
    }
}

#[derive(Clone, Deref, DerefMut)]
struct WhisperState(*mut whisper_state);

unsafe impl Send for WhisperState {}

unsafe impl Sync for WhisperState {}

impl Drop for WhisperState {
    #[doc(alias = "whisper_free_state")]
    fn drop(&mut self) {
        unsafe { whisper_free_state(self.0) }
    }
}

#[derive(Debug, Error)]
pub enum WhisperSessionError {
    #[error("failed to initialize whisper context state")]
    Initialization,
    #[error("failed to process audio samples")]
    Internal,
    #[error("failed to convert  WhisperParams into whisper_full_params: {0}")]
    Params(#[from] WhisperParamsError),
    #[error("string retrieved from whisper.ccp is invalid: {0}")]
    CStr(#[from] std::str::Utf8Error),
}

// Due to using the with state variant of each function, we can use sessions across multiple
// threads, even if the sessions have the same context.
pub struct WhisperSession {
    context: Arc<RwLock<WhisperContext>>,
    state: WhisperState,
    prompt: Vec<whisper_token>,
}

impl WhisperSession {
    #[doc(alias = "whisper_init_state")]
    async fn new(context: Arc<RwLock<WhisperContext>>) -> Result<Self, WhisperSessionError> {
        let state;
        {
            let locked = context.read().await;
            unsafe { state = whisper_init_state(locked.0) };
        }

        if state.is_null() {
            return Err(WhisperSessionError::Initialization);
        }

        Ok(Self {
            context,
            state: WhisperState(state),
            prompt: vec![],
        })
    }

    /// Convert RAW PCM audio to log mel spectrogram.
    /// The resulting spectrogram is stored inside the default state of the provided whisper context.
    /// Returns 0 on success
    #[doc(alias = "whisper_pcm_to_mel_with_state")]
    pub fn pcm_to_mel(&self) {
        todo!()
    }

    /// Convert RAW PCM audio to log mel spectrogram but applies a Phase Vocoder to speed up the audio x2.
    /// The resulting spectrogram is stored inside the default state of the provided whisper context.
    /// Returns 0 on success
    #[doc(alias = "whisper_pcm_to_mel_phase_vocoder_with_state")]
    pub fn pcm_to_mel_phase_vocoder(&self) {
        todo!()
    }

    /// This can be used to set a custom log mel spectrogram inside the default state of the provided whisper context.
    /// Use this instead of whisper_pcm_to_mel() if you want to provide your own log mel spectrogram.
    /// n_mel must be 80
    /// Returns 0 on success
    #[doc(alias = "whisper_set_mel_with_state")]
    pub fn set_mel(&self) {
        todo!()
    }

    /// Run the Whisper encoder on the log mel spectrogram stored inside the default state in the provided whisper context.
    /// Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first.
    /// offset can be used to specify the offset of the first frame in the spectrogram.
    /// Returns 0 on success
    #[doc(alias = "whisper_encode_with_state")]
    pub fn encode(&self) {
        todo!()
    }

    /// Run the Whisper decoder to obtain the logits and probabilities for the next token.
    /// Make sure to call whisper_encode() first.
    /// tokens + n_tokens is the provided context for the decoder.
    /// n_past is the number of tokens to use from previous decoder calls.
    /// Returns 0 on success
    #[doc(alias = "whisper_decode_with_state")]
    pub fn decode(&self) {
        todo!()
    }

    #[doc(alias = "whisper_lang_auto_detect_with_state")]
    pub fn detect_lang(&self) {
        todo!()
    }

    #[doc(alias = "whisper_n_len_from_state")]
    pub fn mel_len(&self) {
        todo!()
    }

    #[doc(alias = "whisper_get_logits_from_state")]
    pub fn logits(&self) {
        todo!()
    }

    /// Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text.
    /// Uses the specified decoding strategy to obtain the text.
    #[doc(alias = "whisper_full_with_state")]
    pub async fn full(
        &mut self,
        mut params: WhisperParams,
        samples: &[f32],
    ) -> Result<(), WhisperSessionError> {
        params.prompt_tokens = self.prompt.clone();
        let locked = self.context.read().await;
        let res = unsafe {
            let (_vec, c_params) = params.c_params()?;
            whisper_full_with_state(
                locked.0,
                self.state.0,
                c_params,
                samples.as_ptr(),
                samples.len() as std::os::raw::c_int,
            )
        };

        if res != 0 {
            return Err(WhisperSessionError::Internal);
        }

        let segments = self.segment_count();

        for s in 0..segments {
            let tokens = self.token_count(s);
            for t in 0..tokens {
                self.prompt.push(self.token_id(s, t));
            }
        }

        Ok(())
    }

    /// Number of generated text segments.
    /// A segment can be a few words, a sentence, or even a paragraph.
    #[doc(alias = "whisper_full_n_segments_from_state")]
    pub fn segment_count(&self) -> u32 {
        let res = unsafe { whisper_full_n_segments_from_state(self.state.0) };

        res as u32
    }

    /// Get the language id associated with the [`WhisperSession`].
    #[doc(alias = "whisper_full_lang_id_from_state")]
    pub fn lang_id(&self) {
        todo!()
    }

    /// Get the start and end time of the specified segment.
    #[doc(alias = "whisper_full_get_segment_t0_from_state")]
    #[doc(alias = "whisper_full_get_segment_t1_from_state")]
    pub fn segment_time(&self) {
        todo!()
    }

    /// Get whether the next segment is predicted as a speaker turn.
    #[doc(alias = "whisper_full_get_segment_speaker_turn_next_from_state")]
    pub fn is_speaker_next(&self) {
        todo!()
    }

    /// Get the text of the specified segment.
    #[doc(alias = "whisper_full_get_segment_text_from_state")]
    pub fn segment_text(&self, segment: u32) -> Result<String, WhisperSessionError> {
        let text = unsafe {
            let res = whisper_full_get_segment_text_from_state(
                self.state.0,
                segment as std::os::raw::c_int,
            );
            CStr::from_ptr(res.cast_mut())
        };

        Ok(text.to_str()?.to_string())
    }

    /// Get number of tokens in the specified segment.
    #[doc(alias = "whisper_full_n_tokens_from_state")]
    pub fn token_count(&self, segment: u32) -> u32 {
        let res = unsafe {
            whisper_full_n_tokens_from_state(self.state.0, segment as std::os::raw::c_int)
        };

        res as u32
    }

    /// Get the token text of the specified token in the specified segment.
    #[doc(alias = "whisper_full_get_token_text_from_state")]
    pub fn token_text(&self) {
        todo!()
    }

    /// Get the token id of the specified token in the specified segment.
    #[doc(alias = "whisper_full_get_token_id_from_state")]
    pub fn token_id(&self, segment: u32, token: u32) -> i32 {
        unsafe {
            whisper_full_get_token_id_from_state(
                self.state.0,
                segment as std::os::raw::c_int,
                token as std::os::raw::c_int,
            )
        }
    }

    /// Get token data for the specified token in the specified segment.
    /// This contains probabilities, timestamps, etc.
    #[doc(alias = "whisper_full_get_token_data_from_state")]
    pub fn token_data(&self) {
        todo!()
    }

    /// Get the probability of the specified token in the specified segment.
    #[doc(alias = "whisper_full_get_token_p_from_state")]
    pub fn token_probability(&self) {
        todo!()
    }
}

#[derive(Debug, Error)]
pub enum WhisperParamsError {
    #[error("failed to convert String to CString: {0}")]
    SessionInitialization(#[from] std::ffi::NulError),
}

#[derive(Debug)]
pub enum WhisperSampling {
    Greedy {
        /// ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L264
        best_of: u32,
    },
    BeamSearch {
        /// ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L265
        beam_size: u32,

        /// not implemented in whisper.cpp, ref: https://arxiv.org/pdf/2204.05424.pdf
        patience: f32,
    },
}

impl WhisperSampling {
    pub fn default_greedy() -> Self {
        Self::Greedy { best_of: 0 }
    }

    pub fn default_beam() -> Self {
        Self::BeamSearch {
            beam_size: 0,
            patience: 0.0,
        }
    }
}

#[derive(Debug)]
pub struct WhisperParams {
    /// The sampling strategy to be used.
    strategy: WhisperSampling,

    /// Number of threads used for the computation.
    pub thread_count: u32,

    /// Max tokens to use from past text as prompt for the decoder.
    max_text_ctx: u32,

    /// Start offset in milliseconds.
    offset_ms: u32,

    /// Audio duration in milliseconds.
    duration_ms: u32,

    /// Translate the audio to ??? (TODO)
    translate: bool,

    /// Do not use past transcription (if any) as initial prompt for the decoder.
    no_context: bool,

    /// Do not generate timestamps.
    no_timestamps: bool,

    /// Force single segment output (useful for streaming).
    single_segment: bool,

    /// Print special tokens (e.g. <SOT>, <EOT>, <BEG>, etc.).
    print_special: bool,

    /// Print progress information.
    print_progress: bool,

    /// Print results from within whisper.cpp (avoid it, use callback instead).
    pub print_realtime: bool,

    /// Print timestamps for each text segment when printing realtime.
    pub print_timestamps: bool,

    /// Enable token-level timestamps.
    token_timestamps: bool,

    /// Timestamp token probability threshold (~0.01).
    thold_pt: f32,

    /// Timestamp token sum probability threshold (~0.01).
    thold_ptsum: f32,

    /// Max segment length in characters.
    max_len: u32,

    /// Split on word rather than on token (when used with max_len).
    split_on_word: bool,

    /// Max tokens per segment (0 = no limit).
    max_tokens: u32,

    /// Speed-up the audio by 2x using Phase Vocoder.
    ///
    /// Note: these can significantly reduce the quality of the output.
    speed_up: bool,

    /// Enable debug_mode provides extra info (eg. Dump log_mel).
    debug_mode: bool,

    /// Overwrite the audio context size (0 = use default).
    audio_ctx: u32,

    /// Enable *tinydiarize* speaker turn detection.
    tdrz_enable: bool,

    /// Initial prompt, appended to any existing text context from a previous call.
    initial_prompt: String,

    /// Tokens to provide to the whisper decoder as initial prompt.
    /// These are prepended to any existing text context from a previous call.
    prompt_tokens: Vec<i32>,

    /// For auto-detection, set to "" or "auto".
    language: String,

    /// Detect the language automatically.
    detect_language: bool,

    /// ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L89
    suppress_blank: bool,

    /// ref: https://github.com/openai/whisper/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/whisper/tokenizer.py#L224-L253
    suppress_non_speech_tokens: bool,

    /// Initial decoding temperature, ref: https://ai.stackexchange.com/a/32478
    temperature: f32,

    /// ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L97
    max_initial_ts: f32,

    /// ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L267
    length_penalty: f32,

    /// Temperature fallback.
    temperature_inc: f32,

    /// Similar to OpenAI's "compression_ratio_threshold".
    entropy_thold: f32,

    ///
    logprob_thold: f32,

    /// Not implemented in whisper.cpp
    no_speech_thold: f32,

    /// Called for every newly generated text segment.
    _new_segment_callback: (),
    _new_segment_callback_user_data: (),

    /// Called on each progress update.
    _progress_callback: (),
    _progress_callback_user_data: (),

    /// Called each time before the encoder starts.
    _encoder_begin_callback: (),
    _encoder_begin_callback_user_data: (),

    /// Called each time before ggml computation starts.
    _abort_callback: (),
    _abort_callback_user_data: (),

    /// Called by each decoder to filter obtained logits.
    _logits_filter_callback: (),
    _logits_filter_callback_user_data: (),

    ///
    _grammar_rules: (),
    grammar_rule_count: usize,
    i_start_rule: usize,
    grammar_penalty: f32,
}

impl WhisperParams {
    pub fn new(sampling_strategy: WhisperSampling) -> Self {
        let c_strategy = match sampling_strategy {
            WhisperSampling::Greedy { .. } => whisper_sampling_strategy_WHISPER_SAMPLING_GREEDY,
            WhisperSampling::BeamSearch { .. } => {
                whisper_sampling_strategy_WHISPER_SAMPLING_BEAM_SEARCH
            }
        };

        let c_params = unsafe { whisper_full_default_params(c_strategy) };

        c_params.into()
    }

    /// Returns a [`whisper_full_params`] equivalent to this [`WhisperParams`].
    ///
    /// SAFETY: The returned [`whisper_full_params`] object must not live longer than the
    /// accompanying [`Vec`] and this [`WhisperParams`], as it contains pointers to the vector's
    /// elements and members of this object instance.
    unsafe fn c_params(&self) -> Result<(Vec<CString>, whisper_full_params), WhisperParamsError> {
        let mut v = vec![];

        fn push_str(
            storage: &mut Vec<CString>,
            value: &str,
        ) -> Result<*const std::os::raw::c_char, std::ffi::NulError> {
            if value.is_empty() {
                Ok(null_mut())
            } else {
                let i = storage.len();
                storage.push(CString::new(value)?);
                Ok(storage[i].as_ptr())
            }
        }

        let c_params = whisper_full_params {
            strategy: match self.strategy {
                WhisperSampling::Greedy { .. } => whisper_sampling_strategy_WHISPER_SAMPLING_GREEDY,
                WhisperSampling::BeamSearch { .. } => {
                    whisper_sampling_strategy_WHISPER_SAMPLING_BEAM_SEARCH
                }
            },
            n_threads: self.thread_count as std::os::raw::c_int,
            n_max_text_ctx: self.max_text_ctx as std::os::raw::c_int,
            offset_ms: self.offset_ms as std::os::raw::c_int,
            duration_ms: self.duration_ms as std::os::raw::c_int,
            translate: self.translate,
            no_context: self.no_context,
            no_timestamps: self.no_timestamps,
            single_segment: self.single_segment,
            print_special: self.print_special,
            print_progress: self.print_progress,
            print_realtime: self.print_realtime,
            print_timestamps: self.print_timestamps,
            token_timestamps: self.token_timestamps,
            thold_pt: self.thold_pt,
            thold_ptsum: self.thold_ptsum,
            max_len: self.max_len as std::os::raw::c_int,
            split_on_word: self.split_on_word,
            max_tokens: self.max_tokens as std::os::raw::c_int,
            speed_up: self.speed_up,
            debug_mode: self.debug_mode,
            audio_ctx: self.audio_ctx as std::os::raw::c_int,
            tdrz_enable: self.tdrz_enable,
            initial_prompt: push_str(&mut v, &self.initial_prompt)?,
            prompt_tokens: {
                if self.prompt_tokens.is_empty() {
                    null_mut()
                } else {
                    self.prompt_tokens.as_ptr()
                }
            },
            prompt_n_tokens: self.prompt_tokens.len() as std::os::raw::c_int,
            language: push_str(&mut v, &self.language)?,
            detect_language: self.detect_language,
            suppress_blank: self.suppress_blank,
            suppress_non_speech_tokens: self.suppress_non_speech_tokens,
            temperature: self.temperature,
            max_initial_ts: self.max_initial_ts,
            length_penalty: self.length_penalty,
            temperature_inc: self.temperature_inc,
            entropy_thold: self.entropy_thold,
            logprob_thold: self.logprob_thold,
            no_speech_thold: self.no_speech_thold,
            greedy: {
                if let WhisperSampling::Greedy { best_of } = self.strategy {
                    whisper_full_params__bindgen_ty_1 {
                        best_of: best_of as std::os::raw::c_int,
                    }
                } else {
                    whisper_full_params__bindgen_ty_1 { best_of: 0 }
                }
            },
            beam_search: {
                if let WhisperSampling::BeamSearch {
                    beam_size,
                    patience,
                } = self.strategy
                {
                    whisper_full_params__bindgen_ty_2 {
                        beam_size: beam_size as std::os::raw::c_int,
                        patience,
                    }
                } else {
                    whisper_full_params__bindgen_ty_2 {
                        beam_size: 0,
                        patience: 0.0,
                    }
                }
            },
            new_segment_callback: None,
            new_segment_callback_user_data: null_mut(),
            progress_callback: None,
            progress_callback_user_data: null_mut(),
            encoder_begin_callback: None,
            encoder_begin_callback_user_data: null_mut(),
            abort_callback: None,
            abort_callback_user_data: null_mut(),
            logits_filter_callback: None,
            logits_filter_callback_user_data: null_mut(),
            grammar_rules: null_mut(),
            n_grammar_rules: self.grammar_rule_count,
            i_start_rule: self.i_start_rule,
            grammar_penalty: self.grammar_penalty,
        };

        Ok((v, c_params))
    }
}

impl From<whisper_full_params> for WhisperParams {
    fn from(value: whisper_full_params) -> Self {
        Self {
            strategy: {
                match value.strategy {
                    #[allow(non_upper_case_globals)]
                    whisper_sampling_strategy_WHISPER_SAMPLING_GREEDY => WhisperSampling::Greedy {
                        best_of: value.greedy.best_of as u32,
                    },
                    #[allow(non_upper_case_globals)]
                    whisper_sampling_strategy_WHISPER_SAMPLING_BEAM_SEARCH => {
                        WhisperSampling::BeamSearch {
                            beam_size: value.beam_search.beam_size as u32,
                            patience: value.beam_search.patience,
                        }
                    }
                    _ => {
                        unimplemented!("Sampling strategy not implemented!")
                    }
                }
            },
            thread_count: std::thread::available_parallelism()
                .unwrap_or(unsafe { NonZeroUsize::new_unchecked(1) })
                .get() as u32,
            max_text_ctx: value.n_max_text_ctx as u32,
            offset_ms: value.offset_ms as u32,
            duration_ms: value.duration_ms as u32,
            translate: value.translate,
            no_context: value.no_context,
            no_timestamps: value.no_timestamps,
            single_segment: value.single_segment,
            print_special: value.print_special,
            print_progress: value.print_progress,
            print_realtime: value.print_realtime,
            print_timestamps: value.print_timestamps,
            token_timestamps: value.token_timestamps,
            thold_pt: value.thold_pt,
            thold_ptsum: value.thold_ptsum,
            max_len: value.max_len as u32,
            split_on_word: value.split_on_word,
            max_tokens: value.max_tokens as u32,
            speed_up: value.speed_up,
            debug_mode: value.debug_mode,
            audio_ctx: value.audio_ctx as u32,
            tdrz_enable: value.tdrz_enable,
            initial_prompt: {
                if value.initial_prompt.is_null() {
                    "".to_string()
                } else {
                    let c_str = unsafe { CStr::from_ptr(value.initial_prompt.cast_mut()) };
                    c_str.to_str().unwrap_or("").to_string()
                }
            },
            prompt_tokens: {
                if value.prompt_tokens.is_null() {
                    vec![]
                } else {
                    let slice = unsafe {
                        slice::from_raw_parts(value.prompt_tokens, value.prompt_n_tokens as usize)
                    };
                    slice.to_vec()
                }
            },
            language: {
                if value.language.is_null() {
                    "".to_string()
                } else {
                    let c_str = unsafe { CStr::from_ptr(value.language.cast_mut()) };
                    c_str.to_str().unwrap_or("").to_string()
                }
            },
            detect_language: value.detect_language,
            suppress_blank: value.suppress_blank,
            suppress_non_speech_tokens: value.suppress_non_speech_tokens,
            temperature: value.temperature,
            max_initial_ts: value.max_initial_ts,
            length_penalty: value.length_penalty,
            temperature_inc: value.temperature_inc,
            entropy_thold: value.entropy_thold,
            logprob_thold: value.logprob_thold,
            no_speech_thold: value.no_speech_thold,
            _new_segment_callback: (),
            _new_segment_callback_user_data: (),
            _progress_callback: (),
            _progress_callback_user_data: (),
            _encoder_begin_callback: (),
            _encoder_begin_callback_user_data: (),
            _abort_callback: (),
            _abort_callback_user_data: (),
            _logits_filter_callback: (),
            _logits_filter_callback_user_data: (),
            _grammar_rules: (),
            grammar_rule_count: value.n_grammar_rules,
            i_start_rule: value.i_start_rule,
            grammar_penalty: value.grammar_penalty,
        }
    }
}
