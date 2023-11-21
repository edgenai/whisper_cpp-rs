use std::ffi::CString;
use std::sync::Arc;

use derive_more::{Deref, DerefMut};
use thiserror::Error;
use tokio::sync::RwLock;

use whisper_cpp_sys::{whisper_context, whisper_context_params, whisper_free, whisper_free_state, whisper_full_params, whisper_full_with_state, whisper_init_from_file_with_params_no_state, whisper_init_state, whisper_state};

#[derive(Clone, Deref, DerefMut)]
struct WhisperContext(*mut whisper_context);

unsafe impl Send for WhisperContext {}

unsafe impl Sync for WhisperContext {}

impl Drop for WhisperContext {
    #[doc(alias = "whisper_free")]
    fn drop(&mut self) {
        unsafe {
            whisper_free(self.0)
        }
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

        let path_bytes = model_path.as_ref().to_string_lossy().to_string().into_bytes();
        let c_str = CString::new(path_bytes).unwrap();

        let context = unsafe {
            whisper_init_from_file_with_params_no_state(c_str.as_ptr(), params)
        };

        if context.is_null() {
            return Err(WhisperError::Initialization);
        }

        Ok(Self { context: Arc::new(RwLock::new(WhisperContext(context))) })
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

impl Drop for WhisperState {
    #[doc(alias = "whisper_free_state")]
    fn drop(&mut self) {
        unsafe {
            whisper_free_state(self.0)
        }
    }
}

#[derive(Debug, Error)]
pub enum WhisperSessionError {
    #[error("failed to initialize whisper context state")]
    Initialization,
    #[error("failed to process audio samples")]
    Internal,
}

// Due to using the with state variant of each function, we can use sessions across multiple
// threads, even if the sessions have the same context.
pub struct WhisperSession {
    context: Arc<RwLock<WhisperContext>>,
    state: WhisperState,
}

impl WhisperSession {
    #[doc(alias = "whisper_init_state")]
    async fn new(context: Arc<RwLock<WhisperContext>>) -> Result<Self, WhisperSessionError> {
        let state;
        {
            let locked = context.read().await;
            unsafe {
                state = whisper_init_state(locked.0)
            };
        }

        if state.is_null() { return Err(WhisperSessionError::Initialization); }

        Ok(Self { context, state: WhisperState(state) })
    }

    /// Convert RAW PCM audio to log mel spectrogram.
    /// The resulting spectrogram is stored inside the default state of the provided whisper context.
    /// Returns 0 on success
    #[doc(alias = "whisper_pcm_to_mel_with_state")]
    fn pcm_to_mel(&self) {
        todo!()
    }

    /// Convert RAW PCM audio to log mel spectrogram but applies a Phase Vocoder to speed up the audio x2.
    /// The resulting spectrogram is stored inside the default state of the provided whisper context.
    /// Returns 0 on success
    #[doc(alias = "whisper_pcm_to_mel_phase_vocoder_with_state")]
    fn pcm_to_mel_phase_vocoder(&self) {
        todo!()
    }

    /// This can be used to set a custom log mel spectrogram inside the default state of the provided whisper context.
    /// Use this instead of whisper_pcm_to_mel() if you want to provide your own log mel spectrogram.
    /// n_mel must be 80
    /// Returns 0 on success
    #[doc(alias = "whisper_set_mel_with_state")]
    fn set_mel(&self) {
        todo!()
    }

    /// Run the Whisper encoder on the log mel spectrogram stored inside the default state in the provided whisper context.
    /// Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first.
    /// offset can be used to specify the offset of the first frame in the spectrogram.
    /// Returns 0 on success
    #[doc(alias = "whisper_encode_with_state")]
    fn encode(&self) {
        todo!()
    }

    /// Run the Whisper decoder to obtain the logits and probabilities for the next token.
    /// Make sure to call whisper_encode() first.
    /// tokens + n_tokens is the provided context for the decoder.
    /// n_past is the number of tokens to use from previous decoder calls.
    /// Returns 0 on success
    #[doc(alias = "whisper_decode_with_state")]
    fn decode(&self) {
        todo!()
    }

    #[doc(alias = "whisper_lang_auto_detect_with_state")]
    fn detect_lang(&self) {
        todo!()
    }

    #[doc(alias = "whisper_n_len_from_state")]
    fn mel_len(&self) {
        todo!()
    }

    #[doc(alias = "whisper_get_logits_from_state")]
    fn logits(&self) {
        todo!()
    }

    /// Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text.
    /// Uses the specified decoding strategy to obtain the text.
    #[doc(alias = "whisper_full_with_state")]
    pub async fn full(&self, samples: &[f32]) -> Result<(), WhisperSessionError> {
        let locked = self.context.read().await;
        let c_params = whisper_full_params {};
        let res = unsafe {
            whisper_full_with_state(locked.0, self.state.0, c_params, samples.as_ptr(), samples.len() as std::os::raw::c_int)
        };

        if res != 0 {
            return Err(WhisperSessionError::Internal);
        }

        Ok(())
    }

    /// Number of generated text segments.
    /// A segment can be a few words, a sentence, or even a paragraph.
    #[doc(alias = "whisper_full_n_segments_from_state")]
    fn segment_count(&self) {
        todo!()
    }

    /// Get the language id associated with the [`WhisperSession`].
    #[doc(alias = "whisper_full_lang_id_from_state")]
    fn lang_id(&self) {
        todo!()
    }

    /// Get the start and end time of the specified segment.
    #[doc(alias = "whisper_full_get_segment_t0_from_state")]
    #[doc(alias = "whisper_full_get_segment_t1_from_state")]
    fn segment_time(&self) {
        todo!()
    }

    /// Get whether the next segment is predicted as a speaker turn.
    #[doc(alias = "whisper_full_get_segment_speaker_turn_next_from_state")]
    fn is_speaker_next(&self) {
        todo!()
    }

    /// Get the text of the specified segment.
    #[doc(alias = "whisper_full_get_segment_text_from_state")]
    fn segment_text(&self) {
        todo!()
    }

    /// Get number of tokens in the specified segment.
    #[doc(alias = "whisper_full_n_tokens_from_state")]
    fn token_count(&self) {
        todo!()
    }

    /// Get the token text of the specified token in the specified segment.
    #[doc(alias = "whisper_full_get_token_text_from_state")]
    fn token_text(&self) {
        todo!()
    }

    /// Get the token id of the specified token in the specified segment.
    #[doc(alias = "whisper_full_get_token_id_from_state")]
    fn token_id(&self) {
        todo!()
    }

    /// Get token data for the specified token in the specified segment.
    /// This contains probabilities, timestamps, etc.
    #[doc(alias = "whisper_full_get_token_data_from_state")]
    fn token_data(&self) {
        todo!()
    }

    /// Get the probability of the specified token in the specified segment.
    #[doc(alias = "whisper_full_get_token_p_from_state")]
    fn token_probability(&self) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Error)]
    enum TestError {
        #[error("whisper error")]
        Whisper(#[from] WhisperError),
        #[error("whisper session error")]
        Session(#[from] WhisperSessionError),
    }

    #[test]
    fn it_works() -> Result<(), TestError> {
        let model = WhisperModel::new_from_file("", false)?;


        Ok(())
    }
}
