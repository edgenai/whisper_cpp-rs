#[cfg(test)]
mod tests {
    use thiserror::Error;

    use whisper_cpp::*;

    #[derive(Debug, Error)]
    enum TestError {
        #[error("whisper error")]
        Whisper(#[from] WhisperError),
        #[error("whisper session error")]
        Session(#[from] WhisperSessionError),
    }

    #[tokio::test]
    async fn it_works() -> Result<(), TestError> {
        let model = WhisperModel::new_from_file("", false)?;
        let session = model.new_session().await?;
        let params = WhisperParams::new(WhisperSampling::default_greedy());
        let samples: Vec<f32> = vec![];

        session.full(params, &samples).await?;

        let mut result = "".to_string();
        for i in 0..session.segment_count() {
            result += &*session.segment_text(i)?;
        }

        println!("{result}");

        Ok(())
    }
}
