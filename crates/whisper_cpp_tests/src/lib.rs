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
        #[error("file was not found: {0}")]
        FileNotFound(#[from] std::io::Error),
    }

    #[tokio::test]
    async fn it_works() -> Result<(), TestError> {
        let model_path_str = std::env::var("WHISPER_TEST_MODEL").unwrap_or_else(|_| {
            eprintln!(
                "WHISPER_TEST_MODEL environment variable not set. \
                Please set this to the path to a GGUF model."
            );

            std::process::exit(1)
        });

        let sample_path_str = std::env::var("WHISPER_TEST_SAMPLE").unwrap_or_else(|_| {
            eprintln!(
                "WHISPER_TEST_SAMPLE environment variable not set. \
                Please set this to the path to a sample wav file."
            );

            std::process::exit(1)
        });

        let model = WhisperModel::new_from_file(model_path_str, false)?;

        let mut session = model.new_session().await?;

        let params = WhisperParams::new(WhisperSampling::default_greedy());

        let mut file = std::fs::File::open(sample_path_str)?;
        let (header, data) = wav::read(&mut file)?;
        let sixteens = data.as_sixteen().unwrap();
        let samples: Vec<_> = sixteens[..sixteens.len() / header.channel_count as usize]
            .iter()
            .map(|v| *v as f32 / 32768.)
            .collect();

        session.full(params, &samples).await?;

        let mut result = "".to_string();
        for i in 0..session.segment_count() {
            result += &*session.segment_text(i)?;
        }

        println!("{result}");

        Ok(())
    }
}
