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
        let model_paths = {
            let dir = std::env::var("WHISPER_TEST_MODEL_DIR").unwrap_or_else(|_| {
                eprintln!(
                    "WHISPER_TEST_MODEL environment variable not set. \
                Please set this to the path to a Whisper GGUF model file for the test to run."
                );

                std::process::exit(0)
            });

            let dir = std::path::Path::new(&dir);

            if !dir.is_dir() {
                panic!("\"{}\" is not a directory", dir.to_string_lossy());
            }

            let mut models = tokio::fs::read_dir(dir).await.unwrap();
            let mut rv = vec![];

            while let Some(model) = models.next_entry().await.unwrap() {
                let path = model.path();

                if path.is_file() {
                    let path = path.to_str().unwrap();
                    if path.ends_with(".bin") {
                        rv.push(path.to_string());
                    }
                }
            }

            rv
        };

        let sample_path_str = std::env::var("WHISPER_TEST_SAMPLE").unwrap_or_else(|_| {
            eprintln!(
                "WHISPER_TEST_SAMPLE environment variable not set. \
                Please set this to the path to a sample wav file for the test to run."
            );

            std::process::exit(0)
        });

        for model_path_str in model_paths {
            let device = if cfg!(any(feature = "cuda")) {
                Some(0)
            } else {
                None
            };

            let model = WhisperModel::new_from_file(model_path_str, device)?;

            let mut session = model.new_session().await?;

            let params = WhisperParams::new(WhisperSampling::default_greedy());

            let mut file = std::fs::File::open(&sample_path_str)?;
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

            println!("\n{result}\n");
        }

        Ok(())
    }
}
