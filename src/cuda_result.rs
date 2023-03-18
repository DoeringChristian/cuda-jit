use crate::cuda_api::CUresult;

#[derive(Debug, thiserror::Error, num_enum::TryFromPrimitive)]
#[allow(non_camel_case_types)]
#[repr(i32)]
pub enum CUError {
    #[error("Value was not inside acceptable value!")]
    CUDA_ERROR_INVALID_VALUE = 1,
    #[error("Unable to allocate enough memory!")]
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    #[error("The CUDA driver has not been initialized with cuInit() !")]
    CUDA_ERROR_NOT_INITIALIZED = 3,
    #[error("CUDA driver is shutting donw!")]
    CUDA_ERROR_DEINITIALIZED = 4,
    #[error("The named symbol was not found!")]
    CUDA_ERROR_NOT_FOUND = 500,
    #[error("Peer access already enabled")]
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    #[error("Unknown Error!")]
    Unknown = -1,
}

pub type Result<T> = std::result::Result<T, CUError>;

impl From<CUresult> for Result<()> {
    fn from(value: CUresult) -> Self {
        match value.0 {
            0 => Ok(()),
            _ => {
                if let Ok(res) = CUError::try_from(value.0) {
                    Err(res)
                } else {
                    Err(CUError::Unknown)
                }
            }
        }
    }
}
