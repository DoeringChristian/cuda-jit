use crate::cuda_api::CUresult;

#[derive(Debug, thiserror::Error)]
#[allow(non_camel_case_types)]
#[repr(i32)]
pub enum CUError {
    #[error("CUDA Result {:?}!", .0)]
    CUResult(CUresult),
    #[error("Unknown CUDA Result {}", .0)]
    CUUnknownResult(i32),
    #[error("Unsupported CUDA version!")]
    CUDAVersion,
    #[error("No Device Found!")]
    NoDevice,
    #[error("Unknown Error!")]
    Unknown,
}

pub type Result<T> = std::result::Result<T, CUError>;

impl From<CUresult> for Result<()> {
    fn from(value: CUresult) -> Self {
        match value {
            CUresult::CUDA_SUCCESS => Ok(()),
            _ => Err(CUError::CUResult(value)),
        }
    }
}
