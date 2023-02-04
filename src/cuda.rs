use dlopen::wrapper::Container;
use log::{error, trace, warn};

use crate::cuda_api::{self, CUresult, CudaApi};

pub struct CUDA {
    cuda: Container<CudaApi>,
}

impl CUDA {
    pub fn create() -> Option<Self> {
        let cuda = unsafe { Container::<CudaApi>::load("/usr/lib64/libcuda.so").unwrap() };

        let res = unsafe { cuda.cuInit(0) };
        if res != CUresult::CUDA_SUCCESS {
            error!("cuInit failed with {:?}!", res);
            return None;
        }

        let mut device_count = 0;
        unsafe {
            cuda.cuDeviceGetCount(&mut device_count);
        }

        trace!("Device Count: {device_count}");

        if device_count < 1 {
            error!("No CUDA enabled devices found!");
            return None;
        }

        let mut cuda_version = 0;
        unsafe {
            cuda.cuDriverGetVersion(&mut cuda_version);
        }
        let cuda_version_major = cuda_version / 1000;
        let cuda_version_minor = (cuda_version % 1000) / 10;

        trace!("Cuda version: {cuda_version_major}.{cuda_version_minor}");

        if cuda_version_major < 10 {
            error!("Cuda version {cuda_version_major}.{cuda_version_minor} is too old!");
            return None;
        }

        Some(Self { cuda })
    }
}
