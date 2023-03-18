mod cuda;
#[allow(unused, non_snake_case, non_camel_case_types)]
mod cuda_api;
mod cuda_result;

use std::sync::Arc;

use dlopen::wrapper::Container;

use self::cuda::{Device, CUDA};

fn main() {
    pretty_env_logger::init();
    let cuda = Arc::new(CUDA::create().unwrap());
    let device = Arc::new(Device::create(&cuda, 0));
}
