mod cuda;
#[allow(unused, non_snake_case, non_camel_case_types)]
mod cuda_api;

use dlopen::wrapper::Container;

use self::cuda::CUDA;

fn main() {
    pretty_env_logger::init();
    let cuda = CUDA::create().unwrap();
}
