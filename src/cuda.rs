use std::ffi::CString;
use std::ptr::null;

use derive_more::Deref;
use dlopen::wrapper::Container;
use log::{error, trace, warn};

use crate::cuda_api::*;

pub struct Device {}

#[derive(Deref)]
pub struct CUDA {
    #[deref]
    api: Container<CudaApi>,
    version: (i32, i32),
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

        Some(Self {
            api: cuda,
            version: (cuda_version_major, cuda_version_minor),
        })
    }

    pub fn device(&self) -> Device {
        let mut device_count = 0;
        unsafe {
            self.cuDeviceGetCount(&mut device_count);
        }

        for i in 0..device_count {
            let mut pci_bus_id = 0;
            let mut pci_dom_id = 0;
            let mut pci_dev_id = 0;
            let mut num_sm = 0;
            let mut unified_addr = 0;
            let mut shared_memory_bytes = 0;
            let mut cc_minor = 0;
            let mut cc_major = 0;
            let mut memory_pool = 0;
            let mut mem_total = 0;

            let mut context: CUcontext = null();
            let mut name = vec![0; 256];

            unsafe { self.cuDevicePrimaryCtxRetain(&mut context, i).check() };

            unsafe {
                self.cuDeviceGetName(name.as_ptr() as *mut i8, 256, i)
                    .check();
            }
            let name = String::from_utf8(name).unwrap();

            unsafe {
                self.cuDeviceGetAttribute(
                    &mut pci_bus_id,
                    CUAttribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID as i32,
                    i,
                )
                .check();
                self.cuDeviceGetAttribute(
                    &mut pci_dev_id,
                    CUAttribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID as i32,
                    i,
                )
                .check();
                self.cuDeviceGetAttribute(
                    &mut pci_dom_id,
                    CUAttribute::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID as i32,
                    i,
                )
                .check();
                self.cuDeviceGetAttribute(
                    &mut num_sm,
                    CUAttribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as i32,
                    i,
                )
                .check();
                self.cuDeviceGetAttribute(
                    &mut unified_addr,
                    CUAttribute::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING as i32,
                    i,
                )
                .check();
                self.cuDeviceGetAttribute(
                    &mut shared_memory_bytes,
                    CUAttribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN as i32,
                    i,
                )
                .check();
                self.cuDeviceGetAttribute(
                    &mut cc_minor,
                    CUAttribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR as i32,
                    i,
                )
                .check();
                self.cuDeviceGetAttribute(
                    &mut cc_major,
                    CUAttribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR as i32,
                    i,
                )
                .check();
                self.cuDeviceTotalMem(&mut mem_total, i).check();
            };

            let cc = cc_major * 10 + cc_minor;

            trace!("Found CUDA Device {name}: PCI_ID {pci_bus_id}, {pci_dev_id}, {pci_dom_id}, compute cap. {cc_major}.{cc_minor}, {num_sm} SMs w/ {shared_memory_bytes}bytes shared mem, {mem_total}bytes global mem.");
        }
        Device {}
    }
}
