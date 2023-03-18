use std::ffi::{c_void, CString};
use std::ptr::{null, null_mut};
use std::sync::Arc;

use derive_more::Deref;
use dlopen::wrapper::Container;
use log::{error, trace, warn};

use crate::cuda_api::*;
use crate::cuda_result::*;

pub struct Buffer {
    device: Arc<Device>,
    dptr: *mut c_void,
    size: usize,
}

impl Buffer {
    pub fn create(device: &Arc<Device>, size: usize) -> Self {
        let mut dptr: *mut c_void = null_mut();
        unsafe { device.cuda.cuMemAlloc(&mut dptr, size as _) };
        Self {
            device: device.clone(),
            dptr,
            size,
        }
    }
    pub fn copy_from_slice(&mut self, src: &[u8]) {
        unsafe {
            self.device
                .cuda
                .cuMemcpy(self.dptr, src.as_ptr() as *const c_void, src.len() as _);
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.cuda.cuMemFree(self.dptr);
        }
    }
}

pub struct Device {
    pub cuda: Arc<CUDA>,
    pub context: CUcontext,

    // info
    pub id: i32,
    pub pci_bus_id: i32,
    pub pci_dom_id: i32,
    pub pci_dev_id: i32,
    pub num_sm: i32,
    pub unified_addr: i32,
    pub shared_memory_bytes: i32,
    pub cc_minor: i32,
    pub cc_major: i32,
    pub mem_total: u64,
}

impl Device {
    pub fn create(cuda: &Arc<CUDA>, id: i32) -> Result<Self> {
        assert!(id < cuda.device_count);
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

        unsafe { cuda.cuDevicePrimaryCtxRetain(&mut context, id).check() };

        unsafe {
            cuda.cuDeviceGetName(name.as_ptr() as *mut i8, 256, id)
                .check()?;
        }
        let name = String::from_utf8(name).unwrap();

        unsafe {
            cuda.cuDeviceGetAttribute(
                &mut pci_bus_id,
                CUAttribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID as i32,
                id,
            )
            .check();
            cuda.cuDeviceGetAttribute(
                &mut pci_dev_id,
                CUAttribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID as i32,
                id,
            )
            .check();
            cuda.cuDeviceGetAttribute(
                &mut pci_dom_id,
                CUAttribute::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID as i32,
                id,
            )
            .check();
            cuda.cuDeviceGetAttribute(
                &mut num_sm,
                CUAttribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as i32,
                id,
            )
            .check();
            cuda.cuDeviceGetAttribute(
                &mut unified_addr,
                CUAttribute::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING as i32,
                id,
            )
            .check();
            cuda.cuDeviceGetAttribute(
                &mut shared_memory_bytes,
                CUAttribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN as i32,
                id,
            )
            .check();
            cuda.cuDeviceGetAttribute(
                &mut cc_minor,
                CUAttribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR as i32,
                id,
            )
            .check();
            cuda.cuDeviceGetAttribute(
                &mut cc_major,
                CUAttribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR as i32,
                id,
            )
            .check();
            cuda.cuDeviceTotalMem(&mut mem_total, id).check();
        };

        trace!("Found CUDA Device {name}: PCI_ID {pci_bus_id}, {pci_dev_id}, {pci_dom_id}, compute cap. {cc_major}.{cc_minor}, {num_sm} SMs w/ {shared_memory_bytes}bytes shared mem, {mem_total}bytes global mem.");

        Ok(Device {
            cuda: cuda.clone(),
            context,
            id,
            pci_bus_id,
            pci_dom_id,
            pci_dev_id,
            num_sm,
            unified_addr,
            shared_memory_bytes,
            cc_minor,
            cc_major,
            mem_total,
        })
    }
}

impl Drop for Device {
    fn drop(&mut self) {}
}

#[derive(Deref)]
pub struct CUDA {
    #[deref]
    api: Container<CudaApi>,
    version: (i32, i32),
    device_count: i32,
}

impl CUDA {
    pub fn create() -> Result<Self> {
        let cuda = unsafe { Container::<CudaApi>::load("/usr/lib64/libcuda.so").unwrap() };

        unsafe { cuda.cuInit(0).check()? };

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
            device_count,
        })
    }
}
