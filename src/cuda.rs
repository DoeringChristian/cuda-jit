use std::ffi::{c_char, c_uint, c_void, CStr, CString};
use std::fmt::Debug;
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
    pub name: String,
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

        unsafe { cuda.cuDevicePrimaryCtxRetain(&mut context, id).check()? };

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
            .check()?;
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
            .check()?;
            cuda.cuDeviceGetAttribute(
                &mut num_sm,
                CUAttribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as i32,
                id,
            )
            .check()?;
            cuda.cuDeviceGetAttribute(
                &mut unified_addr,
                CUAttribute::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING as i32,
                id,
            )
            .check()?;
            cuda.cuDeviceGetAttribute(
                &mut shared_memory_bytes,
                CUAttribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN as i32,
                id,
            )
            .check()?;
            cuda.cuDeviceGetAttribute(
                &mut cc_minor,
                CUAttribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR as i32,
                id,
            )
            .check()?;
            cuda.cuDeviceGetAttribute(
                &mut cc_major,
                CUAttribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR as i32,
                id,
            )
            .check()?;
            cuda.cuDeviceTotalMem(&mut mem_total, id).check()?;
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
            name,
        })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.cuda
                .cuDevicePrimaryCtxRelease(self.id)
                .check()
                .unwrap()
        };
    }
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
            return Err(CUError::NoDevice);
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
            return Err(CUError::CUDAVersion);
        }

        Ok(Self {
            api: cuda,
            version: (cuda_version_major, cuda_version_minor),
            device_count,
        })
    }

    pub fn compile_jit(&self, buf: &mut str) -> Result<()> {
        trace!("Compiling ptx");
        const log_size: usize = 16384;
        let mut error_log = [0 as u8; log_size];
        let mut info_log = [0 as u8; log_size];
        let mut opts = [
            CU_JIT_OPTIMIZATION_LEVEL,
            CU_JIT_LOG_VERBOSE,
            CU_JIT_INFO_LOG_BUFFER,
            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_ERROR_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_GENERATE_LINE_INFO,
            CU_JIT_GENERATE_DEBUG_INFO,
        ];
        let mut opt_vals = [
            4 as *mut c_void,
            1 as *mut c_void,
            info_log.as_ptr() as *mut c_void,
            log_size as c_uint as *mut c_void,
            error_log.as_ptr() as *mut c_void,
            log_size as c_uint as *mut c_void,
            0 as *mut c_void,
            0 as *mut c_void,
        ];
        let mut link_state: CUlinkState = std::ptr::null();
        unsafe {
            self.cuLinkCreate(
                opts.len() as _,
                opts.as_mut_ptr(),
                opt_vals.as_mut_ptr(),
                &mut link_state,
            )
            .check()?;
        }

        let rt = unsafe {
            self.cuLinkAddData(
                link_state,
                CU_JIT_INPUT_PTX,
                buf.as_mut_ptr() as *mut c_void,
                buf.len() as _,
                null(),
                0,
                null_mut(),
                null_mut(),
            )
            .check()
            .or_else(|e| {
                error!("compilation failed with the error: {:?}", unsafe {
                    CStr::from_bytes_until_nul(&error_log)
                });
                Err(e)
            })?;
        };
        todo!()
    }
}
