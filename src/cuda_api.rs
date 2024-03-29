use log::error;
use std::ffi::{c_char, c_float, c_int, c_uchar, c_uint, c_ulong, c_ushort, c_void};

use dlopen::wrapper::{Container, WrapperApi};
use dlopen_derive::WrapperApi;

#[repr(C)]
pub struct CUctx_st {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CUmod_st {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CUfunc_st {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CUlinkState_st {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CUstream_st {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CUevent_st {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CUarray_st {
    _private: [u8; 0],
}
#[repr(C)]
pub struct CUtexObject_st {
    _private: [u8; 0],
}

pub type CUcontext = *const CUctx_st;
pub type CUmodule = *const CUmod_st;
pub type CUfunction = *const CUfunc_st;
pub type CUlinkState = *const CUlinkState_st;
pub type CUstream = *const CUstream_st;
pub type CUevent = *const CUevent_st;
pub type CUarray = *const CUarray_st;
pub type CUtexObject = *const CUtexObject_st;
pub type CUdevice = c_int;
pub type CUdeviceptr = *const c_void;
pub type CUjit_option = c_int;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum CUresult {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
    CUDA_ERROR_STUB_LIBRARY = 34,
    CUDA_ERROR_DEVICE_UNAVAILABLE = 46,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_DEVICE_NOT_LICENSED = 102,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
    CUDA_ERROR_JIT_COMPILATION_DISABLED = 223,
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224,
    CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC = 225,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_TOO_MANY_PEERS = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS = 716,
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
    CUDA_ERROR_INVALID_PC = 718,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_SYSTEM_NOT_READY = 802,
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
    CUDA_ERROR_MPS_CONNECTION_FAILED = 805,
    CUDA_ERROR_MPS_RPC_FAILURE = 806,
    CUDA_ERROR_MPS_SERVER_NOT_READY = 807,
    CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808,
    CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809,
    CUDA_ERROR_MPS_CLIENT_TERMINATED = 810,
    CUDA_ERROR_CDP_NOT_SUPPORTED = 811,
    CUDA_ERROR_CDP_VERSION_MISMATCH = 812,
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
    CUDA_ERROR_CAPTURED_EVENT = 907,
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
    CUDA_ERROR_TIMEOUT = 909,
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
    CUDA_ERROR_EXTERNAL_DEVICE = 911,
    CUDA_ERROR_INVALID_CLUSTER_SIZE = 912,
    CUDA_ERROR_UNKNOWN = 999,
}
impl CUresult {
    pub fn check(self) -> crate::cuda_result::Result<()> {
        if self != CUresult::CUDA_SUCCESS {
            error!("CUDA Error {:?} with code: {}", self, self as i32);
        }
        self.into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum CUAttribute {
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
}

type size_t = c_ulong;

#[derive(Default, Clone, Copy)]
#[repr(C)]
pub struct CUDA_ARRAY_DESCRIPTOR {
    pub Width: size_t,
    pub Height: size_t,
    pub Format: c_int,
    pub NumChannels: c_uint,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct CUDA_ARRAY3D_DESCRIPTOR {
    pub Width: size_t,
    pub Height: size_t,
    pub Depth: size_t,
    pub Format: c_int,
    pub NumChannels: c_uint,
    pub Flags: c_uint,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct CUDA_RESOURCE_DESC {
    pub resType: c_int,
    pub res: C2RustUnnamed,
    pub flags: c_uint,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed {
    pub array: C2RustUnnamed_1,
    pub reserved: C2RustUnnamed_0,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct C2RustUnnamed_0 {
    pub reserved: [c_int; 32],
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct C2RustUnnamed_1 {
    pub hArray: c_int,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct CUDA_TEXTURE_DESC {
    pub addressMode: [c_int; 3],
    pub filterMode: c_int,
    pub flags: c_uint,
    pub maxAnisotropy: c_uint,
    pub mipmapFilterMode: c_int,
    pub mipmapLevelBias: c_float,
    pub minMipmapLevelClamp: c_float,
    pub maxMipmapLevelClamp: c_float,
    pub borderColor: [c_float; 4],
    pub reserved: [c_int; 12],
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct CUDA_RESOURCE_VIEW_DESC {
    pub format: c_int,
    pub width: size_t,
    pub height: size_t,
    pub depth: size_t,
    pub firstMipmapLevel: c_uint,
    pub lastMipmapLevel: c_uint,
    pub firstLayer: c_uint,
    pub lastLayer: c_uint,
    pub reserved: [c_uint; 16],
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct CUDA_MEMCPY2D {
    pub srcXInBytes: size_t,
    pub srcY: size_t,
    pub srcMemoryType: c_int,
    pub srcHost: *const c_void,
    pub srcDevice: c_int,
    pub srcArray: c_int,
    pub srcPitch: size_t,
    pub dstXInBytes: size_t,
    pub dstY: size_t,
    pub dstMemoryType: c_int,
    pub dstHost: *mut c_void,
    pub dstDevice: c_int,
    pub dstArray: c_int,
    pub dstPitch: size_t,
    pub WidthInBytes: size_t,
    pub Height: size_t,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct CUDA_MEMCPY3D {
    pub srcXInBytes: size_t,
    pub srcY: size_t,
    pub srcZ: size_t,
    pub srcLOD: size_t,
    pub srcMemoryType: c_int,
    pub srcHost: *const c_void,
    pub srcDevice: c_int,
    pub srcArray: c_int,
    pub reserved0: *mut c_void,
    pub srcPitch: size_t,
    pub srcHeight: size_t,
    pub dstXInBytes: size_t,
    pub dstY: size_t,
    pub dstZ: size_t,
    pub dstLOD: size_t,
    pub dstMemoryType: c_int,
    pub dstHost: *mut c_void,
    pub dstDevice: c_int,
    pub dstArray: c_int,
    pub reserved1: *mut c_void,
    pub dstPitch: size_t,
    pub dstHeight: size_t,
    pub WidthInBytes: size_t,
    pub Height: size_t,
    pub Depth: size_t,
}

pub const CU_DEVICE_CPU: c_int = -1;

pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;
pub const CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: c_int = 9;
pub const CU_FUNC_CACHE_PREFER_L1: c_int = 2;

pub const CU_JIT_INPUT_PTX: c_int = 1;
pub const CU_JIT_INFO_LOG_BUFFER: c_int = 3;
pub const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: c_int = 4;
pub const CU_JIT_ERROR_LOG_BUFFER: c_int = 5;
pub const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_int = 6;
pub const CU_JIT_OPTIMIZATION_LEVEL: c_int = 7;
pub const CU_JIT_GENERATE_DEBUG_INFO: c_int = 11;
pub const CU_JIT_LOG_VERBOSE: c_int = 12;
pub const CU_JIT_GENERATE_LINE_INFO: c_int = 13;

// const CU_LAUNCH_PARAM_BUFFER_POINTER (void *) 1
// const CU_LAUNCH_PARAM_BUFFER_SIZE (void *) 2
// const CU_LAUNCH_PARAM_END (void *) 0

pub const CU_MEM_ATTACH_GLOBAL: c_int = 1;
pub const CU_MEM_ADVISE_SET_READ_MOSTLY: c_int = 1;
pub const CU_SHAREDMEM_CARVEOUT_MAX_L1: c_int = 0;

pub const CU_STREAM_DEFAULT: c_int = 0;
pub const CU_STREAM_NON_BLOCKING: c_int = 1;
pub const CU_EVENT_DEFAULT: c_int = 0;
pub const CU_EVENT_DISABLE_TIMING: c_int = 2;
pub const CU_MEMORYTYPE_HOST: c_int = 1;
pub const CU_POINTER_ATTRIBUTE_MEMORY_TYPE: c_int = 2;

pub const CU_RESOURCE_TYPE_ARRAY: c_int = 0;
pub const CU_TR_FILTER_MODE_POINT: c_int = 0;
pub const CU_TR_FILTER_MODE_LINEAR: c_int = 1;
pub const CU_TRSF_NORMALIZED_COORDINATES: c_int = 2;
pub const CU_TR_ADDRESS_MODE_WRAP: c_int = 0;
pub const CU_TR_ADDRESS_MODE_CLAMP: c_int = 1;
pub const CU_TR_ADDRESS_MODE_MIRROR: c_int = 2;
pub const CU_MEMORYTYPE_DEVICE: c_int = 2;
pub const CU_MEMORYTYPE_ARRAY: c_int = 3;

pub const CU_AD_FORMAT_FLOAT: c_int = 0x20;
pub const CU_RES_VIEW_FORMAT_FLOAT_1X32: c_int = 0x16;
pub const CU_RES_VIEW_FORMAT_FLOAT_2X32: c_int = 0x17;
pub const CU_RES_VIEW_FORMAT_FLOAT_4X32: c_int = 0x18;

#[derive(WrapperApi)]
pub struct CudaApi {
    cuCtxEnablePeerAccess: unsafe extern "C" fn(peerContext: CUcontext, Flags: c_uint) -> CUresult,
    cuCtxSynchronize: unsafe extern "C" fn() -> CUresult,
    cuDeviceCanAccessPeer: unsafe extern "C" fn(
        canAccessPeer: *mut c_int,
        dev: CUdevice,
        peerDev: CUdevice,
    ) -> CUresult,
    cuDeviceGet: unsafe extern "C" fn(device: *mut CUdevice, ordinal: c_int) -> CUresult,
    cuDeviceGetAttribute:
        unsafe extern "C" fn(pi: *mut c_int, attrib: c_int, dev: CUdevice) -> CUresult,
    cuDeviceGetCount: unsafe extern "C" fn(count: *mut c_int) -> CUresult,
    cuDeviceGetName: unsafe extern "C" fn(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult,
    cuDevicePrimaryCtxRelease: unsafe extern "C" fn(dev: CUdevice) -> CUresult,
    cuDevicePrimaryCtxRetain: unsafe extern "C" fn(pctx: *mut CUcontext, dev: CUdevice) -> CUresult,
    cuDeviceTotalMem: unsafe extern "C" fn(bytes: *mut size_t, dev: CUdevice) -> CUresult,
    cuDriverGetVersion: unsafe extern "C" fn(driverVersion: *mut c_int) -> CUresult,
    cuEventCreate: unsafe extern "C" fn(phEvent: *mut CUevent, Flags: c_uint) -> CUresult,
    cuEventDestroy: unsafe extern "C" fn(hEvent: CUevent) -> CUresult,
    cuEventRecord: unsafe extern "C" fn(hEvent: CUevent, hStream: CUstream) -> CUresult,
    cuEventSynchronize: unsafe extern "C" fn(hEvent: CUevent) -> CUresult,
    cuEventElapsedTime: unsafe extern "C" fn(
        pMilliseconds: *mut c_float,
        hStart: CUevent,
        hEnd: CUevent,
    ) -> CUresult,
    cuFuncSetAttribute:
        unsafe extern "C" fn(hfunc: CUfunction, attrib: c_int, value: c_int) -> CUresult,
    cuGetErrorName: unsafe extern "C" fn(error: CUresult, pStr: *const *mut c_char) -> CUresult,
    cuGetErrorString: unsafe extern "C" fn(error: CUresult, pStr: *const *mut c_char) -> CUresult,
    cuInit: unsafe extern "C" fn(Flags: c_uint) -> CUresult,
    cuLaunchHostFunc: unsafe extern "C" fn(
        hStream: CUstream,
        func: unsafe extern "C" fn(*mut c_void),
        userData: *mut c_void,
    ) -> CUresult,
    cuLaunchKernel: unsafe extern "C" fn(
        f: CUfunction,
        gridDimX: c_uint,
        gridDimY: c_uint,
        gridDimZ: c_uint,
        blockDimX: c_uint,
        blockDimY: c_uint,
        blockDimZ: c_uint,
        sharedMemBytes: c_uint,
        hStream: CUstream,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult,
    cuLinkAddData: unsafe extern "C" fn(
        state: CUlinkState,
        ty: c_int,
        data: *mut c_void,
        size: size_t,
        name: *const c_char,
        numOptions: c_uint,
        options: *mut c_int,
        optionValues: *mut *mut c_void,
    ) -> CUresult,
    cuLinkComplete: unsafe extern "C" fn(
        state: CUlinkState,
        cubinOut: *mut *mut c_void,
        sizeOut: *mut size_t,
    ) -> CUresult,
    cuLinkCreate: unsafe extern "C" fn(
        numOptions: c_uint,
        options: *mut c_int,
        optionValues: *mut *mut c_void,
        stateOut: *mut CUlinkState,
    ) -> CUresult,
    cuLinkDestroy: unsafe extern "C" fn(state: CUlinkState) -> CUresult,
    cuPointerGetAttribute:
        unsafe extern "C" fn(data: *mut c_void, attribute: c_int, ptr: *mut c_void) -> CUresult,
    cuMemAdvise: unsafe extern "C" fn(
        devPtr: *mut c_void,
        count: size_t,
        advice: c_int,
        device: CUdevice,
    ) -> CUresult,
    cuMemAlloc: unsafe extern "C" fn(dptr: *mut *mut c_void, bytesize: size_t) -> CUresult,
    cuMemAllocHost: unsafe extern "C" fn(pp: *mut *mut c_void, bytesize: size_t) -> CUresult,
    cuMemFree: unsafe extern "C" fn(dptr: *mut c_void) -> CUresult,
    cuMemFreeHost: unsafe extern "C" fn(p: *mut c_void) -> CUresult,
    cuMemcpy:
        unsafe extern "C" fn(dst: *mut c_void, src: *const c_void, ByteCount: size_t) -> CUresult,
    cuMemcpyAsync: unsafe extern "C" fn(
        dst: *mut c_void,
        src: *const c_void,
        ByteCount: size_t,
        hStream: CUstream,
    ) -> CUresult,
    cuMemsetD16Async: unsafe extern "C" fn(
        dstDevice: *mut c_void,
        us: c_ushort,
        N: size_t,
        hStream: CUstream,
    ) -> CUresult,
    cuMemsetD32Async: unsafe extern "C" fn(
        dstDevice: *mut c_void,
        ui: c_uint,
        N: size_t,
        hStream: CUstream,
    ) -> CUresult,
    cuMemsetD8Async: unsafe extern "C" fn(
        dstDevice: *mut c_void,
        uc: c_uchar,
        N: size_t,
        hStream: CUstream,
    ) -> CUresult,
    cuModuleGetFunction: unsafe extern "C" fn(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const c_char,
    ) -> CUresult,
    cuModuleLoadData: unsafe extern "C" fn(module: *mut CUmodule, image: *const c_void) -> CUresult,
    cuModuleUnload: unsafe extern "C" fn(hmod: CUmodule) -> CUresult,
    cuOccupancyMaxPotentialBlockSize: unsafe extern "C" fn(
        minGridSize: *mut c_int,
        blockSize: *mut c_int,
        func: CUfunction,
        blockSizeToDynamicSMemSize: *mut c_void,
        dynamicSMemSize: size_t,
        blockSizeLimit: c_int,
    ) -> CUresult,
    cuCtxPushCurrent: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
    cuCtxPopCurrent: unsafe extern "C" fn(pctx: *mut CUcontext) -> CUresult,
    cuStreamCreate: unsafe extern "C" fn(phStream: *mut CUstream, Flags: c_uint) -> CUresult,
    cuStreamDestroy: unsafe extern "C" fn(hStream: CUstream) -> CUresult,
    cuStreamSynchronize: unsafe extern "C" fn(hStream: CUstream) -> CUresult,
    cuStreamWaitEvent:
        unsafe extern "C" fn(hStream: CUstream, hEvent: CUevent, Flags: c_uint) -> CUresult,
    cuMemAllocAsync: unsafe extern "C" fn(
        dptr: *mut CUdeviceptr,
        bytesize: size_t,
        hStream: CUstream,
    ) -> CUresult,
    cuMemFreeAsync: unsafe extern "C" fn(dptr: CUdeviceptr, hStream: CUstream) -> CUresult,

    cuArrayCreate: unsafe extern "C" fn(
        pHanlde: *mut CUarray,
        pAllocateArray: *const CUDA_ARRAY_DESCRIPTOR,
    ) -> CUresult,
    cuArray3DCreate: unsafe extern "C" fn(
        pHandle: *mut CUarray,
        pAllocateArray: *const CUDA_ARRAY3D_DESCRIPTOR,
    ) -> CUresult,
    cuArray3DGetDescriptor: unsafe extern "C" fn(
        pArrayDescriptor: *mut CUDA_ARRAY3D_DESCRIPTOR,
        hArray: CUarray,
    ) -> CUresult,
    cuArrayDestroy: unsafe extern "C" fn(hArray: CUarray) -> CUresult,
    cuTexObjectCreate: unsafe extern "C" fn(
        pTexObject: *mut CUtexObject,
        pResDesc: *const CUDA_RESOURCE_DESC,
        pTexDesc: *const CUDA_TEXTURE_DESC,
        pResViewDesc: *const CUDA_RESOURCE_VIEW_DESC,
    ) -> CUresult,
    cuTexObjectDestroy: unsafe extern "C" fn(texObject: CUtexObject) -> CUresult,
    cuTexObjectGetResourceDesc:
        unsafe extern "C" fn(pResDesc: *mut CUDA_RESOURCE_DESC, texObject: CUtexObject) -> CUresult,
    cuMemcpy3DAsync:
        unsafe extern "C" fn(pCopy: *const CUDA_MEMCPY3D, hStream: CUstream) -> CUresult,
    cuMemcpy2DAsync:
        unsafe extern "C" fn(pCopy: *const CUDA_MEMCPY2D, hStream: CUstream) -> CUresult,
}
