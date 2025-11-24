/*
 * CXL Memory Expander - C++ Python Module
 *
 * Provides Python bindings to NVIDIA CXL P2P API for PyTorch integration
 */

#include <Python.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <map>

// NVIDIA RM API structures (from nvos.h and ctrl2080bus.h)
#define NV01_ROOT                                   (0x00000000)
#define NV01_ROOT_CLIENT                            (0x00000001)
#define NV01_DEVICE_0                               (0x00000080)
#define NV20_SUBDEVICE_0                            (0x00002080)

#define NV_ESC_RM_ALLOC                             0x27
#define NV_ESC_RM_FREE                              0x29
#define NV_ESC_RM_CONTROL                           0x2a

#define NV2080_CTRL_CMD_BUS_GET_CXL_INFO            0x20801833
#define NV2080_CTRL_CMD_BUS_REGISTER_CXL_BUFFER     0x20801835
#define NV2080_CTRL_CMD_BUS_UNREGISTER_CXL_BUFFER   0x20801836
#define NV2080_CTRL_CMD_BUS_CXL_P2P_DMA_REQUEST     0x20801834

#pragma pack(push, 1)

typedef struct {
    uint32_t hRoot;
    uint32_t hObjectParent;
    uint32_t hObjectNew;
    uint32_t hClass;
    void    *pAllocParms;
    uint32_t paramsSize;
    uint32_t status;
} NVOS21_PARAMETERS;

typedef struct {
    uint32_t hClient;
    uint32_t hObject;
    uint32_t cmd;
    void    *params;
    uint32_t paramsSize;
    uint32_t status;
} NVOS54_PARAMETERS;

typedef struct {
    uint32_t deviceId;
    char     deviceName[8];
    // Other fields omitted
} NV0080_ALLOC_PARAMETERS;

typedef struct {
    uint32_t subDeviceId;
} NV2080_ALLOC_PARAMETERS;

typedef struct {
    uint8_t  bIsLinkUp;
    uint8_t  bMemoryExpander;
    uint32_t nrLinks;
    uint32_t maxNrLinks;
    uint32_t linkMask;
    uint32_t perLinkBwMBps;
    uint32_t cxlVersion;
    uint32_t remoteType;
} NV2080_CTRL_CMD_BUS_GET_CXL_INFO_PARAMS;

typedef struct {
    uint64_t baseAddress;
    uint64_t size;
    uint32_t cxlVersion;
    uint64_t bufferHandle;
} NV2080_CTRL_CMD_BUS_REGISTER_CXL_BUFFER_PARAMS;

typedef struct {
    uint64_t bufferHandle;
} NV2080_CTRL_CMD_BUS_UNREGISTER_CXL_BUFFER_PARAMS;

typedef struct {
    uint64_t cxlBufferHandle;
    uint64_t gpuOffset;
    uint64_t cxlOffset;
    uint64_t size;
    uint32_t flags;
    uint32_t transferId;
} NV2080_CTRL_CMD_BUS_CXL_P2P_DMA_REQUEST_PARAMS;

#pragma pack(pop)

// Global context
struct CXLContext {
    int ctlFd;
    int devFd;
    uint32_t hClient;
    uint32_t hDevice;
    uint32_t hSubdevice;

    CXLContext() : ctlFd(-1), devFd(-1), hClient(0), hDevice(0), hSubdevice(0) {}
};

static CXLContext g_ctx;

struct CXLBuffer {
    void *cpuPtr;
    uint64_t size;
    uint64_t driverHandle;
};

static std::map<uint64_t, CXLBuffer> g_buffers;
static uint64_t g_nextBufferId = 1;

// Helper functions
static int rm_alloc(uint32_t hRoot, uint32_t hParent, uint32_t hObject,
                   uint32_t hClass, void *params, uint32_t paramsSize) {
    NVOS21_PARAMETERS p;
    memset(&p, 0, sizeof(p));
    p.hRoot = hRoot;
    p.hObjectParent = hParent;
    p.hObjectNew = hObject;
    p.hClass = hClass;
    p.pAllocParms = params;
    p.paramsSize = paramsSize;

    if (ioctl(g_ctx.ctlFd, NV_ESC_RM_ALLOC, &p) < 0) {
        return -1;
    }
    return p.status;
}

static int rm_control(uint32_t hClient, uint32_t hObject, uint32_t cmd,
                     void *params, uint32_t paramsSize) {
    NVOS54_PARAMETERS p;
    memset(&p, 0, sizeof(p));
    p.hClient = hClient;
    p.hObject = hObject;
    p.cmd = cmd;
    p.params = params;
    p.paramsSize = paramsSize;

    if (ioctl(g_ctx.ctlFd, NV_ESC_RM_CONTROL, &p) < 0) {
        return -1;
    }
    return p.status;
}

// Initialize CXL connection
static PyObject* cxl_init(PyObject* self, PyObject* args) {
    if (g_ctx.ctlFd >= 0) {
        PyErr_SetString(PyExc_RuntimeError, "Already initialized");
        return NULL;
    }

    // Open control device
    g_ctx.ctlFd = open("/dev/nvidiactl", O_RDWR);
    if (g_ctx.ctlFd < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to open /dev/nvidiactl");
        return NULL;
    }

    // Allocate RM client
    g_ctx.hClient = 0xcaf10001;
    if (rm_alloc(g_ctx.hClient, g_ctx.hClient, g_ctx.hClient, NV01_ROOT_CLIENT, NULL, 0) != 0) {
        close(g_ctx.ctlFd);
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate RM client");
        return NULL;
    }

    // Open device
    g_ctx.devFd = open("/dev/nvidia0", O_RDWR);
    if (g_ctx.devFd < 0) {
        close(g_ctx.ctlFd);
        PyErr_SetString(PyExc_RuntimeError, "Failed to open /dev/nvidia0");
        return NULL;
    }

    // Allocate device
    NV0080_ALLOC_PARAMETERS devParams;
    memset(&devParams, 0, sizeof(devParams));
    devParams.deviceId = 0xb000;  // First GPU

    g_ctx.hDevice = 0xcaf10002;
    if (rm_alloc(g_ctx.hClient, g_ctx.hClient, g_ctx.hDevice, NV01_DEVICE_0,
                &devParams, sizeof(devParams)) != 0) {
        close(g_ctx.devFd);
        close(g_ctx.ctlFd);
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate device");
        return NULL;
    }

    // Allocate subdevice
    NV2080_ALLOC_PARAMETERS subdevParams;
    memset(&subdevParams, 0, sizeof(subdevParams));
    subdevParams.subDeviceId = 0;

    g_ctx.hSubdevice = 0xcaf10003;
    if (rm_alloc(g_ctx.hClient, g_ctx.hDevice, g_ctx.hSubdevice, NV20_SUBDEVICE_0,
                &subdevParams, sizeof(subdevParams)) != 0) {
        close(g_ctx.devFd);
        close(g_ctx.ctlFd);
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate subdevice");
        return NULL;
    }

    Py_RETURN_NONE;
}

// Query CXL capabilities
static PyObject* cxl_query_info(PyObject* self, PyObject* args) {
    if (g_ctx.ctlFd < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Not initialized");
        return NULL;
    }

    NV2080_CTRL_CMD_BUS_GET_CXL_INFO_PARAMS params;
    memset(&params, 0, sizeof(params));

    if (rm_control(g_ctx.hClient, g_ctx.hSubdevice, NV2080_CTRL_CMD_BUS_GET_CXL_INFO,
                  &params, sizeof(params)) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to query CXL info");
        return NULL;
    }

    return Py_BuildValue("{s:i,s:i,s:i,s:i,s:i}",
                        "link_up", params.bIsLinkUp,
                        "memory_expander", params.bMemoryExpander,
                        "num_links", params.nrLinks,
                        "version", params.cxlVersion,
                        "bandwidth_mbps", params.perLinkBwMBps);
}

// Allocate CXL buffer
static PyObject* cxl_alloc_buffer(PyObject* self, PyObject* args) {
    unsigned long long size;
    if (!PyArg_ParseTuple(args, "K", &size)) {
        return NULL;
    }

    // Allocate system memory using mmap
    void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (ptr == MAP_FAILED) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate memory");
        return NULL;
    }

    // Touch pages to ensure they're allocated
    memset(ptr, 0, size);

    // Register with driver
    NV2080_CTRL_CMD_BUS_REGISTER_CXL_BUFFER_PARAMS params;
    memset(&params, 0, sizeof(params));
    params.baseAddress = (uint64_t)ptr;
    params.size = size;
    params.cxlVersion = 2;

    if (rm_control(g_ctx.hClient, g_ctx.hSubdevice, NV2080_CTRL_CMD_BUS_REGISTER_CXL_BUFFER,
                  &params, sizeof(params)) != 0) {
        munmap(ptr, size);
        PyErr_SetString(PyExc_RuntimeError, "Failed to register CXL buffer");
        return NULL;
    }

    // Store buffer info
    uint64_t bufferId = g_nextBufferId++;
    CXLBuffer buf;
    buf.cpuPtr = ptr;
    buf.size = size;
    buf.driverHandle = params.bufferHandle;
    g_buffers[bufferId] = buf;

    return Py_BuildValue("K", bufferId);
}

// Free CXL buffer
static PyObject* cxl_free_buffer(PyObject* self, PyObject* args) {
    unsigned long long bufferId;
    if (!PyArg_ParseTuple(args, "K", &bufferId)) {
        return NULL;
    }

    auto it = g_buffers.find(bufferId);
    if (it == g_buffers.end()) {
        PyErr_SetString(PyExc_ValueError, "Invalid buffer ID");
        return NULL;
    }

    CXLBuffer &buf = it->second;

    // Unregister from driver
    NV2080_CTRL_CMD_BUS_UNREGISTER_CXL_BUFFER_PARAMS params;
    memset(&params, 0, sizeof(params));
    params.bufferHandle = buf.driverHandle;

    rm_control(g_ctx.hClient, g_ctx.hSubdevice, NV2080_CTRL_CMD_BUS_UNREGISTER_CXL_BUFFER,
              &params, sizeof(params));

    // Free memory
    munmap(buf.cpuPtr, buf.size);
    g_buffers.erase(it);

    Py_RETURN_NONE;
}

// Get CPU pointer for buffer
static PyObject* cxl_get_buffer_ptr(PyObject* self, PyObject* args) {
    unsigned long long bufferId;
    if (!PyArg_ParseTuple(args, "K", &bufferId)) {
        return NULL;
    }

    auto it = g_buffers.find(bufferId);
    if (it == g_buffers.end()) {
        PyErr_SetString(PyExc_ValueError, "Invalid buffer ID");
        return NULL;
    }

    return Py_BuildValue("K", (unsigned long long)it->second.cpuPtr);
}

// GPU to CXL transfer
static PyObject* cxl_gpu_to_cxl(PyObject* self, PyObject* args) {
    unsigned long long bufferId, gpuOffset, cxlOffset, size;
    if (!PyArg_ParseTuple(args, "KKKK", &bufferId, &gpuOffset, &cxlOffset, &size)) {
        return NULL;
    }

    auto it = g_buffers.find(bufferId);
    if (it == g_buffers.end()) {
        PyErr_SetString(PyExc_ValueError, "Invalid buffer ID");
        return NULL;
    }

    NV2080_CTRL_CMD_BUS_CXL_P2P_DMA_REQUEST_PARAMS params;
    memset(&params, 0, sizeof(params));
    params.cxlBufferHandle = it->second.driverHandle;
    params.gpuOffset = gpuOffset;
    params.cxlOffset = cxlOffset;
    params.size = size;
    params.flags = 0;  // GPU -> CXL

    if (rm_control(g_ctx.hClient, g_ctx.hSubdevice, NV2080_CTRL_CMD_BUS_CXL_P2P_DMA_REQUEST,
                  &params, sizeof(params)) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "DMA transfer failed");
        return NULL;
    }

    Py_RETURN_NONE;
}

// CXL to GPU transfer
static PyObject* cxl_cxl_to_gpu(PyObject* self, PyObject* args) {
    unsigned long long bufferId, gpuOffset, cxlOffset, size;
    if (!PyArg_ParseTuple(args, "KKKK", &bufferId, &gpuOffset, &cxlOffset, &size)) {
        return NULL;
    }

    auto it = g_buffers.find(bufferId);
    if (it == g_buffers.end()) {
        PyErr_SetString(PyExc_ValueError, "Invalid buffer ID");
        return NULL;
    }

    NV2080_CTRL_CMD_BUS_CXL_P2P_DMA_REQUEST_PARAMS params;
    memset(&params, 0, sizeof(params));
    params.cxlBufferHandle = it->second.driverHandle;
    params.gpuOffset = gpuOffset;
    params.cxlOffset = cxlOffset;
    params.size = size;
    params.flags = 1;  // CXL -> GPU

    if (rm_control(g_ctx.hClient, g_ctx.hSubdevice, NV2080_CTRL_CMD_BUS_CXL_P2P_DMA_REQUEST,
                  &params, sizeof(params)) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "DMA transfer failed");
        return NULL;
    }

    Py_RETURN_NONE;
}

// Method definitions
static PyMethodDef CXLMethods[] = {
    {"init", cxl_init, METH_NOARGS, "Initialize CXL connection"},
    {"query_info", cxl_query_info, METH_NOARGS, "Query CXL capabilities"},
    {"alloc_buffer", cxl_alloc_buffer, METH_VARARGS, "Allocate CXL buffer"},
    {"free_buffer", cxl_free_buffer, METH_VARARGS, "Free CXL buffer"},
    {"get_buffer_ptr", cxl_get_buffer_ptr, METH_VARARGS, "Get buffer CPU pointer"},
    {"gpu_to_cxl", cxl_gpu_to_cxl, METH_VARARGS, "Transfer GPU to CXL"},
    {"cxl_to_gpu", cxl_cxl_to_gpu, METH_VARARGS, "Transfer CXL to GPU"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef cxlmodule = {
    PyModuleDef_HEAD_INIT,
    "cxl_memory",
    "CXL Memory Expander Module",
    -1,
    CXLMethods
};

PyMODINIT_FUNC PyInit_cxl_memory(void) {
    return PyModule_Create(&cxlmodule);
}
