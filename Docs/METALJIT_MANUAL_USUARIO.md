# MetalJIT

**Manual de usuario**

**Version:** 1.0  
**Plataforma:** macOS 13 o posterior, Apple Silicon  
**Lenguajes:** Swift, C, C++, Objective-C++ y Python  
**Tecnologias:** Metal, Metal Shading Language, memoria unificada, Swift Concurrency

MetalJIT permite compilar shaders Metal en tiempo de ejecucion y ejecutarlos directamente sobre memoria del usuario. Esta pensado para aplicaciones nativas de macOS que necesitan combinar la ergonomia de Swift con un nucleo de computo GPU de baja latencia.

El framework incluye una API Swift de alto nivel, una ABI C estable para integracion directa y un binding Python basado en `ctypes`.

## Que aporta MetalJIT

MetalJIT resuelve tres tareas habituales en aplicaciones de computo sobre Apple Silicon:

- Compilar codigo MSL en tiempo de ejecucion.
- Crear y reutilizar pipelines Metal Compute.
- Despachar kernels sobre buffers existentes sin copias intermedias CPU-GPU.

En equipos Apple Silicon, CPU y GPU comparten memoria fisica. MetalJIT aprovecha esa arquitectura envolviendo punteros existentes en buffers Metal cuando la ruta GPU esta disponible. Para cargas pequenas, el despachador puede usar una ruta CPU de baja latencia.

## Requisitos

- macOS 13.0 o posterior.
- Mac con Apple Silicon.
- Xcode 15 o posterior para compilar desde fuente.
- Metal incluido en el sistema.

MetalJIT no requiere dependencias externas para la API Swift o C. El binding Python utiliza la libreria dinamica generada por el proyecto.

## Instalar MetalJIT

### Desde una release

La distribucion publica incluye dos artefactos notarizados:

- `MetalJIT.dmg`
- `MetalJIT_Notarized.zip`

Ambos contienen los frameworks necesarios:

- `MetalJITCore.xcframework`
- `MetalJIT.xcframework`

Para usar MetalJIT en una aplicacion Xcode:

1. Abre el `.dmg` o descomprime el `.zip`.
2. Arrastra `MetalJITCore.xcframework` y `MetalJIT.xcframework` a tu proyecto.
3. En el target de la app, abre **General > Frameworks, Libraries, and Embedded Content**.
4. Configura ambos frameworks como **Embed & Sign**.
5. Importa `MetalJIT` desde Swift.

```swift
import MetalJIT
```

### Desde Swift Package Manager

Durante desarrollo tambien puedes usar el paquete local:

```swift
dependencies: [
    .package(path: "../MetalJIT")
],
targets: [
    .target(
        name: "MiApp",
        dependencies: ["MetalJIT"]
    )
]
```

Compila el paquete con:

```bash
swift build
```

## Primer uso en Swift

El flujo habitual tiene dos objetos:

- `JITCompiler`, que compila shaders y devuelve un `Pipeline`.
- `ComputeDispatcher`, que ejecuta el pipeline sobre buffers Swift.

```swift
import MetalJIT

let compiler = JITCompiler()
let dispatcher = ComputeDispatcher()

let shader = """
#include <metal_stdlib>
using namespace metal;

kernel void duplicar(
    device const uint64_t* input  [[buffer(0)]],
    device       uint64_t* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = input[id] * 2;
}
"""

let pipeline = try compiler.compile(
    shaderSource: shader,
    kernelName: "duplicar"
)

var input: [UInt64] = [1, 2, 3, 4, 5]
var output: [UInt64] = [0, 0, 0, 0, 0]

try input.withUnsafeMutableBufferPointer { inputBuffer in
    try output.withUnsafeMutableBufferPointer { outputBuffer in
        try dispatcher.dispatch(
            pipeline: pipeline,
            input: inputBuffer,
            output: outputBuffer
        )
    }
}

print(output)
```

Salida:

```text
[2, 4, 6, 8, 10]
```

## Pipelines

Un `Pipeline` representa un shader Metal compilado y preparado para despacho.

```swift
let pipeline = try compiler.compile(
    shaderSource: shader,
    kernelName: "kernelName"
)
```

Tambien puedes crear un pipeline CPU-only. Es util para pruebas, fallback o validacion de integracion:

```swift
let pipeline = try compiler.compile()
```

Los pipelines liberan sus recursos automaticamente al ser desasignados. Si quieres liberar recursos de forma explicita:

```swift
try pipeline.destroy()
```

## Despacho de buffers

MetalJIT trabaja con buffers mutables para evitar copias innecesarias y poder entregar punteros estables al motor.

### UInt64

```swift
var input: [UInt64] = [10, 20, 30]
var output: [UInt64] = [0, 0, 0]

try input.withUnsafeMutableBufferPointer { inputBuffer in
    try output.withUnsafeMutableBufferPointer { outputBuffer in
        try dispatcher.dispatch(
            pipeline: pipeline,
            input: inputBuffer,
            output: outputBuffer
        )
    }
}
```

### Float32

```swift
try input.withUnsafeMutableBufferPointer { inputBuffer in
    try output.withUnsafeMutableBufferPointer { outputBuffer in
        try dispatcher.dispatchFloat32(
            pipeline: pipeline,
            input: inputBuffer,
            output: outputBuffer
        )
    }
}
```

### Float64

```swift
try input.withUnsafeMutableBufferPointer { inputBuffer in
    try output.withUnsafeMutableBufferPointer { outputBuffer in
        try dispatcher.dispatchFloat64(
            pipeline: pipeline,
            input: inputBuffer,
            output: outputBuffer
        )
    }
}
```

### Int32 y Float16

La API Swift tambien incluye:

```swift
try dispatcher.dispatchInt32(pipeline: pipeline, input: inputBuffer, output: outputBuffer)
try dispatcher.dispatchFloat16(pipeline: pipeline, input: inputBuffer, output: outputBuffer)
```

`Float16` esta orientado a Apple Silicon y a escenarios de ML, inferencia o procesamiento numerico donde la precision reducida sea adecuada.

## Despacho por lotes

El modo batch procesa un buffer plano de payloads y escribe un resultado por lote.

```swift
let batches = 3
let elementsPerBatch = 4

var payloads: [UInt64] = [
    1, 2, 3, 4,
    10, 20, 30, 40,
    100, 200, 300, 400
]

var results: [UInt64] = [0, 0, 0]

try payloads.withUnsafeMutableBufferPointer { payloadBuffer in
    try results.withUnsafeMutableBufferPointer { resultBuffer in
        try dispatcher.dispatchBatch(
            pipeline: pipeline,
            payloads: payloadBuffer,
            results: resultBuffer,
            numBatches: batches,
            elementsPerBatch: elementsPerBatch
        )
    }
}
```

Tambien hay variantes para `Float32` y `Float64`. Usan la misma forma que `dispatchBatch`, cambiando el tipo de los buffers:

```swift
try dispatcher.dispatchBatchFloat32(
    pipeline: pipeline,
    payloads: payloadBuffer,
    results: resultBuffer,
    numBatches: batches,
    elementsPerBatch: elementsPerBatch
)

try dispatcher.dispatchBatchFloat64(
    pipeline: pipeline,
    payloads: payloadBuffer,
    results: resultBuffer,
    numBatches: batches,
    elementsPerBatch: elementsPerBatch
)
```

## Swift Concurrency

MetalJIT ofrece despachos asincronos compatibles con `async/await`.

```swift
try await dispatcher.dispatchAsync(
    pipeline: pipeline,
    input: inputBuffer,
    output: outputBuffer
)
```

Batch asincrono:

```swift
try await dispatcher.dispatchBatchAsync(
    pipeline: pipeline,
    payloads: payloadBuffer,
    results: resultBuffer,
    numBatches: batches,
    elementsPerBatch: elementsPerBatch
)
```

La llamada vuelve cuando el trabajo ha terminado o cuando el motor informa un error.

## Kernels integrados

MetalJIT incluye kernels predefinidos para pruebas, prototipos y cargas comunes.

```swift
let crypto = try compiler.compile(builtIn: .crypto)
let tensor = try compiler.compile(builtIn: .tensor)
let logic = try compiler.compile(builtIn: .logic)
let physics = try compiler.compile(builtIn: .physics)
let topology = try compiler.compile(builtIn: .topology)
```

Cada kernel integrado incluye:

- Codigo MSL.
- Nombre de kernel.
- Tipo de dato nativo.
- Fallback CPU registrado automaticamente.

| Kernel | Uso | Tipo nativo | Elementos por lote |
| --- | --- | --- | --- |
| `.crypto` | Hash avalancha sobre datos `UInt64` | `UInt64` | 6 |
| `.tensor` | Suma ponderada y activacion | `Float64` | 9 |
| `.logic` | Reduccion bitwise | `UInt64` | 3 |
| `.physics` | Interseccion rayo-esfera | `Float64` | 6 |
| `.topology` | Proyeccion armonica e indice topologico | `Float64` | 6 |

## Tipos soportados

| Tipo | Swift | C | Python | Enum C | Bytes |
| --- | --- | --- | --- | --- | --- |
| Entero sin signo 64-bit | `UInt64` | `uint64_t` | `DTYPE_UINT64` | `MJIT_TYPE_UINT64` | 8 |
| Float 32-bit | `Float` | `float` | `DTYPE_FLOAT32` | `MJIT_TYPE_FLOAT32` | 4 |
| Float 64-bit | `Double` | `double` | `DTYPE_FLOAT64` | `MJIT_TYPE_FLOAT64` | 8 |
| Entero 32-bit | `Int32` | `int32_t` | `DTYPE_INT32` | `MJIT_TYPE_INT32` | 4 |
| Entero 64-bit | Motor C | `int64_t` | `DTYPE_INT64` | `MJIT_TYPE_INT64` | 8 |
| Float 16-bit | `Float16` | Motor C | No expuesto | `MJIT_TYPE_FLOAT16` | 2 |

La API Swift publica expone wrappers directos para `UInt64`, `Float32`, `Float64`, `Int32` y `Float16`. El motor C tambien reconoce `Int64`.

## API C

Incluye el umbrella header:

```c
#include <MetalJITCore/MetalJITCore.h>
```

Compilar un shader:

```c
const char* shader =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void duplicar(device const uint64_t* input [[buffer(0)]], "
    "device uint64_t* output [[buffer(1)]], "
    "uint id [[thread_position_in_grid]]) { "
    "output[id] = input[id] * 2; }";

char error[512];
MJITPipelineHandle pipeline = mjit_compile_pipeline(
    shader,
    "duplicar",
    error,
    sizeof(error)
);

if (pipeline <= 0) {
    printf("Error: %s\n", error);
    return 1;
}
```

Despachar:

```c
uint64_t input[] = {1, 2, 3, 4};
uint64_t output[] = {0, 0, 0, 0};

int result = mjit_dispatch_uint64(pipeline, input, output, 4);

if (result != MJIT_SUCCESS) {
    printf("Dispatch failed: %d\n", result);
}

mjit_destroy_pipeline(pipeline);
```

Despacho generico:

```c
mjit_dispatch(
    pipeline,
    input,
    output,
    count,
    MJIT_TYPE_FLOAT64
);
```

Batch:

```c
mjit_dispatch_batch_uint64(
    pipeline,
    payloads,
    results,
    num_batches,
    elements_per_batch
);
```

## API Python

El binding Python carga la libreria dinamica de MetalJIT y expone una API sencilla para scripts, pruebas y prototipos.

```python
from MetalJITBinding import MetalJIT

api = MetalJIT()

handle = api.compile("""
#include <metal_stdlib>
using namespace metal;
kernel void duplicar(
    device const uint64_t* input [[buffer(0)]],
    device uint64_t* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = input[id] * 2;
}
""", "duplicar")

entrada = [1, 2, 3, 4]
salida = [0, 0, 0, 0]

api.dispatch(handle, entrada, salida)
print(salida)

api.destroy(handle)
```

Para usar punteros crudos, por ejemplo con NumPy:

```python
import ctypes
import numpy as np
from MetalJITBinding import MetalJIT

api = MetalJIT()
handle = api.compile_cpu()

entrada = np.arange(1_000_000, dtype=np.uint64)
salida = np.zeros_like(entrada)

ptr_in = entrada.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
ptr_out = salida.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))

api.dispatch_raw(handle, ptr_in, ptr_out, len(entrada), api.DTYPE_UINT64)
api.destroy(handle)
```

Si la libreria no esta en una ruta conocida, define `METALJIT_LIB` antes de ejecutar Python:

```bash
export METALJIT_LIB=/ruta/a/libMetalJITCore.dylib
```

## Memoria y Zero-Copy

MetalJIT esta disenado para minimizar copias:

- La API recibe buffers existentes.
- El motor trabaja con punteros estables durante el despacho.
- En Apple Silicon, la memoria unificada permite que CPU y GPU observen el mismo almacenamiento fisico.

Para obtener los mejores resultados:

- Mantén vivos los arrays hasta que el despacho termine.
- No modifiques buffers desde otro hilo durante un despacho.
- Usa los metodos asincronos solo cuando el ciclo de vida de los buffers este claramente controlado.
- Prefiere batch cuando tengas muchos trabajos pequenos con la misma forma.

## Heaps

La API C incluye un wrapper sobre `MTLHeap` para reutilizar memoria GPU en bucles de alto rendimiento.

```c
MJITHeapHandle heap = mjit_heap_create(1024 * 1024);

void* a = mjit_heap_allocate(heap, 256, 0);
void* b = mjit_heap_allocate(heap, 512, 0);

size_t total = 0;
size_t used = 0;
mjit_heap_stats(heap, &total, &used);

mjit_heap_reset(heap);
mjit_heap_destroy(heap);
```

El parametro `align` se conserva por compatibilidad de API. La alineacion real se obtiene desde Metal mediante el dispositivo activo.

## Cache de shaders

La cache de shaders guarda fuente MSL validada en disco. Al cargar, MetalJIT recompila la fuente y devuelve un nuevo pipeline.

```c
char error[512];

mjit_cache_compile_and_save(
    shader,
    "/tmp/metaljit/kernel.cache",
    error,
    sizeof(error)
);

MJITPipelineHandle pipeline = mjit_cache_load_library(
    "/tmp/metaljit/kernel.cache",
    "duplicar",
    error,
    sizeof(error)
);
```

Esta cache no serializa `MTLLibrary`. Para cache binaria de PSO, usa `MTLBinaryArchive`.

## MTLBinaryArchive

MetalJIT puede usar `MTLBinaryArchive` para acelerar la creacion de pipeline state objects.

```c
char error[512];

mjit_archive_save(
    shader,
    "duplicar",
    "/tmp/metaljit/kernel.metallib",
    error,
    sizeof(error)
);

MJITPipelineHandle pipeline = mjit_compile_with_archive(
    shader,
    "duplicar",
    "/tmp/metaljit/kernel.metallib",
    error,
    sizeof(error)
);
```

La fuente MSL sigue siendo necesaria. El archive acelera la creacion del PSO; no sustituye al codigo fuente del shader.

## Seleccion de dispositivo

En sistemas con mas de una GPU, la API C permite consultar y seleccionar dispositivo.

```c
int count = mjit_device_count();

for (int i = 0; i < count; i++) {
    printf("GPU %d: %s\n", i, mjit_device_name(i));
}

mjit_select_device(0);
```

La seleccion afecta a compilaciones y despachos posteriores.

## Errores

En Swift, los errores se exponen como `JITCompilerError` y `ComputeDispatcherError`.

En C, las funciones devuelven `MJIT_SUCCESS` o un codigo de error. Las funciones que crean pipelines devuelven un handle positivo si tienen exito.

| Codigo | Constante | Significado |
| --- | --- | --- |
| `0` | `MJIT_SUCCESS` | Operacion completada |
| `101` | `MJIT_ERR_UNINITIALIZED` | Dispositivo o motor no disponible |
| `102` | `MJIT_ERR_COMPILATION_FAILED` | Error compilando MSL o creando pipeline |
| `103` | `MJIT_ERR_INVALID_BUFFER` | Buffer nulo, vacio o invalido |
| `104` | `MJIT_ERR_INVALID_HANDLE` | Pipeline no valido o destruido |
| `105` | `MJIT_ERR_OVERFLOW` | Valor fuera de rango |
| `106` | `MJIT_ERR_UNDERFLOW` | Datos insuficientes |
| `107` | `MJIT_ERR_TYPE_MISMATCH` | Tipo de dato incompatible |

Algunas rutas de compilacion o carga pueden devolver el codigo en negativo para distinguirlo de un handle valido. Trata cualquier valor menor o igual que cero como fallo cuando esperas un `MJITPipelineHandle`.

## Rendimiento

MetalJIT enruta automaticamente el trabajo:

- Cargas pequenas: CPU con Grand Central Dispatch para reducir latencia.
- Cargas grandes: GPU con Metal Compute para maximizar throughput.

Como regla practica, agrupa trabajos pequenos en batch y reutiliza pipelines. La compilacion JIT es una operacion relativamente cara; el despacho de un pipeline ya compilado es la ruta optimizada.

## Buenas practicas

- Compila una vez y reutiliza el `Pipeline`.
- Mantén vivos los buffers hasta que termine el despacho.
- Usa tipos Swift/C que coincidan exactamente con el shader MSL.
- Valida primero con tamanos pequenos y aumenta despues.
- Usa batch para muchas unidades de trabajo pequenas.
- Destruye pipelines explicitamente cuando controles ciclos de vida largos.

## Ejemplos incluidos

El repositorio incluye ejemplos completos para cada superficie de API:

```text
Sources/MetalJITExamples/MetalJITExample.swift
Sources/MetalJITExamples/MetalJITExample.mm
Sources/MetalJITExamples/MetalJITExample.py
```

Ejecutar el ejemplo Swift:

```bash
swift run MetalJITRunner
```

## Licencia

MetalJIT se distribuye bajo licencia Apache 2.0. Consulta el archivo `LICENSE` incluido en el repositorio.
