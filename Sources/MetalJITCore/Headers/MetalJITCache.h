// ==========================================================================
// MetalJITCache.h — Cache de shaders Metal (fuente verificada)
// ==========================================================================
// Guarda fuente MSL verificada (compila y valida antes de guardar).
// Al cargar, recompila JIT desde la fuente cacheada (no carga binario).
//
// Limitacion conocida: no existe API runtime para serializar un MTLLibrary
// a .metallib. La via oficial para cache binaria es usar herramientas
// offline (metal -c) o MTLBinaryArchive para cache de PSO.
// ==========================================================================

#ifndef MetalJITCache_h
#define MetalJITCache_h

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Compila un shader, valida que sea correcto, y guarda la fuente en disco.
// Retorna 0 si OK, codigo de error si falla la compilacion o escritura.
int mjit_cache_compile_and_save(const char* shader_src,
                                 const char* cache_path,
                                 char* error_msg_out,
                                 int error_msg_capacity);

// Lee la fuente MSL desde disco, la recompila JIT, y devuelve un pipeline.
// Retorna handle de pipeline (>0) o codigo de error (<0).
int mjit_cache_load_library(const char* cache_path,
                             const char* kernel_name,
                             char* error_msg_out,
                             int error_msg_capacity);

// Verifica si existe un archivo de cache para la ruta dada.
int mjit_cache_exists(const char* cache_path);

#ifdef __cplusplus
}
#endif

#endif /* MetalJITCache_h */
