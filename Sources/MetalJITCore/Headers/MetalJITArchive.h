// ==========================================================================
// MetalJITArchive.h — Compilacion AOT con MTLBinaryArchive
// ==========================================================================
// Guarda binarios GPU precompilados. En cargas posteriores, el archive
// acelera la creacion del PSO (Pipeline State Object), aunque la compilacion
// de fuente MSL a MTLLibrary sigue siendo necesaria en modo JIT.
// Requiere macOS 13+.
//
// Nota: mjit_compile_with_archive intenta guardar el archive como cache
// best-effort; si el guardado falla, el pipeline compilado se devuelve igual.
// ==========================================================================

#ifndef MetalJITArchive_h
#define MetalJITArchive_h

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Compila un shader y guarda el binario GPU en archive.
// Retorna 0 si OK, <0 si falla.
int mjit_archive_save(const char* shader_src,
                       const char* kernel_name,
                       const char* archive_path,
                       char* error_msg_out,
                       int error_msg_capacity);

// Compila un pipeline usando archive como cache.
// Si el archive existe, acelera la creacion del PSO (evita recompilar el
// pipeline state en GPU). La fuente MSL se recompila a MTLLibrary siempre.
// Si no existe, compila JIT e intenta guardar el archive (best-effort;
// si falla el guardado, el pipeline se devuelve igual).
// Retorna handle (>0) o codigo de error (<0).
int mjit_compile_with_archive(const char* shader_src,
                               const char* kernel_name,
                               const char* archive_path,
                               char* error_msg_out,
                               int error_msg_capacity);

// Verifica si existe un archive.
int mjit_archive_exists(const char* archive_path);

#ifdef __cplusplus
}
#endif

#endif /* MetalJITArchive_h */
