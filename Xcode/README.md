# Xcode Project Setup — MetalJIT

## Generacion automatica (xcodegen)

```bash
cd Xcode
xcodegen generate --spec project.yml
```

Esto crea `MetalJIT.xcodeproj` con dos targets:

| Target | Tipo | Lenguaje |
|--------|------|----------|
| MetalJITCore | Framework | C++/ObjC |
| MetalJIT | Framework | Swift |

## Construir XCFrameworks

```bash
./build_framework.sh release
```

Salida: `Output/MetalJITCore.xcframework/` y `Output/MetalJIT.xcframework/`.

Con firma Developer ID:
```bash
DEVELOPER_IDENTITY="Developer ID Application: Tu Nombre (XXXXXX)" \
  ./build_framework.sh release sign
```

## Estructura del proyecto

```
Xcode/
  project.yml              # Definicion para xcodegen
  MetalJIT.xcodeproj/      # Generado (no editar a mano)
  MetalJITCore.framework/
    Info.plist
    module.modulemap
  MetalJIT.framework/
    Info.plist
```

## Regenerar tras cambios

Si modificas `project.yml` o agregas/quitas archivos fuente:

```bash
cd Xcode
xcodegen generate --spec project.yml
```

El `.xcodeproj` se regenera desde cero. No edites el `.xcodeproj` a mano.
