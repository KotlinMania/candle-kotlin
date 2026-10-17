{
  "targets": [
    {
      "target_name": "candle_native",
      "sources": [
        "src/candle_bindings.cpp",
        "../../src/nativeInterop/cinterop/candle_wrapper.c"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "../../src/nativeInterop/cinterop"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "defines": [ "NAPI_DISABLE_CPP_EXCEPTIONS" ],
      "conditions": [
        [
          "OS=='mac'",
          {
            "link_settings": {
              "libraries": [
                "-framework Accelerate"
              ]
            },
            "xcode_settings": {
              "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
              "CLANG_CXX_LIBRARY": "libc++",
              "MACOSX_DEPLOYMENT_TARGET": "14.0"
            }
          }
        ]
      ]
    }
  ]
}
