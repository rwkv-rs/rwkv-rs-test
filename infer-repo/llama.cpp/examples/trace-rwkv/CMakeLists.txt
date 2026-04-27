set(TARGET llama-trace-rwkv)
add_executable(${TARGET} trace-rwkv.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

find_package(Threads REQUIRED)

llama_add_compile_flags()
