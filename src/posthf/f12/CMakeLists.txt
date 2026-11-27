# The same graph builds the packaged through-i shell library and the smaller
# explicit qualification domains accepted by the source-checkout builder.
set(GENERATIVEQC_F12_SHELL_DOMAIN i CACHE STRING "Maximum combined-basis angular momentum for molecular F12")
set_property(CACHE GENERATIVEQC_F12_SHELL_DOMAIN PROPERTY STRINGS g h i)
if(NOT GENERATIVEQC_F12_SHELL_DOMAIN MATCHES "^[ghi]$")
  message(FATAL_ERROR "GENERATIVEQC_F12_SHELL_DOMAIN must be g, h or i")
endif()

set(shell_target focal_shell_${GENERATIVEQC_F12_SHELL_DOMAIN})
set(shell_sources
    f12_molecular_integral_sink.cu
    f12_cabs_h_one_electron.cu
    f12_ao_mo_project_cuda.cu
    f12_spin_fock_1e_cuda.cu
    f12_cphf_relax_cuda.cu
    f12_ext_fock_cuda.cu)
if(GENERATIVEQC_ENABLE_CUDA)
  set(backend cuda)
else()
  set(backend host)
  # CMP0119 makes the compiler explicitly treat these shared .cu units as C++.
  set_source_files_properties(${shell_sources} PROPERTIES LANGUAGE CXX)
endif()

function(gqc_f12_install target manifest)
  set(exports "${PROJECT_SOURCE_DIR}/manifests/f12/${manifest}.map")
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(${target} PRIVATE "LINKER:--version-script,${exports}"
                        "LINKER:--no-undefined-version" "LINKER:-z,defs")
    set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS "${exports}")
  endif()
  install(TARGETS ${target}
          LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT f12
          RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT f12)
  install(FILES "${exports}" DESTINATION "${CMAKE_INSTALL_DATADIR}/generativeqc/f12"
          COMPONENT f12)
endfunction()

# Compile the sink twice, once in each generated operator namespace.
set(generated "${CMAKE_CURRENT_BINARY_DIR}/generated/${backend}/${GENERATIVEQC_F12_SHELL_DOMAIN}")
list(REMOVE_ITEM shell_sources f12_molecular_integral_sink.cu)
add_library(${shell_target} SHARED ${shell_sources} f12_shell_projection.cpp
            "${PROJECT_SOURCE_DIR}/src/molecule/basis.cpp")
set(shell_targets ${shell_target})
foreach(operator IN ITEMS coulomb geminal)
  set(kernel "${shell_target}_${operator}")
  set(header "${generated}/molecular-${operator}.hpp")
  add_library(${kernel} OBJECT f12_molecular_integral_sink.cu)
  generativeqc_register_generated_sources(
    TARGET ${kernel} ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/generate_shell_header.py"
    OUTPUTS "${header}"
    ARGS --backend "${backend}" --domain "${GENERATIVEQC_F12_SHELL_DOMAIN}"
         --operator "${operator}" --output "${header}"
    COMMENT "Generating molecular F12 ${operator} shells for ${backend}")
  target_include_directories(${kernel} PRIVATE "${generated}")
  if(operator STREQUAL "coulomb")
    target_compile_definitions(${kernel} PRIVATE GQC_SINK_COULOMB)
  endif()
  set_target_properties(${kernel} PROPERTIES CUDA_STANDARD 17 CXX_STANDARD 17)
  target_sources(${shell_target} PRIVATE $<TARGET_OBJECTS:${kernel}>)
  list(APPEND shell_targets ${kernel})
endforeach()

foreach(target IN LISTS shell_targets)
  target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/include")
  if(GENERATIVEQC_F12_SHELL_DOMAIN STREQUAL "i")
    target_compile_definitions(${target} PRIVATE GQC_F12_CABS_I)
  elseif(GENERATIVEQC_F12_SHELL_DOMAIN STREQUAL "h")
    target_compile_definitions(${target} PRIVATE GQC_F12_CABS_H)
  endif()
  if(NOT GENERATIVEQC_ENABLE_CUDA)
    target_compile_definitions(${target} PRIVATE GQC_SINK_HOST_REFERENCE)
  endif()
  if(_generativeqc_cuda_compile_pool)
    set_property(TARGET ${target} PROPERTY JOB_POOL_COMPILE "${_generativeqc_cuda_compile_pool}")
  endif()
endforeach()
target_compile_definitions(${shell_target} PRIVATE GQC_F12_CABS_I_ONE_ELECTRON)
if(NOT GENERATIVEQC_ENABLE_CUDA)
  set_target_properties(${shell_target} PROPERTIES OUTPUT_NAME "${shell_target}_host")
endif()
gqc_f12_install(${shell_target} focal_shell_i_${backend})

if(GENERATIVEQC_ENABLE_CUDA)
  add_library(focal_hf_cabs_response SHARED
      f12_molecular_metric.cu
      ../cabs_projector_cuda.cu ../cabs_singles_cuda.cu
      ../../solver/cuda/symmetric_eigen_handles.cpp
      ../../solver/cuda/symmetric_eigen_provider.cpp
      ../../solver/cuda/cholesky_provider.cpp)
  add_dependencies(focal_hf_cabs_response generativeqc_matrix_function_codegen)
  target_include_directories(focal_hf_cabs_response PRIVATE
      "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}/generated")
  if(_generativeqc_cuda_compile_pool)
    set_property(TARGET focal_hf_cabs_response PROPERTY JOB_POOL_COMPILE "${_generativeqc_cuda_compile_pool}")
  endif()
  if(GENERATIVEQC_PYTHON_WHEEL)
    generativeqc_attach_cuda_implib(${shell_target})
    generativeqc_attach_cuda_implib(focal_hf_cabs_response)
  else()
    target_link_libraries(${shell_target} PRIVATE CUDA::cudart CUDA::cublas)
    target_link_libraries(focal_hf_cabs_response PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
  endif()
  gqc_f12_install(focal_hf_cabs_response focal_hf_cabs_response)
endif()
