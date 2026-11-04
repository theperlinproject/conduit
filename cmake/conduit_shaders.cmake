include_guard(GLOBAL)

if(NOT CONDUIT_PACK_SHADERS)
    foreach(candidate "${CMAKE_CURRENT_LIST_DIR}/../tools/pack_shaders.py" "${CMAKE_CURRENT_LIST_DIR}/../../../share/conduit/pack_shaders.py"
                      "${CMAKE_CURRENT_LIST_DIR}/../../../${CMAKE_INSTALL_DATADIR}/conduit/pack_shaders.py")
        if(EXISTS "${candidate}")
            get_filename_component(candidate "${candidate}" ABSOLUTE)
            set(CONDUIT_PACK_SHADERS "${candidate}" CACHE INTERNAL "conduit shader packer")
            break()
        endif()
    endforeach()
endif()

function(conduit_add_shader_pack name)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "OUTPUT;SLOTS;SLANGC;CACHE;LIBRARY_PATH" "TARGETS;INCLUDE_DIRS;PROGRAMS;DEPENDS")
    if(NOT arg_OUTPUT OR NOT arg_SLOTS OR NOT arg_SLANGC OR NOT arg_TARGETS OR NOT arg_PROGRAMS)
        message(FATAL_ERROR "conduit_add_shader_pack: OUTPUT, SLOTS, SLANGC, TARGETS and PROGRAMS are required")
    endif()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    list(JOIN arg_TARGETS "," targets)
    set(command "${Python3_EXECUTABLE}" -I "${CONDUIT_PACK_SHADERS}" --slangc "${arg_SLANGC}" --slots "${arg_SLOTS}" --targets "${targets}")
    foreach(dir IN LISTS arg_INCLUDE_DIRS)
        list(APPEND command -I "${dir}")
    endforeach()
    if(arg_CACHE)
        list(APPEND command --cache "${arg_CACHE}")
    endif()
    if(arg_LIBRARY_PATH)
        list(APPEND command --library-path "${arg_LIBRARY_PATH}")
    endif()
    get_filename_component(out_dir "${arg_OUTPUT}" DIRECTORY)
    add_custom_command(OUTPUT "${arg_OUTPUT}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${out_dir}"
        COMMAND ${command} -o "${arg_OUTPUT}" ${arg_PROGRAMS}
        DEPENDS ${arg_DEPENDS} "${arg_SLOTS}" "${CONDUIT_PACK_SHADERS}"
        COMMENT "Packing shaders into ${arg_OUTPUT} (${targets})"
        VERBATIM)
    add_custom_target(${name} DEPENDS "${arg_OUTPUT}")
endfunction()
