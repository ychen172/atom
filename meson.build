# Project
project('atom', 'cpp', 'cython', version: '0.0.1', meson_version: '>= 1.11', default_options: ['cpp_std=c++17'])

# Tools and dependencies
py = import('python').find_installation(pure: false) #contain compiled compiled code
cantera_dep = dependency('cantera')

# Include folders
inc = include_directories('src')

# Get c++ into static library
atom_core = static_library('atomcore',
                           'src/thermTrans.cpp',
                           include_directories: inc,
                           dependencies: cantera_dep)

# cython
py.extension_module('_atom',
                    'python/atom/_atom.pyx',
                    override_options: ['cython_language=cpp'],
                    include_directories: inc,
                    link_with: atom_core,
                    dependencies: cantera_dep,
                    install: true,
                    subdir: 'atom')

# install the pure python header
py.install_sources('python/atom/__init__.py', subdir: 'atom')