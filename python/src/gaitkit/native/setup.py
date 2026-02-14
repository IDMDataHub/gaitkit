from setuptools import Extension, setup


ext_modules = [
    Extension(
        name="_gait_native",
        sources=["_gait_native.c"],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="recode-native",
    version="0.1.0",
    description="Native C helpers for gait detection",
    ext_modules=ext_modules,
)
