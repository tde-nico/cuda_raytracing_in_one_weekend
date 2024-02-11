# cuda_raytracing_in_one_weekend

[Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) in cuda.

## Usage

Compile
```
> make
```

Run
```
> make run
```

Compile Docs
```
> make latex
```


## Attunement

In order to change the setting, open includes/raytracer.cuh, you'll find the settings defined as macros.
In order to change the scenery, open srcs/main.cu, you'll find the function "create_world" where all the objects are defined.


## Data

There are also some samples of possible image outputs in the data folder, and also some sources to recreate them.
