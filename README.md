# components-of-an-ml-system
This repo contains a notebook that demonstrates components of production ML systems using simple Python.

To run:
```bash
docker build -t components-of-an-ml-system .
docker run -it -v $(pwd):/workspace -p 8888:8888 components-of-an-ml-system 
```