FROM python:3.10

WORKDIR /workspace

COPY ./requirements.txt /workspace
RUN python -m pip install -r requirements.txt

CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--allow-root"]